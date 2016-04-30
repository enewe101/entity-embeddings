import re
import random
import t4k
import itertools as itools
from collections import defaultdict
from multiprocessing import Queue, Process, Pipe
from Queue import Empty
from word2vec import CorpusReader, UnigramDictionary
import numpy as np
import gzip
import os
from word2vec.token_map import UNK

TAB_SPLITTER = re.compile(r'\t+')

def parse(filename):
	tokenized_sentences = []
	for line in open(filename):

		# skip empty lines
		if line.strip() == '':
			continue

		# Note that in cases where there are no entity spans, we get
		# two tab delimiters in a row.  This tab splitter will consider
		# any number of consecutive tabs as a single delimiter
		fields = TAB_SPLITTER.split(line.strip())
		num_unique, tokens = fields[0], fields[1]
		entity_spans, fname = fields[2:-1], fields[-1]
		tokens = tokens.split()

		# We'll parse out the entity spans.  They consist of an entity id
		# followed by the start and stop index of the span.  A given 
		# entity may appear multiple times, and so have multiple spans, 
		# so for each entity we store a list of span tuples
		parsed_spans = defaultdict(list)
		for entity_span in entity_spans:
			try:
				entity_id, start, stop = entity_span.rsplit(',', 2)
			except ValueError:
				print line
				print 'num entity spans:', len(entity_spans)
			span = (int(start), int(stop))
			parsed_spans[entity_id].append(span)

		tokenized_sentences.append((tokens, parsed_spans))

	return tokenized_sentences


class MinibatchGenerator(object):

	NOT_DONE = 0
	DONE = 1

	def __init__(
		self,
		files=[],
		directories=[],
		skip=[],
		entity_dictionary=None,
		context_dictionary=None,
		noise_ratio=15,
		batch_size = 1000,
		parse=parse,
		verbose=True
	):

		# Get a corpus reader
		self.corpus_reader = CorpusReader(
			files=files, directories=directories, skip=skip, parse=parse,
			verbose=verbose
		)

		# Load the dictionary, if supplied
		if entity_dictionary is not None:
			self.entity_dictionary = entity_dictionary
		else:
			self.entity_dictionary = UnigramDictionary()

		# Load the context dictionary, if supplied
		if context_dictionary:
			self.context_dictionary = context_dictionary
		else:
			self.context_dictionary = UnigramDictionary()

		self.noise_ratio = noise_ratio
		self.batch_size = batch_size


	def load(self, directory):
		'''
		Load both the dictionary and context_dictionary, assuming default filenames
		(dictionary.gz and unigram-dictionary.gz), by specifying their containing
		directory
		'''
		self.entity_dictionary.load(os.path.join(
			directory, 'entity-dictionary'
		))
		self.context_dictionary.load(os.path.join(
			directory, 'context-dictionary'
		))
	
	def load_entity_dictionary(self, filename):
		self.entity_dictionary.load(filename)

	def load_context_dictionary(self, filename):
		self.context_dictionary.load(filename)


	def save(self, directory):
		'''
		Save both the dictionary and context_dictionary, using default filenames
		(dictionary.gz and unigram-dictionary.gz), by specifying only their containing
		directory
		'''
		self.entity_dictionary.save(
			os.path.join(directory, 'entity-dictionary'))
		self.context_dictionary.save(
			os.path.join(directory, 'context-dictionary'))


	def save_entity_dictionary(self, filename):
		self.entity_dictionary.save(filename)

	def save_context_dictionary(self, filename):
		self.context_dictionary.save(filename)

	def check_access(self, savedir):

		savedir = os.path.abspath(savedir)
		path, dirname = os.path.split(savedir)

		# Make sure that the directory we want exists (make it if not)
		if not os.path.isdir(path):
			raise IOError('%s is not a directory or does not exist' % path)
		if not os.path.exists(savedir):
			os.mkdir(savedir)
		elif os.path.isfile(savedir):
			raise IOError('%s is a file. % savedir')

		# Make sure we can write to the file
		f = open(os.path.join(savedir, '.__test-w2v-access'), 'w')
		f.write('test')
		f.close
		os.remove(os.path.join(savedir, '.__test-w2v-access'))


	def prepare(self, savedir=None):
		# Before any minibatches can be generated, we need to run over
		# the corpus to determine the context_dictionary distribution and create
		# a dictionary mapping all words in the corpus vocabulary to int's.

		# But first, if a savedir was supplied do an IO check
		if savedir is not None:
			self.check_access(savedir)

		# For each line, get the context tokens and entity tokens.
		# Add both to the respective dictionaries.  Also add the context
		# tokens (after converting them to ids) to the context_dictionary noise model
		for line in self.corpus_reader.read_no_q():
			context_tokens, entity_spans = line
			self.context_dictionary.update(context_tokens)
			self.entity_dictionary.update(entity_spans.keys())

		# save the dictionaries and context_dictionary noise model
		if savedir is not None:
			self.save(savedir)


	def prune(self, min_frequency=5):
		'''
		Exposes the prune function for the underlying UnigramDictionary
		used for the context_dictionary.
		'''
		self.context_dictionary.prune(min_frequency)
		self.entity_dictionary.prune(min_frequency)


	def __iter__(self):

		# Once iter is called, a subprocess will be started which
		# begins generating minibatches.  These accumulate in a queue
		# and iteration pulls from that queue.  That way, iteration
		# can begin as soon as the first minibatch is prepared, and 
		# later minibatches are prepared in the background while earlier
		# minibatches are used.  The idea is that this will keep the 
		# CPU(s) busy while training occurs on the GPU.

		# TODO: enable having multiple reader processes.  This could
		# provide a speed up for clusters with distributed IO

		self.minibatches = Queue()
		self.recv_pipe, send_pipe = Pipe()

		# We'll fork a process to assemble minibatches, and return 
		# immediatetely so that minibatches can be used as they are 
		# constructed.

		# TODO: currently the only randomness in minibatching comes from
		# the signal context and noise contexts that are drawn for a 
		# given entity query tuple.  But the entity query tuples are read
		# deterministically in order through the corpus  Ideally examples
		# should be totally shuffled..

		minibatch_preparation = Process(
			target=self.enqueue_minibatches,
			args=(self.minibatches, send_pipe)
		)
		minibatch_preparation.start()

		# Because we assemble the batches within a forked process, it's 
		# access to randomness doesn't alter the state of the parent's 
		# random number generator.  Multiple calls to this function
		# would produce the same minibatching, which is not
		# desired.  We make a call to the numpy random number generator
		# to advance the parent's random number generator's state to avoid
		# this problem:
		np.random.uniform()

		return self


	def init_batch(self):
		# Initialize np.array's to store the minibatch data.  We know
		# how big the batch is ahead of time.  Initialize by filling
		# the arrays with UNK tokens.  Doing this means that, at the end
		# of the corpus, when we don't necessarily have a full minibatch,
		# the final minibatch is padded with UNK tokens in order to be
		# of the desired shape.  This has no effect on training, because
		# we don't care about the embedding of the UNK token
		signal_batch = np.full(
			(self.batch_size, 3),
			UNK,
			dtype='int32'
		)
		noise_batch = np.full(
			(self.batch_size * self.noise_ratio, 3),
			UNK,
			dtype='int32'
		)
		return signal_batch, noise_batch


	def eliminate_spans(self, token_ids, spans):
		'''
		Return the list of token_ids, but with the tokens that are 
		part of entity spans removed.  The entity_spans are listed as
		(start, stop) tuples in spans using coreNLP indexing convention.
		In that convention, indexing starts from 1, and the tuple
		(1,2) designates a span including the first and second token  
		(note that this is different from Python slice indexing in which 
		the stop token is not actually included).
		'''

		# Convert spans to Python slice notation, then delegate to 
		# t4k's skip function.
		adjusted_spans = []
		for start, stop in spans:
			adjusted_spans.append((start-1, stop))

		return t4k.skip(token_ids, adjusted_spans)


	def generate(self):
		'''
		This iterates minibatches.  You can call this directly to provide
		an iterable to loop through the dataset.  However, the 
		MinibatchGenerator is itself iterable, so you can provide a 
		MinibatchGenerator instance as the iterable in a looping construct.

		These are not quite equivalent approaches.  Using a 
		MinibatchGenerator instance starts a process in the background
		that reads through the corpus and enqueues minibatches,
		whereas using .generate() produces each minibatch as it is 
		requested.  
		
		If the production of minibatches takes a similar 
		amounts of time as their consumption, then passing the 
		MinibatchGenerator instance will be faster.

		If the consumption of minibatches takes longer, then this 
		approach could fill up all the memory. (see TODO below).

		If minibatches are consumed much faster than they are produced,
		then it will make little difference which approach you use.
		'''
		# TODO: add some controls to avoid filling up all of the 
		# 	memory by enqueing too many minibatches

		signal_batch, noise_batch = self.init_batch()

		# i keeps track of position in the signal batch
		i = -1
		for line in self.corpus_reader.read_no_q():

			context_tokens, entity_spans = line

			# Sentences with less than two entities can't be used for 
			# learning
			if len(entity_spans) < 2:
				continue

			token_ids = self.context_dictionary.get_ids(context_tokens)

			# We'll now generate generate signal examples and noise
			# examples for training.  Iterate over every pairwise 
			# of entities in this line
			for e1, e2 in itools.combinations(entity_spans, 2):

				# Increment position within the batch
				i += 1

				# TODO test this
				# Get the context tokens minus the entity_spans
				filtered_token_ids = self.eliminate_spans(
					token_ids, entity_spans[e1] + entity_spans[e2]
				)

				# Sample a token from the context
				context_token_id = np.random.choice(filtered_token_ids, 1)[0]

				# convert entities into ids
				e1_id, e2_id = self.entity_dictionary.get_ids([e1, e2])

				# Add the signal example
				signal_batch[i, :] = [e1_id, e2_id, context_token_id]

				# Sample tokens from the noise
				noise_context_ids = self.context_dictionary.sample(
					(self.noise_ratio,))

				# Figure out the position within the noise batch
				j = i*self.noise_ratio

				# block-assign the noise samples to the noise batch array
				noise_batch[j:j+self.noise_ratio, :] = [
					[e1_id, e2_id, noise_context_id]
					for noise_context_id in noise_context_ids
				]

				# Once we've finished assembling a minibatch, enqueue it
				# and start assembling a new minibatch
				if i == self.batch_size - 1:
					yield (signal_batch, noise_batch)
					signal_batch, noise_batch = self.init_batch()
					i = -1

		# Normally we'll have a partially filled minibatch after processing
		# the corpus.  The elements in the batch that weren't overwritten
		# contain UNK tokens, which act as padding.  Yield the partial
		# minibatch.
		if i >= 0:
			yield (signal_batch, noise_batch)


	def enqueue_minibatches(self, minibatch_queue, send_pipe):

		# Continuously iterate through the dataset, enqueing each
		# minibatch.  The consumer will process minibatches from
		# the queue at it's own pace.
		for signal_batch, noise_batch in self.generate():
			minibatch_queue.put((signal_batch, noise_batch))

		# Notify parent process that iteration through the corpus is
		# complete (so it doesn't need to wait for more minibatches)
		send_pipe.send(self.DONE)


	def next(self):
		status = self.NOT_DONE
		while status == self.NOT_DONE:
			try:
				return self.minibatches.get(timeout=0.1)
			except Empty:
				if self.recv_pipe.poll():
					status = self.recv_pipe.recv()

		raise StopIteration

	def entity_vocab_size(self):
		return len(self.entity_dictionary)

	def context_vocab_size(self):
		return len(self.context_dictionary)

				

