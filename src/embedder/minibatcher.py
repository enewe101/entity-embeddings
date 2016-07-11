from iterable_queue import IterableQueue
import re
import random
import t4k
import itertools as itools
from collections import defaultdict
from multiprocessing import Queue, Process, Pipe
from Queue import Empty
from word2vec import (
	Word2VecMinibatcher as _Word2VecMinibatcher, Minibatcher,
	UnigramDictionary
)
import numpy as np
import gzip
import os
from word2vec.token_map import UNK

TAB_SPLITTER = re.compile(r'\t+')

def word2vec_parse(filename):

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
		tokens = tokens.split()

		tokenized_sentences.append(tokens)

	return tokenized_sentences


def relation2vec_parse(filename):

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


# We need to slightly change the parser used in the word2vec minibatcher
# in order to handle the file format in this project
class Word2VecMinibatcher(_Word2VecMinibatcher):
	def parse(self, filename):
		return word2vec_parse(filename)



#class Relation2VecMinibatcher(object):
#
#	NOT_DONE = 0
#	DONE = 1
#
#	def __init__(
#		self,
#		files=[],
#		directories=[],
#		skip=[],
#		entity_dictionary=None,
#		context_dictionary=None,
#		noise_ratio=15,
#		batch_size = 1000,
#		verbose=True,
#		num_example_generators=10,
#	):
#
#		self.num_example_generators = num_example_generators
#		self.verbose = verbose
#		self.files = files
#		self.directories = directories
#		self.skip = skip
#
#		# Load the dictionary, if supplied
#		if entity_dictionary is not None:
#			self.entity_dictionary = entity_dictionary
#		else:
#			self.entity_dictionary = UnigramDictionary()
#
#		# Load the context dictionary, if supplied
#		if context_dictionary:
#			self.context_dictionary = context_dictionary
#		else:
#			self.context_dictionary = UnigramDictionary()
#
#		self.noise_ratio = noise_ratio
#		self.batch_size = batch_size
#
#
#	def parse(self, filename):
#		return relation2vec_parse(filename)
#
#	def load(self, directory):
#		'''
#		Load both the dictionary and context_dictionary, assuming default 
#		filenames (dictionary.gz and unigram-dictionary.gz), by specifying 
#		their containing directory.
#		'''
#		self.entity_dictionary.load(os.path.join(
#			directory, 'entity-dictionary'
#		))
#		self.context_dictionary.load(os.path.join(
#			directory, 'context-dictionary'
#		))
#	
#	def load_entity_dictionary(self, filename):
#		self.entity_dictionary.load(filename)
#
#	def load_context_dictionary(self, filename):
#		self.context_dictionary.load(filename)
#
#
#	def save(self, directory):
#		'''
#		Save both the dictionary and context_dictionary, using default 
#		filenames (dictionary.gz and unigram-dictionary.gz), by specifying 
#		only their containing directory
#		'''
#		self.entity_dictionary.save(
#			os.path.join(directory, 'entity-dictionary'))
#		self.context_dictionary.save(
#			os.path.join(directory, 'context-dictionary'))
#
#
#	def save_entity_dictionary(self, filename):
#		self.entity_dictionary.save(filename)
#
#	def save_context_dictionary(self, filename):
#		self.context_dictionary.save(filename)
#
#	def check_access(self, savedir):
#
#		savedir = os.path.abspath(savedir)
#		path, dirname = os.path.split(savedir)
#
#		# Make sure that the directory we want exists (make it if not)
#		if not os.path.isdir(path):
#			raise IOError('%s is not a directory or does not exist' % path)
#		if not os.path.exists(savedir):
#			os.mkdir(savedir)
#		elif os.path.isfile(savedir):
#			raise IOError('%s is a file. % savedir')
#
#		# Make sure we can write to the file
#		f = open(os.path.join(savedir, '.__test-w2v-access'), 'w')
#		f.write('test')
#		f.close
#		os.remove(os.path.join(savedir, '.__test-w2v-access'))
#
#	def preparation(self, savedir):
#		# For each line, get the context tokens and entity tokens.
#		# Add both to the respective dictionaries.  Also add the context
#		# tokens (after converting them to ids) to the context_dictionary 
#		# noise model
#		for filename in self.generate_filenames():
#			for line in self.parse(filename):
#				context_tokens, entity_spans = line
#				self.context_dictionary.update(context_tokens)
#				self.entity_dictionary.update(entity_spans.keys())
#
#
#	def prepare(self, savedir=None, *args, **kwargs):
#		'''
#		Used to perform any preparation steps that are needed before
#		minibatching can be done.  E.g. assembling a dictionary that
#		maps tokens to integers, and determining the total vocabulary size
#		of the corpus.  It is assumed that files will need
#		to be saved as part of this process, and that they should be
#		saved under `savedir`, with `self.save()` managing the details 
#		of writing files under `savedir`.
#
#		INPUTS
#
#		* Note About Inputs *
#		the call signature of this method is variable and is
#		determined by the call signature of the core 
#		`self.preparation()` method.  Refer to that method's call 
#		signature.  Minimally, this method accepts `savedir`
#
#		* savedir [str]: path to directory in which preparation files 
#			should be saved.
#
#		RETURNS
#		* [None]
#		'''
#		# Before any minibatches can be generated, we need to run over
#		# the corpus to determine the context_dictionary distribution and 
#		# create a dictionary mapping all words in the corpus vocabulary to 
#		# int's.
#
#		# But first, if a savedir was supplied do an IO check
#		if savedir is not None:
#			self.check_access(savedir)
#
#		self.preparation(savedir, *args, **kwargs)
#
#		# save the dictionaries and context_dictionary noise model
#		if savedir is not None:
#			self.save(savedir)
#
#
#	def prune(self, min_frequency=5):
#		'''
#		Exposes the prune function for the underlying UnigramDictionary
#		used for the context_dictionary.
#		'''
#		self.context_dictionary.prune(min_frequency)
#		self.entity_dictionary.prune(min_frequency)
#
#
#	def batch_examples(self, example_iterator):
#
#		signal_batch, noise_batch = self.init_batch()
#
#		# i keeps track of position in the signal batch
#		i = -1
#		for signal_example, noise_examples in example_iterator:
#
#			# Increment position within the batch
#			i += 1
#
#			# Add the signal example
#			signal_batch[i, :] = signal_example
#
#			# Figure out the position within the noise batch
#			j = i*self.noise_ratio
#
#			# block-assign the noise samples to the noise batch array
#			noise_batch[j:j+self.noise_ratio, :] = noise_examples
#
#			# Once we've finished assembling a minibatch, enqueue it
#			# and start assembling a new minibatch
#			if i == self.batch_size - 1:
#				yield (signal_batch, noise_batch)
#				signal_batch, noise_batch = self.init_batch()
#				i = -1
#
#		# Normally we'll have a partially filled minibatch after processing
#		# the corpus.  The elements in the batch that weren't overwritten
#		# contain UNK tokens, which act as padding.  Yield the partial
#		# minibatch.
#		if i >= 0:
#			yield (signal_batch, noise_batch)
#
#
#	def generate_minibatches(self):
#		for minibatch in self.batch_examples(self.generate_examples()):
#			yield minibatch
#
#
#	def generate_examples(self):
#		for filename in self.generate_filenames():
#			for example in self.process_file(filename):
#				yield example
#
#
#	def generate_minibatches_async(self, example_queue, minibatch_queue):
#		for minibatch in self.batch_examples(example_queue):
#			minibatch_queue.put(minibatch)
#		minibatch_queue.close()
#
#
#	def __iter__(self):
#
#		# TODO: enable having multiple reader processes.  This could
#		# provide a speed up for clusters with distributed IO
#		# TODO: currently the only randomness in minibatching comes from
#		# the signal context and noise contexts that are drawn for a 
#		# given entity query tuple.  But the entity query tuples are read
#		# deterministically in order through the corpus  Ideally examples
#		# should be totally shuffled..
#
#		file_queue = IterableQueue()
#		example_queue = IterableQueue()
#		minibatch_queue = IterableQueue()
#
#		# Fill the file queue
#		file_producer = file_queue.get_producer()
#		for filename in self.generate_filenames():
#			file_producer.put(filename)
#		file_producer.close()
#
#		# Make processes that process the files and put examples onto
#		# the example queue
#		for i in range(self.num_example_generators):
#			Process(target=self.process_file_async, args=(
#				file_queue.get_consumer(),
#				example_queue.get_producer()
#			)).start()
#			
#		# Make a processes that batches the files and puts examples onto
#		# the minibatch queue
#		Process(target=self.generate_minibatches_async, args=(
#			example_queue.get_consumer(),
#			minibatch_queue.get_producer()
#		)).start()
#
#		# Before closing the queues, make a consumer that will be used for 
#		# yielding minibatches to the external call for iteration.
#		self.minibatch_consumer = minibatch_queue.get_consumer()
#
#		# Close all queues
#		file_queue.close()
#		example_queue.close()
#		minibatch_queue.close()
#
#		# This is necessary because accessing randomness in the child 
#		# processes doesn't advance the random state here in the parent
#		# process, which would, mean that the exact same minibatch sequence 
#		# would being generated on subsequent calls to `__iter__()`, which 
#		# is not desired.  The simplest solution is to advance the 
#		# random state by sampling randomness once.
#		np.random.uniform()
#
#		# Return the minibatch_consumer as the iterator
#		return self.minibatch_consumer
#		
#
#	def process_file_async(self, file_queue, example_queue):
#		for filename in file_queue:
#			for example in self.process_file(filename):
#				example_queue.put(example)
#
#		example_queue.close()
#
#
#	def process_file(self, filename):
#		'''
#		Generator that yields training examples.  Accepts a filename, which
#		is read and parsed into a file-format-independant form by 
#		`self.parse`, and then is used to generate training examples.
#
#		INPUTS
#		* filename [str]: path to file to be processed.
#
#		YIELDS
#		* example [any]: object representing a training example.
#		'''
#
#		parsed = self.parse(filename)
#		examples = self.build_examples(parsed)
#		for example in examples:
#			yield example
#
#
#	def build_examples(self, parsed):
#
#		'''
#		Assembles bunches of examples from the parsed data coming from
#		files that were read.  Normally, this function might yield 
#		individual examples, however, in this case, we need to maintain
#		a distinction between the noise- and signal-examples, and to
#		keep them in consistent proportions.  So, here, we yield small 
#		bunches that consist of 1 signal example, and X noise examples,
#		where X depends on `self.noise_ratio`.
#		'''
#
#		for line in parsed:
#
#			context_tokens, entity_spans = line
#
#			# Sentences with less than two entities can't be used for 
#			# learning
#			if len(entity_spans) < 2:
#				continue
#
#			token_ids = self.context_dictionary.get_ids(context_tokens)
#
#			# We'll now generate generate signal examples and noise
#			# examples for training.  Iterate over every pairwise 
#			# of entities in this line
#			for e1, e2 in itools.combinations(entity_spans, 2):
#
#				# TODO test this
#				# Get the context tokens minus the entity_spans
#				filtered_token_ids = self.eliminate_spans(
#					token_ids, entity_spans[e1] + entity_spans[e2]
#				)
#
#				# We can't train if there are no context words
#				if len(filtered_token_ids) == 0:
#					break
#
#				# Sample a token from the context
#				context_token_id = np.random.choice(
#					filtered_token_ids, 1)[0]
#
#				# convert entities into ids
#				e1_id, e2_id = self.entity_dictionary.get_ids([e1, e2])
#
#				# Add the signal example
#				signal_example = [e1_id, e2_id, context_token_id]
#
#				# Sample tokens from the noise
#				noise_context_ids = self.context_dictionary.sample(
#					(self.noise_ratio,))
#
#				# block-assign the noise samples to the noise batch array
#				noise_examples = [
#					[e1_id, e2_id, noise_context_id]
#					for noise_context_id in noise_context_ids
#				]
#
#				# Yield the example
#				yield (signal_example, noise_examples)
#
#
#	def generate_filenames(self):
#		'''
#		Generator that yields all the filenames (absolute paths) that 
#		make up the corpus.  Files might have been specified as a list
#		of directories and/or as a list of files.  This process consults 
#		the filesystem and resolves them to a list of files.  Also,
#		the `skip` argument can provide a list of regexes matching files
#		and directories to ignore, so this filters out files / directories
#		that match an entry in `skip`.
#		'''
#
#		# Process all the files listed in files, unles they match an
#		# entry in skip
#		#print 'starting reading'
#		if self.files is not None:
#			for filename in self.files:
#				filename = os.path.abspath(filename)
#
#				# Skip files if they match a regex in skip
#				if any([s.search(filename) for s in self.skip]):
#					continue
#
#				if self.verbose:
#					print 'processing', filename
#
#				yield filename
#
#		# Process all the files listed in each directory, unless they
#		# match an entry in skip
#		if self.directories is not None:
#			for dirname in self.directories:
#				dirname = os.path.abspath(dirname)
#
#				# Skip directories if they match a regex in skip
#				if any([s.search(dirname) for s in self.skip]):
#					continue
#
#				for filename in os.listdir(dirname):
#					filename = os.path.join(dirname, filename)
#
#					# Only process the *files* under the given directories
#					if not os.path.isfile(filename):
#						continue
#
#					# Skip files if they match a regex in skip
#					if any([s.search(filename) for s in self.skip]):
#						continue
#
#					if self.verbose:
#						print 'processing', filename
#
#					yield filename
#
#
#	def init_batch(self):
#		# Initialize np.array's to store the minibatch data.  We know
#		# how big the batch is ahead of time.  Initialize by filling
#		# the arrays with UNK tokens.  Doing this means that, at the end
#		# of the corpus, when we don't necessarily have a full minibatch,
#		# the final minibatch is padded with UNK tokens in order to be
#		# of the desired shape.  This has no effect on training, because
#		# we don't care about the embedding of the UNK token
#		signal_batch = np.full(
#			(self.batch_size, 3),
#			UNK,
#			dtype='int32'
#		)
#		noise_batch = np.full(
#			(self.batch_size * self.noise_ratio, 3),
#			UNK,
#			dtype='int32'
#		)
#		return signal_batch, noise_batch
#
#
#	def eliminate_spans(self, token_ids, spans):
#		'''
#		Return the list of token_ids, but with the tokens that are 
#		part of entity spans removed.  The entity_spans are listed as
#		(start, stop) tuples in spans using coreNLP indexing convention.
#		In that convention, indexing starts from 1, and the tuple
#		(1,2) designates a span including the first and second token  
#		(note that this is different from Python slice indexing in which 
#		the stop token is not actually included).
#		'''
#
#		# Convert spans to Python slice notation, then delegate to 
#		# t4k's skip function.
#		adjusted_spans = []
#		for start, stop in spans:
#			adjusted_spans.append((start-1, stop))
#
#		return t4k.skip(token_ids, adjusted_spans)
#
#
#
#	#def generate(self):
#	#	'''
#	#	This iterates minibatches.  You can call this directly to provide
#	#	an iterable to loop through the dataset.  However, the 
#	#	Relation2VecMinibatcher is itself iterable, so you can provide a 
#	#	Relation2VecMinibatcher instance as the iterable in a looping construct.
#
#	#	These are not quite equivalent approaches.  Using a 
#	#	Relation2VecMinibatcher instance starts a process in the background
#	#	that reads through the corpus and enqueues minibatches,
#	#	whereas using .generate() produces each minibatch as it is 
#	#	requested.  
#	#	
#	#	If the production of minibatches takes a similar 
#	#	amounts of time as their consumption, then passing the 
#	#	Relation2VecMinibatcher instance will be faster.
#
#	#	If the consumption of minibatches takes longer, then this 
#	#	approach could fill up all the memory. (see TODO below).
#
#	#	If minibatches are consumed much faster than they are produced,
#	#	then it will make little difference which approach you use.
#	#	'''
#	#	# TODO: add some controls to avoid filling up all of the 
#	#	# 	memory by enqueing too many minibatches
#
#	#	signal_batch, noise_batch = self.init_batch()
#
#	#	# i keeps track of position in the signal batch
#	#	i = -1
#	#	for line in self.corpus_reader.read_no_q():
#
#	#		context_tokens, entity_spans = line
#
#	#		# Sentences with less than two entities can't be used for 
#	#		# learning
#	#		if len(entity_spans) < 2:
#	#			continue
#
#	#		token_ids = self.context_dictionary.get_ids(context_tokens)
#
#	#		# We'll now generate generate signal examples and noise
#	#		# examples for training.  Iterate over every pairwise 
#	#		# of entities in this line
#	#		for e1, e2 in itools.combinations(entity_spans, 2):
#
#	#			# Increment position within the batch
#	#			i += 1
#
#	#			# TODO test this
#	#			# Get the context tokens minus the entity_spans
#	#			filtered_token_ids = self.eliminate_spans(
#	#				token_ids, entity_spans[e1] + entity_spans[e2]
#	#			)
#	#			
#	#			# We can't train if there are no context words
#	#			if len(filtered_token_ids) == 0:
#	#				break
#
#	#			# Sample a token from the context
#	#			context_token_id = np.random.choice(filtered_token_ids, 1)[0]
#
#	#			# convert entities into ids
#	#			e1_id, e2_id = self.entity_dictionary.get_ids([e1, e2])
#
#	#			# Add the signal example
#	#			signal_batch[i, :] = [e1_id, e2_id, context_token_id]
#
#	#			# Sample tokens from the noise
#	#			noise_context_ids = self.context_dictionary.sample(
#	#				(self.noise_ratio,))
#
#	#			# Figure out the position within the noise batch
#	#			j = i*self.noise_ratio
#
#	#			# block-assign the noise samples to the noise batch array
#	#			noise_batch[j:j+self.noise_ratio, :] = [
#	#				[e1_id, e2_id, noise_context_id]
#	#				for noise_context_id in noise_context_ids
#	#			]
#
#	#			# Once we've finished assembling a minibatch, enqueue it
#	#			# and start assembling a new minibatch
#	#			if i == self.batch_size - 1:
#	#				yield (signal_batch, noise_batch)
#	#				signal_batch, noise_batch = self.init_batch()
#	#				i = -1
#
#	#	# Normally we'll have a partially filled minibatch after processing
#	#	# the corpus.  The elements in the batch that weren't overwritten
#	#	# contain UNK tokens, which act as padding.  Yield the partial
#	#	# minibatch.
#	#	if i >= 0:
#	#		yield (signal_batch, noise_batch)
#
#
#	def get_minibatches(self):
#		'''
#		Reads through the entire corpus, generating all of the minibatches
#		up front, storing them in memory as a list.  Returns the list of
#		minibatches.
#		'''
#		minibatches = []
#		for minibatch in self.generate_minibatches():
#			minibatches.append(minibatch)
#
#		return minibatches
#
#
#	#def enqueue_minibatches(self, minibatch_queue, send_pipe):
#
#	#	'''
#	#	Reads through the minibatches, placing them on a queue as they
#	#	are ready.  This usually shouldn't be called directly, but 
#	#	is used when the Relation2VecMinibatcher is treated as an iterator, e.g.:
#
#	#		for signal, noise in my_minibatch_generator:
#	#			do_something_with(signal, noise)
#
#	#	It causes the minibatches to be prepared in a separate process
#	#	using this function, placing them on a queue, while a generator
#	#	construct pulls them off the queue as the client process requests
#	#	them.  This keeps minibatch preparation running in the background
#	#	while the client process is busy processing previously yielded 
#	#	minibatches.
#	#	'''
#
#	#	# Continuously iterate through the dataset, enqueing each
#	#	# minibatch.  The consumer will process minibatches from
#	#	# the queue at it's own pace.
#	#	for signal_batch, noise_batch in self.generate():
#	#		minibatch_queue.put((signal_batch, noise_batch))
#
#	#	# Notify parent process that iteration through the corpus is
#	#	# complete (so it doesn't need to wait for more minibatches)
#	#	send_pipe.send(self.DONE)
#
#
#	def entity_vocab_size(self):
#		return len(self.entity_dictionary)
#
#	def context_vocab_size(self):
#		return len(self.context_dictionary)

				

class Relation2VecMinibatcher(Minibatcher):

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
		num_example_generators=10,
		verbose=True,
	):

		super(Relation2VecMinibatcher, self).__init__(
			files,
			directories,
			skip,
			batch_size,
			num_example_generators,
			verbose
		)

		# Register the variable not already registered by `super()`
		self.noise_ratio = noise_ratio

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


	def parse(self, filename):
		return relation2vec_parse(filename)


	def entity_vocab_size(self):
		return len(self.entity_dictionary)


	def context_vocab_size(self):
		return len(self.context_dictionary)


	def load(self, directory):
		'''
		Load both the dictionary and context_dictionary, assuming default 
		filenames (dictionary.gz and unigram-dictionary.gz), by specifying 
		their containing directory.
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
		Save both the dictionary and context_dictionary, using default 
		filenames (dictionary.gz and unigram-dictionary.gz), by specifying 
		only their containing directory
		'''
		self.entity_dictionary.save(
			os.path.join(directory, 'entity-dictionary'))
		self.context_dictionary.save(
			os.path.join(directory, 'context-dictionary'))


	def save_entity_dictionary(self, filename):
		self.entity_dictionary.save(filename)


	def save_context_dictionary(self, filename):
		self.context_dictionary.save(filename)


	def preparation(self, savedir):
		# For each line, get the context tokens and entity tokens.
		# Add both to the respective dictionaries.  Also add the context
		# tokens (after converting them to ids) to the context_dictionary 
		# noise model
		for filename in self.generate_filenames():
			for line in self.parse(filename):
				context_tokens, entity_spans = line
				self.context_dictionary.update(context_tokens)
				self.entity_dictionary.update(entity_spans.keys())


	def prune(self, min_frequency=5):
		'''
		Exposes the prune function for the underlying UnigramDictionary
		used for the context_dictionary.
		'''
		self.context_dictionary.prune(min_frequency)
		self.entity_dictionary.prune(min_frequency)


	def batch_examples(self, example_iterator):

		signal_batch, noise_batch = self.init_batch()

		# i keeps track of position in the signal batch
		i = -1
		for signal_example, noise_examples in example_iterator:

			# Increment position within the batch
			i += 1

			# Add the signal example
			signal_batch[i, :] = signal_example

			# Figure out the position within the noise batch
			j = i*self.noise_ratio

			# block-assign the noise samples to the noise batch array
			noise_batch[j:j+self.noise_ratio, :] = noise_examples

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


	def build_examples(self, parsed):

		'''
		Assembles bunches of examples from the parsed data coming from
		files that were read.  Normally, this function might yield 
		individual examples, however, in this case, we need to maintain
		a distinction between the noise- and signal-examples, and to
		keep them in consistent proportions.  So, here, we yield small 
		bunches that consist of 1 signal example, and X noise examples,
		where X depends on `self.noise_ratio`.
		'''

		for line in parsed:

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

				# TODO test this
				# Get the context tokens minus the entity_spans
				filtered_token_ids = self.eliminate_spans(
					token_ids, entity_spans[e1] + entity_spans[e2]
				)

				# We can't train if there are no context words
				if len(filtered_token_ids) == 0:
					break

				# Sample a token from the context
				context_token_id = np.random.choice(
					filtered_token_ids, 1)[0]

				# convert entities into ids
				e1_id, e2_id = self.entity_dictionary.get_ids([e1, e2])

				# Add the signal example
				signal_example = [e1_id, e2_id, context_token_id]

				# Sample tokens from the noise
				noise_context_ids = self.context_dictionary.sample(
					(self.noise_ratio,))

				# block-assign the noise samples to the noise batch array
				noise_examples = [
					[e1_id, e2_id, noise_context_id]
					for noise_context_id in noise_context_ids
				]

				# Yield the example
				yield (signal_example, noise_examples)


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

