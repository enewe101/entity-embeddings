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
	UnigramDictionary, NoiseSymbolicMinibatcherMixin
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


class Relation2VecMinibatcher(NoiseSymbolicMinibatcherMixin, Minibatcher):

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

		self.noise_ratio = noise_ratio

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

			# We'll now generate signal examples and noise
			# examples for training.  Iterate over every pair
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

				# Assemble the list of noise examples
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


