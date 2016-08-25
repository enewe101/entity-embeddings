from subprocess import check_output
from iterable_queue import IterableQueue
import re
import random
import t4k
import itertools as itools
from collections import defaultdict
from multiprocessing import Queue, Process, Pipe
from Queue import Empty
from word2vec import (
	DatasetReader as Word2VecDatasetReader, 
	DataSetReaderIllegalStateException, #TheanoMinibatcher,
	UnigramDictionary, reseed
)
import numpy as np
import gzip
import os
from word2vec.token_map import UNK

TAB_SPLITTER = re.compile(r'\t+')
RANDOM_SINGLE_CHOICE = 0
FULL_CONTEXT = 1
MAX_LEN_SENTENCE = 80
MAX_ENTITIES_PER_SENTENCE = 10

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


def relation2vec_parse(filename, verbose=True):

	tokenized_sentences = []
	num_skipped = 0
	num_lines = 0
	for line in open(filename):

		# skip empty lines
		if line.strip() == '':
			continue

		# Note that in cases where there are no entity spans, we get
		# two tab delimiters in a row.  This tab splitter will consider
		# any number of consecutive tabs as a single delimiter
		fields = TAB_SPLITTER.split(line.strip())
		num_unique, tokens = int(fields[0]), fields[1]
		entity_spans, fname = fields[2:-1], fields[-1]
		tokens = tokens.split()

		# in some cases, due to unexpected text structure, we will have
		# a very long uninterpretable sentence with too many entities.
		# Skip anything with more than 10 unique entities, or more than 
		# 80 words
		num_lines += 1
		if (
			num_unique > MAX_ENTITIES_PER_SENTENCE 
			or len(tokens) > MAX_LEN_SENTENCE
		):
			num_skipped += 1
			continue

		# We'll parse out the entity spans.  They consist of an entity id
		# followed by the start and stop index of the span.  A given
		# entity may appear multiple times, and so have multiple spans,
		# so for each entity we store a list of span tuples
		parsed_spans = defaultdict(list)
		for entity_span in entity_spans:
			entity_id, start, stop = entity_span.rsplit(',', 2)
			span = (int(start), int(stop))
			parsed_spans[entity_id].append(span)

		tokenized_sentences.append((tokens, parsed_spans))

	if verbose:
		print 'num_lines,num_skipped', '=', (num_lines, num_skipped)

	return tokenized_sentences


class EntityPair2VecDatasetReader(Word2VecDatasetReader):

	NOT_DONE = 0
	DONE = 1

	def __init__(
		self,
		files=[],
		directories=[],
		skip=[],
		noise_ratio=15,
		t=1.0,
		num_processes=3,
		query_dictionary=None,
		context_dictionary=None,
		load_dictionary_dir=None,
		max_queue_size=0,
		macrobatch_size=20000,
		parse=relation2vec_parse,
		#signal_sample_mode=RANDOM_SINGLE_CHOICE,
		verbose=True,
	):


		# These configurations, which exist for the base class,
		# are clamped to None here because they don't make sense here.
		unigram_dictionary = None
		kernel = None

		# Call the base class' __init__ function
		super(EntityPair2VecDatasetReader, self).__init__(
			files=files,
			directories=directories,
			skip=skip,
			noise_ratio=noise_ratio,
			t=t,
			num_processes=num_processes,
			unigram_dictionary=unigram_dictionary,
			kernel=kernel,
			max_queue_size=max_queue_size,
			macrobatch_size=macrobatch_size,
			parse=parse,
			verbose=verbose
		)

		# Register the parameters that don't exist for the base class
		query_dictionary=query_dictionary
		context_dictionary=context_dictionary
		load_dictionary_dir=load_dictionary_dir
		#self.signal_sample_mode=signal_sample_mode

		# Usually the dictionaries are made from scratch
		self.query_dictionary = UnigramDictionary()
		self.context_dictionary = UnigramDictionary()

		# But if a dictionary dir was given, load from there
		if load_dictionary_dir is not None:
			if verbose:
				print (
					'Loading dictionary from: %s...' 
					% load_dictionary_dir
				)
			self.load_dictionary(load_dictionary_dir)

		# Or, if an existing dictionary was passed in, use it
		if query_dictionary is not None:
			if verbose:
				print 'An entity dictionary was supplied.'
			self.query_dictionary = query_dictionary

		# (same but for context dictionary)
		if context_dictionary:
			if verbose:
				print 'A context dictionary was supplied.'
			self.context_dictionary = context_dictionary

		# Keep track of whether the dictionaries have been prepared
		self.prepared = False
		if load_dictionary_dir is not None:
			self.prepared = True
		elif context_dictionary and query_dictionary:
			self.prepared = True


	def get_vocab_size(self):
		raise NotImplementedError(
			'Relation2VecDatasetReader: does not support '
			'`get_vocab_size()`. use `entity_vocab_size()` or '
			'`context_vocab_size()`.'
		)


	def query_vocab_size(self):
		return len(self.query_dictionary)


	def context_vocab_size(self):
		return len(self.context_dictionary)


	def load_dictionary(self, load_dir):
		'''
		Load both the dictionary and context_dictionary, assuming default
		filenames subfolers (entity-pair-dictionary and context-dictionary)
		'''
		self.query_dictionary.load(os.path.join(
			load_dir, 'query-dictionary'
		))
		self.context_dictionary.load(os.path.join(
			load_dir, 'context-dictionary'
		))


	def load_query_dictionary(self, filename):
		self.query_dictionary.load(filename)


	def load_context_dictionary(self, filename):
		self.context_dictionary.load(filename)


	def save_dictionary(self, save_dir):
		'''
		Save both the dictionary and context_dictionary, using default
		filenames (coungter-sampler.gz and token-map.gz), by specifying
		only their containing directory (save_dir).  `save_dir` will be
		created if it doesn't exist.
		'''

		# Make save_dir if it doesn't exist
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)

		# Delegate saving the dictionaries to the dictionary instances
		self.query_dictionary.save(
			os.path.join(save_dir, 'query-dictionary'))
		self.context_dictionary.save(
			os.path.join(save_dir, 'context-dictionary'))


	def save_query_dictionary(self, filename):
		self.query_dictionary.save(filename)


	def save_context_dictionary(self, filename):
		self.context_dictionary.save(filename)


	def preparation(self, savedir):
		# For each line, get the context tokens and entity tokens.
		# We're treating entity-pairs as the query objects to embed,
		# so generate all pairwise combinations.  Add all pairwise
		# entity combiations and all context tokens to the respective
		# dictionaries
		for filename in self.generate_filenames():
			for line in self.parse(filename):

				context_tokens, entity_spans = line
				entity_pairs = self.get_pairs(entity_spans).keys()

				# Update the entity-pair dictionary and context dictionary
				self.context_dictionary.update(context_tokens)
				self.query_dictionary.update(entity_pairs)


	def get_pairs(self, entity_spans):
		'''
		The input, entity_spans is a dictionary with keys being entity
		canonicalized names, and values being the token spans corresponding
		to references to that entity in a given sentence.  This function
		produces all of the *pairs of entities*, and the 2-tuples of
		their corresponding spans (one set of spans for each entity)
		The entity pairs are guaranteed to be in lexicographic order so
		that every entity-pair has a unique name
		'''

		# Get all the entity pairs.  Enforce a unique ordering
		# of the pair, by arbitrarily using lexicographic order
		entity_pairs = [
			(e1, e2) if e1 < e2 else (e2,e1) for e1, e2 in 
			itools.combinations(entity_spans.keys(),2)
		]

		# Convert the entity pairs to strings, and keep the corresponding
		# spans associated to them
		entity_pair_spans = {
			'%s:::%s' % p : (entity_spans[p[0]], entity_spans[p[1]])
			for p in entity_pairs
		}

		return entity_pair_spans


	def prune(self, min_query_frequency, min_context_frequency):
		'''
		Very similar to baseclass implementation, but allows pruning the
		context and query dictionaries based on different thresholds.
		'''
		if self.verbose:
			print 'prunning dictionaries...'
			print (
				'\t...original vocabularies: entity, context = %d, %d'
				% (
					len(self.context_dictionary),
					len(self.query_dictionary)
			))

		self.context_dictionary.prune(min_context_frequency)
		self.query_dictionary.prune(min_query_frequency)

		if self.verbose:
			print (
				'\t...pruned vocabularies: entity, context = %d, %d'
				% (
					len(self.context_dictionary),
					len(self.query_dictionary)
			))


	def generate_examples(self, filename_iterator):

		num_examples = 0
		for fname in filename_iterator:

			parsed = self.parse(fname)
			for line in parsed:

				context_tokens, entity_spans = line

				# Sentences with less than two entities can't be used for
				# learning
				if len(entity_spans) < 2:
					continue

				entity_pair_spans = self.get_pairs(entity_spans)
				token_ids = self.context_dictionary.get_ids(context_tokens)

				# We'll now generate signal examples and noise
				# examples for training.  Iterate over every pair
				# of entities in this line
				for pair_str in entity_pair_spans:

					signal_examples = (
						self.generate_signal_examples_between(
							pair_str, entity_pair_spans, token_ids
					))

					num_examples += len(signal_examples)

					# Continue if we couldn't pull out any signal examples
					# (e.g. if there aren't enough context tokens).
					if len(signal_examples) == 0:
						continue

					# Generate the noise examples by replacing an entity
					# or the context by random entity or context
					noise_examples = self.generate_noise_examples(
						signal_examples)

					num_examples += len(noise_examples)

					yield (signal_examples, noise_examples)

		if self.verbose:
			print 'num_examples', num_examples


	# TODO: test!
	def generate_signal_examples_between(
		self, entity_pair_str, entity_pair_spans, token_ids
	):
		'''
		Extracts the tokens arising between e1 and e2 as context, and,
		combining these with the entities, produces a signal example
		'''
		
		# convert entity pairs into ids
		entity_pair_id = self.query_dictionary.get_id(
			entity_pair_str)

		# Get the tokens between the entities as context
		context_indices = find_tokens_between_closest_pair(
			*entity_pair_spans[entity_pair_str]
		)
		context_ids = [token_ids[i] for i in context_indices]

		# We can't train if there are no context words
		if len(context_ids) == 0:
			return []

		# We create a single example for each of the context words.
		signal_examples = [
			[entity_pair_id, context_id] for context_id in context_ids
		]

		return signal_examples
			

	def generate_noise_examples(self, signal_examples):
		'''
		Generate the noise examples by replacing an entity or the context 
		by random entity or context.  This is almost identical to the 
		method in the baseclass, but in the base class the query and
		context dictionaries are identical, whereas here we must be sure to
		sample from the context dictionary.
		'''

		noise_examples = []
		for query_token_id, context_token_id in signal_examples:
			noise_examples.extend([
				[query_token_id, self.context_dictionary.sample()]
				for i in range(self.noise_ratio)
			])

		return noise_examples



class Relation2VecDatesetReaderException(Exception):
	pass
class SampleModeException(Relation2VecDatesetReaderException):
	pass


class Relation2VecDatasetReader(Word2VecDatasetReader):

	NOT_DONE = 0
	DONE = 1

	def __init__(
		self,
		files=[],
		directories=[],
		skip=[],
		noise_ratio=15,
		entity_noise_ratio=0.0,
		num_processes=3,
		entity_dictionary=None,
		context_dictionary=None,
		load_dictionary_dir=None,
		max_queue_size=0,
		macrobatch_size=20000,
		signal_sample_mode=RANDOM_SINGLE_CHOICE,
		verbose=True,
		len_context=1,
	):

		# TODO: test that no discarding is occurring
		# TODO: ensure that kernel is not being used to sample signal 
		# context 
		unigram_dictionary = None
		kernel=None
		t = 1.0

		super(Relation2VecDatasetReader, self).__init__(
			files=files,
			directories=directories,
			skip=skip,
			noise_ratio=noise_ratio,
			t=t,
			num_processes=num_processes,
			unigram_dictionary=unigram_dictionary,
			kernel=kernel,
			verbose=verbose
		)

		self.entity_noise_ratio = entity_noise_ratio
		self.max_queue_size = max_queue_size
		self.macrobatch_size = macrobatch_size
		self.signal_sample_mode=signal_sample_mode
		self.len_context = len_context

		# Currently only the signal sample mode "between" can be used
		# with non-unity len_context
		if self.len_context != 1:
			if self.signal_sample_mode != 'between':
				raise SampleModeException(
					'Non-unity len_context can only be used when '
					'signal_sample_mode is "between".'
				)

		# Usually the dictionaries are made from scratch
		self.entity_dictionary = UnigramDictionary()
		self.context_dictionary = UnigramDictionary()

		# But if a dictionary dir was given, load from there
		if load_dictionary_dir is not None:
			if verbose:
				print 'Loading dictionary from: %s...' % load_dictionary_dir
			self.load_dictionary(load_dictionary_dir)

		# Or, if an existing dictionary was passed in, use it
		if entity_dictionary is not None:
			if verbose:
				print 'An entity dictionary was supplied.'
			self.entity_dictionary = entity_dictionary

		# (same but for context dictionary)
		if context_dictionary:
			if verbose:
				print 'A context dictionary was supplied.'
			self.context_dictionary = context_dictionary

		# Keep track of whether the dictionaries have been prepared
		self.prepared = False
		if load_dictionary_dir is not None:
			self.prepared = True
		elif context_dictionary and entity_dictionary:
			self.prepared = True


	def get_vocab_size(self):
		raise NotImplementedError(
			'Relation2VecDatasetReader: does not support '
			'`get_vocab_size()`. use `entity_vocab_size()` or '
			'`context_vocab_size()`.'
		)

	def parse(self, filename):
		return relation2vec_parse(filename, self.verbose)


	def entity_vocab_size(self):
		return len(self.entity_dictionary)


	def context_vocab_size(self):
		return len(self.context_dictionary)


	def load_dictionary(self, load_dir):
		'''
		Load both the dictionary and context_dictionary, assuming default
		filenames (dictionary.gz and unigram-dictionary.gz), by specifying
		their containing load_dir.
		'''
		self.entity_dictionary.load(os.path.join(
			load_dir, 'entity-dictionary'
		))
		self.context_dictionary.load(os.path.join(
			load_dir, 'context-dictionary'
		))


	def load_entity_dictionary(self, filename):
		self.entity_dictionary.load(filename)


	def load_context_dictionary(self, filename):
		self.context_dictionary.load(filename)


	def save_dictionary(self, save_dir):
		'''
		Save both the dictionary and context_dictionary, using default
		filenames (coungter-sampler.gz and token-map.gz), by specifying
		only their containing directory (save_dir).  `save_dir` will be
		created if it doesn't exist.
		'''
		# We will make the save_dir (but not its parents) if it doesn't exist
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)

		# Delegate saving the dictionaries to the dictionary instances
		self.entity_dictionary.save(
			os.path.join(save_dir, 'entity-dictionary'))
		self.context_dictionary.save(
			os.path.join(save_dir, 'context-dictionary'))


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


	def prune(self, min_query_frequency, min_context_frequency):
		'''
		Exposes the prune function for the underlying UnigramDictionary
		used for the context_dictionary.
		'''
		if self.verbose:
			print 'prunning dictionaries...'
			print (
				'\t...original vocabularies: entity, context = %d, %d'
				% (
					len(self.context_dictionary),
					len(self.entity_dictionary)
			))

		self.context_dictionary.prune(min_context_frequency)
		self.entity_dictionary.prune(min_query_frequency)

		if self.verbose:
			print (
				'\t...pruned vocabularies: entity, context = %d, %d'
				% (
					len(self.context_dictionary),
					len(self.entity_dictionary)
			))

	
	def get_padding_row(self):
		return [UNK,UNK] + [UNK] * self.len_context


	def generate_examples(self, filename_iterator):

		num_examples = 0
		for fname in filename_iterator:
			parsed = self.parse(fname)
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

					# Generate signal examples according to sample mode
					if self.signal_sample_mode == RANDOM_SINGLE_CHOICE:	
						signal_examples = self.generate_signal_examples(
							e1, e2, token_ids, entity_spans)

					elif self.signal_sample_mode == FULL_CONTEXT:	
						signal_examples = self.generate_signal_examples(
							e1, e2, token_ids, entity_spans)

					elif self.signal_sample_mode == 'between':
						signal_examples = self.generate_signal_examples_between(
							e1, e2, token_ids, entity_spans)

					else:
						raise SampleModeException(
							'Unrecognized sample mode: %s' 
							% self.sample_mode
						)

					num_examples += len(signal_examples)

					# Continue if we couldn't pull out any signal examples.
					# This can happen if there aren't enough context tokens.
					if len(signal_examples) == 0:
						continue

					# Generate the noise examples by replacing an entity
					# or the context by random entity or context
					noise_examples = self.generate_noise_examples(
						signal_examples)

					num_examples += len(noise_examples)

					yield (signal_examples, noise_examples)

		if self.verbose:
			print 'num_examples', num_examples


	def generate_signal_examples_between(
		self, e1, e2, token_ids, entity_spans
	):
		'''
		Extracts the tokens arising between e1 and e2 as context, and,
		combining these with the entities, produces a signal example
		'''
		

		# convert entities into ids
		e1_id, e2_id = self.entity_dictionary.get_ids([e1, e2])

		# Get the tokens between the entities as context
		context_indices = find_tokens_between_closest_pair(
			entity_spans[e1], entity_spans[e2]
		)
		context = [token_ids[i] for i in context_indices]

		# We can't train if there are no context words
		if len(context) == 0:
			return []

		# We need to ensure a consistent size for the signal examples
		# If the current row is too long, downsample
		if len(context) > self.len_context:
			context = np.random.choice(
				context, self.len_context, replace=False)

		# If the current row is too short, upsample
		elif len(context) < self.len_context:
			context = np.random.choice(
				context, self.len_context, replace=True)

		# We create a single example, but we include all of the context 
		# in that one example. They will be averaged...
		signal_examples = [
			[e1_id, e2_id] + list(context)
		]

		return signal_examples
			

	def generate_signal_examples(self, e1, e2, token_ids, entity_spans):

		# convert entities into ids
		e1_id, e2_id = self.entity_dictionary.get_ids([e1, e2])

		# Get the context tokens minus the entity_spans
		filtered_context_tokens = self.eliminate_spans(
			token_ids, entity_spans[e1] + entity_spans[e2]
		)

		# We can't train if there are no context words
		if len(filtered_context_tokens) == 0:
			return []

		# Add a signal example.  We generate a singal example from a 
		# randomly chosen token:
		if self.signal_sample_mode == RANDOM_SINGLE_CHOICE:
			context = np.random.choice(filtered_context_tokens)
			signal_examples = [[e1_id, e2_id, context]]

		# Or generate many examples by including all context tokens.
		elif self.signal_sample_mode == FULL_CONTEXT:
			signal_examples = [
				[e1_id, e2_id, context] 
				for context in filtered_context_tokens
			]

		return signal_examples


	def generate_noise_examples(self, signal_examples):
		'''
		Generate the noise examples by replacing an entity or the context 
		by random entity or context.
		'''

		noise_examples = []
		for row in signal_examples:

			# This somewhat odd way of splitting e1, e2, and the context
			# accounts for the fact that the context can be one or more
			# elements
			e1_id, e2_id = row[:2]
			context = row[2:]

			# Determine how many noise examples of each type to generate
			num_noise_entities = int(np.round(
				self.noise_ratio * self.entity_noise_ratio
			))
			num_noise_contexts = self.noise_ratio - num_noise_entities

			# Sample random entities and context tokens
			noise_contexts = self.context_dictionary.sample(
				(num_noise_contexts, len(context)))
			noise_entities = self.entity_dictionary.sample(
				(num_noise_entities,))

			# Generate noise examples by inserting random contexts
			noise_examples.extend([
				[e1_id, e2_id] +  list(noise_context)
				for noise_context in noise_contexts
			])

			# Generate noise examples by inserting random entity.
			# Randomly choose which entity in the pair to replace
			for noise_entity in noise_entities:
				if np.random.uniform() < 0.5:
					noise_example = [noise_entity, e2_id] + list(context)
					noise_examples.append(noise_example)
				else:
					noise_example = [e1_id, noise_entity] + list(context)
					noise_examples.append(noise_example)

		return noise_examples


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



def find_tokens_between_closest_pair(indexes1, indexes2):
	'''
	Input: two lists of "entity-spans".  An entity span is the start and
		end token indices that define where in a sentance an entity
		mention occurs.  The inputs are each lists of entity spans (because
		a given entity often occurs in multiple places in a sentence) due
		to coreference.

	Finds the indices of tokens arising between the two nearest mentions,
	one outlined by indexes1, the other outlined by indexes2.  However,
	it won't consider the case where the two entities are immediately 
	adjacent as a valid case, and would instead return the tokens between
	the next-closest pair of mentions.

	The idea here is that the tokens between two entities that are close
	together give some semantic indication of their relationship (but if
	the two entities are immediately adjacent, we don't get any context
	tokens).
	'''
	distances = []

	if len(indexes1) < 1 or len(indexes2) < 1:
		raise ValueError(
			'indexes1 and indexes2 must be non-empty lists of index'
			' tuples.'
		)

	# Look at each pairing of indices for the entities, and calculate
	# the distance between them
	for i1 in indexes1:
		for i2 in indexes2:

			# The start index of entity spans are always off by one
			# This is too costly to correct in the dataset, so 
			# compensate here.  Now the span is in slice notation.
			i1 = i1[0] - 1, i1[1]
			i2 = i2[0] - 1, i2[1]

			# Get the distance between the entity spans.  How to do
			# this depends on which span comes first.
			if i1[0] < i2[0]:
				diff = min(i2) - max(i1)
				tokens_between = range(max(i1), min(i2))

			else:
				diff = min(i1) - max(i2)
				tokens_between = range(max(i2), min(i1))

			# Store the calculated distance, and intervening tokens 
			distances.append((diff, tokens_between))

	# Sort distances so that the nearest pair of entities is last
	distances.sort(reverse=True)

	# We want the nearest pair of entities, but not entities that 
	# are immediately next to one another
	diff, tokens_between = distances.pop()
	while len(tokens_between) < 1 and len(distances) > 0:
		diff, tokens_between = distances.pop()

	# Return the token ids between the nearest pair of entities
	# that are not immediately next to one another
	return tokens_between

#class NoiseContrastiveTheanoMinibatcher(TheanoMinibatcher):
#	pass
