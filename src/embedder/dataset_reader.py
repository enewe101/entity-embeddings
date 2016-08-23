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
		if num_unique > 10 or len(tokens) > 80:
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
		unigram_dictionary=None,
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


	def save_data(self, save_dir):
		'''
		Save the generated dataset to the given directory.
		'''

		if not self.data_loaded:
			raise DataSetReaderIllegalStateException(
				'DatasetReader: cannot save the dataset before any data has '
				'been generated.'
			)

		examples_dir = os.path.join(save_dir, 'examples')
		# We are willing to create both the save_dir, and the
		# 'examples' subdir, but not their parents
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
		if not os.path.exists(examples_dir):
			os.mkdir(examples_dir)

		path = os.path.join(examples_dir, '0.npz')
		np.savez(
			path,
			signal_examples=self.signal_examples,
			noise_examples=self.noise_examples
		)


	MATCH_EXAMPLE_STORE = re.compile(r'[0-9]+\.npz')
	def load_data(self, save_dir):
		'''
		Load the dataset from the given directory
		'''
		examples_dir = os.path.join(save_dir, 'examples')
		fnames = check_output(['ls %s' % examples_dir], shell=True).split()
		signal_macrobatches = []
		noise_macrobatches = []
		for fname in fnames:
			if not self.MATCH_EXAMPLE_STORE.match(fname):
				continue
			f = np.load(os.path.join(examples_dir, fname))
			signal_macrobatches.append(f['signal_examples'].astype('int32'))
			noise_macrobatches.append(f['noise_examples'].astype('int32'))

		if len(signal_macrobatches) < 1 or len(noise_macrobatches) < 1:
			raise IOError(
				'DatasetReader: no example data files found in %s.' 
				% examples_dir
			)

		self.signal_examples = np.concatenate(signal_macrobatches)
		self.noise_examples = np.concatenate(noise_macrobatches)
		self.data_loaded = True
		return self.signal_examples, self.noise_examples


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


	def prune(self, min_frequency=5):
		'''
		Exposes the prune function for the underlying UnigramDictionary
		used for the context_dictionary.
		'''
		if self.verbose:
			print 'prunning dictionaries...'
			print (
				'\t...original vocabularies: entity, context = %d, %d'
				% (len(self.context_dictionary), len(self.entity_dictionary)
			))

		self.context_dictionary.prune(min_frequency)
		self.entity_dictionary.prune(min_frequency)

		if self.verbose:
			print (
				'\t...pruned vocabularies: entity, context = %d, %d'
				% (len(self.context_dictionary), len(self.entity_dictionary)
			))


	# Integrate into dataset generation pipeline
	# TODO: this should be called 'produce macrobatches'
	def generate_macrobatches(self, filename_iterator):

		'''
		Generator that produces macrobatches of signal and noise examples.
		It calls on file reading, parsing, and example genearating 
		machinery, but is itself only really responsible for the grouping
		of yielded examples into macrobatches.  Each iteration of this 
		generator yields a tuple consisting of a signal and noise 
		macrobatch.
		'''

		mcbatch_size = self.macrobatch_size
		noise_ratio = self.noise_ratio
		signal_examples = []
		noise_examples = []

		examples = self.generate_examples(filename_iterator)
		for signal_chunk, noise_chunk in examples:

			signal_examples.extend(signal_chunk)
			noise_examples.extend(noise_chunk)

			while len(signal_examples) > mcbatch_size:
				if self.verbose:
					print 'numpyifying'
				signal_macrobatch = self.numpyify(
					signal_examples[:mcbatch_size])
				noise_macrobatch = self.numpyify(
					noise_examples[:mcbatch_size * noise_ratio])

				entities = signal_macrobatch[0][:2]
				context = signal_macrobatch[0][2:]
				entities = self.entity_dictionary.get_tokens(entities)
				context = self.context_dictionary.get_tokens(context)
				if self.verbose:
					print 'no-padding:', len(signal_macrobatch)
				yield signal_macrobatch, noise_macrobatch

				signal_examples = signal_examples[mcbatch_size:]
				noise_examples = noise_examples[mcbatch_size*noise_ratio:]

		if len(signal_examples) > 0:
			signal_remaining = mcbatch_size - len(signal_examples)
			noise_remaining = (
				mcbatch_size * noise_ratio - len(noise_examples))

			if self.verbose:
				print 'padding and numpyifying'

			padding_row = [UNK,UNK] + [UNK] * self.len_context
			signal_macrobatch = self.numpyify(
				signal_examples + [padding_row] * signal_remaining)
			noise_macrobatch = self.numpyify(
				noise_examples + [padding_row] * noise_remaining)

			if self.verbose:
				print 'padded to length:', len(signal_macrobatch)
			yield signal_macrobatch, noise_macrobatch


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


	def find_tokens_between_closest_pair(self, indexes1, indexes2):
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
		context_indices = self.find_tokens_between_closest_pair(
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


	def generate_dataset_serial(self, save_dir=None):
		'''
		Generate the dataset from files handed to the constructor.  A single
		process is used, and all the data is stored in a single file at
		'save_dir/examples/0.npz'.
		'''

		# This cannot be called before calling prepare(), unless a prepared
		# UnigramDictionary was passed to the self's constructor
		if not self.prepared:
			raise DataSetReaderIllegalStateException(
				"DatasetReader: generate_examples() cannot be called before "
				"prepare() is called unless a prepared UnigramDictionary has "
				"was passed into the DatasetReader's constructor."
			)

		# We save dataset in the "examples" subdir of the model_dir
		if save_dir is not None:
			examples_dir = os.path.join(save_dir, 'examples')

			# We are willing to create both the save_dir, and the
			# 'examples' subdir, but not their parents
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)
			if not os.path.exists(examples_dir):
				os.mkdir(examples_dir)

		else:
			examples_dir = None

		# Generate the data for each file
		file_iterator = self.generate_filenames()
		macrobatches = self.generate_macrobatches(file_iterator)
		for signal_examples, noise_examples in macrobatches:

			## Save the data
			#if examples_dir is not None:
			#	save_path = os.path.join(examples_dir, '0.npz')
			#	np.savez(
			#		save_path,
			#		signal_examples=self.signal_examples,
			#		noise_examples=self.noise_examples
			#	)

			yield signal_examples, noise_examples


	def generate_dataset_worker(self, file_iterator, macrobatch_queue):
		macrobatches = self.generate_macrobatches(file_iterator)
		for signal_examples, noise_examples in macrobatches:
			if self.verbose:
				print 'sending macrobatch to parent process'
			macrobatch_queue.put((signal_examples, noise_examples))

		macrobatch_queue.close()


	def generate_dataset_parallel(self, save_dir=None):
		'''
		Parallel version of generate_dataset_serial.  Each worker is 
		responsible for saving its own part of the dataset to disk, called 
		a macrobatch.  the files are saved at 
		'save_dir/examples/<batch-num>.npz'.
		'''
		# This cannot be called before calling prepare(), unless a prepared
		# UnigramDictionary was passed to the self's constructor
		if not self.prepared:
			raise DataSetReaderIllegalStateException(
				"DatasetReader: generate_examples() cannot be called "
				"before prepare() is called unless a prepared "
				"UnigramDictionary has was passed into the DatasetReader's "
				"constructor."
			)

		# We save dataset in the "examples" subdir of the model_dir
		if save_dir is not None:
			examples_dir = os.path.join(save_dir, 'examples')
			# We are willing to create both the save_dir, and the
			# 'examples' subdir, but not their parents
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)
			if not os.path.exists(examples_dir):
				os.mkdir(examples_dir)
		else:
			examples_dir = None

		file_queue = IterableQueue()
		macrobatch_queue = IterableQueue(self.max_queue_size)

		# Put all the filenames on a producer queue
		file_producer = file_queue.get_producer()
		for filename in self.generate_filenames():
			file_producer.put(filename)
		file_producer.close()

		# Start a bunch of worker processes
		for process_num in range(self.num_processes):
			# Hop to a new location in the random-number-generator's state chain
			reseed()
			# Start child process that generates a portion of the dataset
			args = (
				file_queue.get_consumer(),
				macrobatch_queue.get_producer()
			)
			Process(target=self.generate_dataset_worker, args=args).start()

		# This will receive the macrobatches from all workers
		macrobatch_consumer = macrobatch_queue.get_consumer()

		# Close the iterable queues
		file_queue.close()
		macrobatch_queue.close()

		# Retrieve the macrobatches from the workers, write them to file
		signal_macrobatches = []
		noise_macrobatches = []
		for macrobatch_num, (signal_macrobatch, noise_macrobatch) in enumerate(macrobatch_consumer):
			if self.verbose:
				print 'receiving macrobatch from child process'
			if examples_dir is not None:
				save_path = os.path.join(examples_dir, '%d.npz' % macrobatch_num)
				#np.savez(
				#	save_path,
				#	signal_examples=signal_macrobatch,
				#	noise_examples=noise_macrobatch
				#)

			yield signal_macrobatch, noise_macrobatch
			#signal_macrobatches.append(signal_macrobatch)
			#noise_macrobatches.append(noise_macrobatch)

		# Concatenate the macrobatches, and return the dataset
		#print 'amalgamating macrobatches'
		#self.signal_examples = np.concatenate(signal_macrobatches)
		#self.noise_examples = np.concatenate(noise_macrobatches)
		#self.data_loaded = True
		#print 'returning dataset'
		#return self.signal_examples, self.noise_examples


	def numpyify(self, examples):
		'''
		Make an int32-type numpy array, ensuring that, even if the list of
		examples is empty, the array is two-dimensional, with the second
		dimension (i.e. number of columns) being 3.
		'''

		try:
			if len(examples) > 0:
				examples = np.array(examples, dtype='int32')
			else:
				examples = np.empty(shape=(0,3), dtype='int32')
		except ValueError:
			for row in examples:
				print len(row), row
			raise

		return examples



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



#class NoiseContrastiveTheanoMinibatcher(TheanoMinibatcher):
#	pass
