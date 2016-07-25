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
	DatasetReader as Word2VecDatasetReader, DataSetReaderIllegalStateException, TheanoMinibatcher,
	UnigramDictionary, reseed
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


## We need to slightly change the parser used in the word2vec minibatcher
## in order to handle the file format in this project
#class Word2VecMinibatcher(_Word2VecMinibatcher):
#	def parse(self, filename):
#		return word2vec_parse(filename)


class Relation2VecDatasetReader(Word2VecDatasetReader):

	NOT_DONE = 0
	DONE = 1

	def __init__(
		self,
		files=[],
		directories=[],
		skip=[],
		noise_ratio=15,
		num_processes=3,
		entity_dictionary=None,
		context_dictionary=None,
		load_dictionary_dir=None,
		max_queue_size=0,
		macrobatch_size=20000,
		verbose=True,

	):

		# TODO: test that no discarding is occurring
		# TODO: ensure that kernel is not being used to sample signal 
		# context 
		# TODO: what is hapening with unigram_dictionary -- we don't want 
		#	to use it
		unigram_dictionary=None,
		kernel=None
		verbose=True
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

		self.max_queue_size = max_queue_size
		self.macrobatch_size = macrobatch_size

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
			'Relation2VecDatasetReader: does not support `get_vocab_size()`. '
			'use `entity_vocab_size()` or `context_vocab_size()`.'
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
		filenames (dictionary.gz and unigram-dictionary.gz), by specifying
		only their containing save_dir
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
		self.context_dictionary.prune(min_frequency)
		self.entity_dictionary.prune(min_frequency)


	# Integrate into dataset generation pipeline
	# TODO: this should be called 'produce macrobatches'
	def generate_macrobatches(self, filename_iterator):

		'''
		Assembles bunches of examples from the parsed data coming from
		files that were read.  Normally, this function might yield
		individual examples, however, in this case, we need to maintain
		a distinction between the noise- and signal-examples, and to
		keep them in consistent proportions.  So, here, we yield small
		bunches that consist of 1 signal example, and X noise examples,
		where X depends on `self.noise_ratio`.
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

			signal_macrobatch = self.numpyify(
				signal_examples + [[UNK,UNK,UNK]] * signal_remaining)
			noise_macrobatch = self.numpyify(
				noise_examples + [[UNK,UNK,UNK]] * noise_remaining)

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

					# convert entities into ids
					e1_id, e2_id = self.entity_dictionary.get_ids([e1, e2])

					# TODO test this
					# Get the context tokens minus the entity_spans
					filtered_context_tokens = self.eliminate_spans(
						token_ids, entity_spans[e1] + entity_spans[e2]
					)

					# We can't train if there are no context words
					if len(filtered_context_tokens) == 0:
						break

					# Add a signal example.
					context = np.random.choice(filtered_context_tokens)
					signal_examples = [[e1_id, e2_id, context]]
					num_examples += 1

					# Sample tokens from the noise
					noise_context_ids = self.context_dictionary.sample(
						(self.noise_ratio * len(signal_examples),))
					noise_examples = [
						[e1_id, e2_id, noise_context_id]
						for noise_context_id in noise_context_ids
					]
					num_examples += len(noise_examples)

					yield (signal_examples, noise_examples)

		if self.verbose:
			print 'num_examples', num_examples


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

		if len(examples) > 0:
			examples = np.array(examples, dtype='int32')
		else:
			examples = np.empty(shape=(0,3), dtype='int32')

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



class NoiseContrastiveTheanoMinibatcher(TheanoMinibatcher):
	pass
