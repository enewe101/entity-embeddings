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
		batch_size = 1000,
		noise_ratio=15,
		num_processes=3,
		entity_dictionary=None,
		context_dictionary=None,
		verbose=True,
	):

		# TODO: test that no discarding is occurring
		# TODO: ensure that kernel is not being used to sample signal context
		# TODO: what is hapening with unigram_dictionary -- we don't want to use it
		unigram_dictionary=None,
		kernel=None
		verbose=True
		t = 1.0

		super(Relation2VecDatasetReader, self).__init__(
			files=files,
			directories=directories,
			skip=skip,
			batch_size=batch_size,
			noise_ratio=noise_ratio,
			t=t,
			num_processes=num_processes,
			unigram_dictionary=unigram_dictionary,
			kernel=kernel,
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

	def get_vocab_size(self):
		raise NotImplementedError(
			'Relation2VecDatasetReader: does not support `get_vocab_size()`. '
			'use `entity_vocab_size()` or `context_vocab_size()`.'
		)

	def parse(self, filename):
		return relation2vec_parse(filename)


	def entity_vocab_size(self):
		return len(self.entity_dictionary)


	def context_vocab_size(self):
		return len(self.context_dictionary)


	# TODO: harmonize this with the two-level loading of DatasetReader (examples and dictionaries)
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
				'DatasetReader: no example data files found in %s.' % examples_dir
			)

		self.signal_examples = np.concatenate(signal_macrobatches)
		self.noise_examples = np.concatenate(noise_macrobatches)
		self.data_loaded = True
		return self.signal_examples, self.noise_examples


	# TODO: harmonize this with the two-level saving / loading of DatasetReader (examples and dictionaries)
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
	def produce_examples(self, filename_iterator):

		'''
		Assembles bunches of examples from the parsed data coming from
		files that were read.  Normally, this function might yield
		individual examples, however, in this case, we need to maintain
		a distinction between the noise- and signal-examples, and to
		keep them in consistent proportions.  So, here, we yield small
		bunches that consist of 1 signal example, and X noise examples,
		where X depends on `self.noise_ratio`.
		'''

		signal_examples = []
		noise_examples = []
		for fname in filename_iterator:

			parsed = self.parse(fname)
			add_signal_examples, add_noise_examples = self.build_examples(parsed)

			signal_examples.extend(add_signal_examples)
			noise_examples.extend(add_noise_examples)

		# Numpyify the dataset
		signal_examples = self.numpyify(signal_examples)
		noise_examples = self.numpyify(noise_examples)

		return signal_examples, noise_examples


	def build_examples(self, parsed):

		signal_examples = []
		noise_examples = []

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
				filtere_context_tokens = self.eliminate_spans(
					token_ids, entity_spans[e1] + entity_spans[e2]
				)

				# We can't train if there are no context words
				if len(filtere_context_tokens) == 0:
					break

				# Add the signal examples.  Note that we sample all context
				# rather than just one word as would be done in word2vec
				add_signal_examples = [
					[e1_id, e2_id, context_token_id]
					for context_token_id in filtere_context_tokens
				]
				signal_examples.extend(add_signal_examples)

				# Sample tokens from the noise
				noise_context_ids = self.context_dictionary.sample(
					(self.noise_ratio * len(add_signal_examples),))
				add_noise_examples = [
					[e1_id, e2_id, noise_context_id]
					for noise_context_id in noise_context_ids
				]
				noise_examples.extend(add_noise_examples)

		return signal_examples, noise_examples


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
		self.signal_examples, self.noise_examples = self.produce_examples(file_iterator)
		self.data_loaded = True

		# Save the data
		if examples_dir is not None:
			save_path = os.path.join(examples_dir, '0.npz')
			np.savez(
				save_path,
				signal_examples=self.signal_examples,
				noise_examples=self.noise_examples
			)

		# Return it
		return self.signal_examples, self.noise_examples


	def generate_dataset_worker(self, file_iterator, macrobatch_queue):
		macrobatch = self.produce_examples(file_iterator)
		macrobatch_queue.put(macrobatch)
		macrobatch_queue.close()


	def generate_dataset_parallel(self, save_dir=None):
		'''
		Parallel version of generate_dataset_serial.  Each worker is responsible
		for saving its own part of the dataset to disk, called a macrobatch.
		the files are saved at 'save_dir/examples/<batch-num>.npz'.
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

		file_queue = IterableQueue()
		macrobatch_queue = IterableQueue()

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
			if examples_dir is not None:
				save_path = os.path.join(examples_dir, '%d.npz' % macrobatch_num)
				np.savez(
					save_path,
					signal_examples=signal_macrobatch,
					noise_examples=noise_macrobatch
				)
			signal_macrobatches.append(signal_macrobatch)
			noise_macrobatches.append(noise_macrobatch)

		# Concatenate the macrobatches, and return the dataset
		self.signal_examples = np.concatenate(signal_macrobatches)
		self.noise_examples = np.concatenate(noise_macrobatches)
		self.data_loaded = True
		return self.signal_examples, self.noise_examples


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
