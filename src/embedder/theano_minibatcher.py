from iterable_queue import IterableQueue
import re
import random
import t4k
import itertools as itools
from collections import defaultdict
from multiprocessing import Queue, Process, Pipe
from Queue import Empty
from word2vec import DatasetReader, TheanoMinibatcher, UnigramDictionary
import numpy as np
import gzip
import os
from word2vec.token_map import UNK
from theano import shared, function, tensor as T

class NoiseContrastiveTheanoMinibatcher(TheanoMinibatcher):

	def __init__(
		self,
		batch_size=1000,
		noise_ratio=15,
		dtype="float32",
		num_dims=2
	):
		self.batch_size = batch_size
		self.noise_ratio = noise_ratio
		self.dtype = dtype
		self.num_dims = num_dims

		self._setup_batching()


	def _setup_batching(self):

		# Make an empty shared variable that will store the dataset
		# Although empty, we can setup the relationship between the
		# minibatch variable and the full dataset
		self.signal_examples = shared(
			self._initialize_data_container(self.num_dims, self.dtype)
		)
		self.noise_examples = shared(
			self._initialize_data_container(self.num_dims, self.dtype)
		)

		# Make minibatch by indexing into the dataset
		self.batch_num = shared(np.int32(0))

		# Take a sliding minibatch window on the signal_examples
		signal_batch_start = self.batch_num * self.batch_size
		signal_batch_end = signal_batch_start + self.batch_size
		signal_batch = self.signal_examples[signal_batch_start : signal_batch_end,]

		# Take a sliding minibatch window on the noise_examples
		noise_batch_start = self.batch_num * self.batch_size * self.noise_ratio
		noise_batch_end = noise_batch_start + self.batch_size * self.noise_ratio
		noise_batch = self.noise_examples[noise_batch_start : noise_batch_end,]

		# Concatenate the signal and noise minibatch into the full minibatch
		self.batch = T.concatenate((signal_batch, noise_batch))

		# Define an update that moves the batch window through the dataset
		self.updates = [(self.batch_num, self.batch_num+1)]


	def load_dataset(self, signal_examples, noise_examples):
		'''
		Load the dataset onto the GPU.  Determine (and return) the number of
		minibatches.
		'''

		# Reset the internal pointer
		self.reset()

		# Determine the total number of minibatches
		self.num_batches = int(np.ceil(len(signal_examples) / float(self.batch_size)))

		# Check if the dataset divides evenly into batches
		warn_last_batch = False
		expected_len_signal = self.num_batches * self.batch_size
		if expected_len_signal > len(signal_examples):
			warn_last_batch = True

		expected_len_noise = self.num_batches * self.batch_size * self.noise_ratio
		if expected_len_noise > len(noise_examples):
			warn_last_batch = True

		# If dataset doesn't divide evenly into batches, warn the user, and
		# drop the last batch
		if warn_last_batch:
			print 'Warning: incomplete last batch will be ommitted'
			# We ommit the last batch simply by reporting fewer total batches.
			# It is actually up to the caller to only use self.batch_num
			# minibatches.
			self.num_batches -= 1

		# Load the dataset onto the gpu
		self.signal_examples.set_value(signal_examples)
		self.noise_examples.set_value(noise_examples)

		return self.num_batches
