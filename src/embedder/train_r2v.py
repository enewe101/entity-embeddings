#!/usr/bin/env python

# TODO: test dict_dir, save_dir, and learn_embeddings options

import t4k
import numpy as np
from theano import tensor as T
import time
import sys
sys.path.append('..')
import re
from theano_minibatcher import NoiseContrastiveTheanoMinibatcher
from relation2vec_embedder import Relation2VecEmbedder
from dataset_reader import Relation2VecDatasetReader
import os
from SETTINGS import DATA_DIR, COOCCURRENCE_DIR

# Seed randomness for reproducibility
np.random.seed(0)

DIRECTORIES = [
	#os.path.join(COOCCURRENCE_DIR, 'cooccurrence')
]
FILES = [
	os.path.join(COOCCURRENCE_DIR, '%s.tsv' % file_num)
	for file_num in 
	['002', '003']#, '006', '007', '009', '00d', '00e', '010', '017', '018']
]
SKIP = [
	re.compile('README.txt')
]
BATCH_SIZE=1000
NOISE_RATIO = 15
SAVE_DIR = os.path.join(DATA_DIR, 'relation2vec')
MIN_FREQUENCY = 20
NUM_EMBEDDING_DIMENSIONS = 500
NUM_EPOCHS = 1
LEARNING_RATE = 0.01

def prepare_dataset():

	minibatcher = Relation2VecDatasetReader(
		directories=DIRECTORIES,
		files=FILES,
		skip=SKIP,
		batch_size=BATCH_SIZE,
		noise_ratio=NOISE_RATIO,
		num_processes=4,
	)
	minibatcher.prepare(
		save_dir=SAVE_DIR
	)
	minibatcher.generate_dataset_parallel(
		save_dir=SAVE_DIR
	)


def train(iteration_mode, learn_embeddings=True):

	if iteration_mode not in ('generate', 'before', 'background'):
		raise ValueError(
			'Got unexpected iteration_mode: %s' % iteration_mode
		)

	print 'one'

	# Define the input theano variables
	signal_input = T.imatrix('query_input')
	noise_input = T.imatrix('noise_input')

	print 'two'
	# Make a NoiseContraster, and get the combined input
	noise_contraster = NoiseContraster(
		signal_input, noise_input, learning_rate=LEARNING_RATE
	)
	combined_input = noise_contraster.get_combined_input()

	print 'three'
	# Make a Relation2VecDatasetReader
	minibatcher = Relation2VecDatasetReader(
		files=FILES, directories=DIRECTORIES, skip=SKIP,
		noise_ratio=NOISE_RATIO,
		batch_size=BATCH_SIZE,
	)

	print 'four'
	# load the minibatch generator.  Prune very rare tokens.
	minibatcher.load(SAVE_DIR)

	# For debuggin tests, don't prune -- everything is rare!
	#minibatcher.prune(min_frequency=MIN_FREQUENCY)

	print 'entity vocabulary:', len(minibatcher.entity_dictionary)
	print 'context vocabulary:', len(minibatcher.context_dictionary)

	print 'five'
	# Make a Relation2VecEmbedder object, feed it the combined input
	entity_embedder = Relation2VecEmbedder(
		combined_input,
		batch_size=BATCH_SIZE,
		entity_vocab_size=len(minibatcher.entity_dictionary),
		context_vocab_size=len(minibatcher.context_dictionary),
		num_embedding_dimensions = NUM_EMBEDDING_DIMENSIONS
	)

	print 'six'
	# Get the params and output from the word2vec embedder, feed that
	# back to the noise_contraster to get the training function
	combined_output = entity_embedder.get_output()
	params = entity_embedder.get_params()

	# If embeddings are to be kept fixed, then only keep the parameters
	# defining how relationships are composed out of embeddings
	if not learn_embeddings:
		params = params[2:]

	train = noise_contraster.get_train_func(combined_output, params)

	batching_start = time.time()
	print 'seven'
	# Figure out which iterator to use
	if iteration_mode == 'generate':
		print 'Generating minibatches to order'
		get_minibatch_iterator = minibatcher.generate_minibatches

	elif iteration_mode == 'before':
		print 'Generating all minibatches upfront (this could take awhile)'
		minibatches = minibatcher.get_minibatches()
		get_minibatch_iterator = lambda: minibatches
		print 'Done generating minibatches.'

	elif iteration_mode == 'background':
		print 'Generating minibatches in the background'
		get_minibatch_iterator = lambda: minibatcher

	else:
		raise ValueError(
			'Got unexpected iteration_mode: %s' % iteration_mode
		)

	# Iterate over the corpus, training the embeddings
	training_start = time.time()
	for epoch in range(NUM_EPOCHS):

		print 'starting epoch %d' % epoch
		epoch_start = time.time()
		batch_num = -1
		for signal_batch, noise_batch in get_minibatch_iterator():
			t4k.out('.')
			batch_num += 1
			loss = train(signal_batch, noise_batch)
			if batch_num % 100 == 0:
				print '\tloss: %f' % loss

		epoch_elapsed = time.time() - epoch_start
		print (
			'\tFinished epoch %d.  Time for epoch was %2.1f.' 
			% (epoch, epoch_elapsed)
		)

	print 'Time needed for batching and training:', (
		time.time() - batching_start)
	print 'Time needed for training: %f' % (time.time() - training_start)

	print 'Saving the model...'
	# Save the model (the embeddings) if save_dir was provided
	embeddings_filename = os.path.join(SAVE_DIR, 'embeddings.npz')
	entity_embedder.save(embeddings_filename)

	# Return the trained entity_embedder
	return entity_embedder



if __name__ == '__main__':

	if sys.argv[1] == 'prepare':

		start = time.time()
		prepare_dataset()
		elapsed = time.time() - start
		print 'Elapsed:', elapsed

	elif sys.argv[1] == 'train':
		iteration_mode = sys.argv[2]
		if sys.argv[3] == 'learn-embeddings':
			learn_embeddings == True
		elif sys.argv[3] == 'fix-embeddings':
			learn_embeddings == False
		else:
			raise ValueError(
				'Third argument must either be "learn-embeddings" or '
				'"fix-embeddings".'
			)

		dict_dir = os.path.join(DATA_DIR, sys.argv[4])
		save_dir = os.path.join(DATA_DIR, sys.argv[5])

		train(iteration_mode, learn_embeddings, dict_dir, save_dir)
		print 'success'

	else:
		print 'usage: ./train_w2v.py [ prepare | train ]'

