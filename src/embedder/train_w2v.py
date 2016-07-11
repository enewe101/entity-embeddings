#!/usr/bin/env python
import t4k
import os
from theano import tensor as T
import time
import sys
sys.path.append('..') # This means that settings is importable
from SETTINGS import DATA_DIR, COOCCURRENCE_DIR
import re
from word2vec import (
    NoiseContraster, Word2VecEmbedder, Word2VecMinibatcher
)

FILES = [
	os.path.join(COOCCURRENCE_DIR, '%s.tsv' % file_num)
	for file_num in [
		'002', '003', '006', '007', '009', '00d'
		'00e', '010', '017', '018', '01b', '01d'
	]
]
DIRECTORIES = [
	#os.path.join(CORPUS_DIR, 'cooccurrence'),
	#os.path.join(CORPUS_DIR, 'cooccurrence/0')
]
SKIP = [
	re.compile(r'README\.txt'),
]
BATCH_SIZE=1000
THRESHOLD = 1 # This means there will be no discarding
NOISE_RATIO = 15
SAVEDIR = os.path.join(DATA_DIR, 'word2vec')
MIN_FREQUENCY = 20
NUM_EMBEDDING_DIMENSIONS = 500
NUM_EPOCHS = 1


def prepare():
	minibatcher = Word2VecMinibatcher(
		files=FILES,
		directories=DIRECTORIES,
		skip=SKIP,
		batch_size=BATCH_SIZE,
		t=THRESHOLD,
	)
	minibatcher.prepare(
		savedir=SAVEDIR
	)


def train(iteration_mode):

	if iteration_mode not in ('generate', 'before', 'background'):
		raise ValueError(
			'Got unexpected iteration_mode: %s' % iteration_mode
		)

	# Define the input theano variables
	print 'Making noise and signal channels'
	signal_input = T.imatrix('query_input')
	noise_input = T.imatrix('noise_input')

	# Make a NoiseContraster, and get the combined input
	print 'Making NoiseContraster'
	noise_contraster = NoiseContraster(signal_input, noise_input)
	combined_input = noise_contraster.get_combined_input()

	# Make a Word2VecMinibatcher
	print 'Making Word2VecMinibatcher'
	minibatcher = Word2VecMinibatcher(
		files=FILES,
		directories=DIRECTORIES,
		skip=SKIP,
		noise_ratio=NOISE_RATIO,
		t=THRESHOLD,
		batch_size=BATCH_SIZE, 
		verbose=True,
		num_example_generators=4
	)

	# load the minibatch generator.  Prune very rare tokens.
	print 'Loading and pruning dictionaries'
	minibatcher.load(SAVEDIR)
	minibatcher.prune(min_frequency=MIN_FREQUENCY)

	# Make a Word2VecEmbedder object, feed it the combined input
	print 'Making Word2VecEmbedder'
	word2vec_embedder = Word2VecEmbedder(
		combined_input,
		batch_size=BATCH_SIZE,
		vocabulary_size=len(minibatcher.unigram_dictionary),
		num_embedding_dimensions = NUM_EMBEDDING_DIMENSIONS
	)

	# Get the params and output from the word2vec embedder, feed that
	# back to the noise_contraster to get the training function
	print 'Compiling training function'
	combined_output = word2vec_embedder.get_output()
	params = word2vec_embedder.get_params()
	train = noise_contraster.get_train_func(combined_output, params)

	batching_start = time.time()
	print 'seven'
	# Figure out which iterator to use
	if iteration_mode == 'generate':
		print 'Generating minibatches to order'
		minibatch_iterator = minibatcher.generate_minibatches()

	elif iteration_mode == 'before':
		print 'Generating all minibatches upfront (this could take awhile)'
		minibatches = minibatcher.get_minibatches()
		minibatch_iterator = minibatches
		print 'Done generating minibatches.'

	elif iteration_mode == 'background':
		print 'Generating minibatches in the background'
		minibatch_iterator = minibatcher

	else:
		raise ValueError(
			'Got unexpected iteration_mode: %s' % iteration_mode
		)


	# Iterate over the corpus, training the embeddings
	print 'Starting training.'
	training_start = time.time()
	for epoch in range(NUM_EPOCHS):
		print 'starting epoch %d' % epoch
		epoch_start = time.time()
		batch_num = -1
		for signal_batch, noise_batch in minibatch_iterator:
			batch_num += 1
			t4k.out('.')
			loss = train(signal_batch, noise_batch)
			if batch_num % 100 == 0:
				print '\tloss: %f' % loss

		epoch_elapsed = time.time() - epoch_start
		print (
			'\tFinished epoch %d.  Time elapsed %2.1f.' 
			% (epoch, epoch_elapsed)
		)

	print 'Time needed for batching and training:', (
		time.time() - batching_start)
	print 'Time needed for training: %f' % (time.time() - training_start)

#	# Save the model (the embeddings) if savedir was provided
#	print 'Saving model...'
#	embeddings_filename = os.path.join(SAVEDIR, 'embeddings.npz')
#	word2vec_embedder.save(embeddings_filename)
#
#	print 'Total training time: %f' % (time.time() - training_start)
#	# Return the trained word2vec_embedder
#	return word2vec_embedder


if __name__ == '__main__':
	if sys.argv[1] == 'prepare':

		start = time.time()
		prepare()
		elapsed = time.time() - start
		print 'Elapsed:', elapsed

	elif sys.argv[1] == 'train':
		iteration_mode = sys.argv[2]
		train(iteration_mode)
		print 'success'

	else:
		print 'usage: ./train_w2v.py [ prepare | train ]'

