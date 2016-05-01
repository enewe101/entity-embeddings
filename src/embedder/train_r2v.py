#!/usr/bin/env python
import numpy as np
from theano import tensor as T
import time
import sys
sys.path.append('..')
import re
from minibatch_generator import MinibatchGenerator
from entity_embedder import EntityEmbedder
from word2vec import NoiseContraster
import os
from SETTINGS import DATA_DIR, CORPUS_DIR

# Seed randomness for reproducibility
np.random.seed(0)

DIRECTORIES = [
	os.path.join(CORPUS_DIR, 'cooccurrence'),
	os.path.join(CORPUS_DIR, 'cooccurrence/0')
]
FILES = [
	#os.path.join(CORPUS_DIR, 'cooccurrence/0/%s.tsv' % file_num)
	#for file_num in 
	#['002', '003', '006', '007', '009', '00d', '00e', '010', '017', '018']
]
SKIP = [
	re.compile('README.txt')
]
BATCH_SIZE=1000
NOISE_RATIO = 15
SAVEDIR = os.path.join(DATA_DIR, 'relation2vec')
MIN_FREQUENCY = 20
NUM_EMBEDDING_DIMENSIONS = 500
NUM_EPOCHS = 1
LEARNING_RATE = 0.01

def prepare():
	minibatch_generator = MinibatchGenerator(
		directories=DIRECTORIES,
		files=FILES,
		skip=SKIP,
		batch_size=BATCH_SIZE,
	)
	minibatch_generator.prepare(
		savedir=SAVEDIR
	)


def train():

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
	# Make a MinibatchGenerator
	minibatch_generator = MinibatchGenerator(
		files=FILES, directories=DIRECTORIES, skip=SKIP,
		noise_ratio=NOISE_RATIO,
		batch_size=BATCH_SIZE,
	)

	print 'four'
	# load the minibatch generator.  Prune very rare tokens.
	minibatch_generator.load(SAVEDIR)
	minibatch_generator.prune(min_frequency=MIN_FREQUENCY)
	print 'entity vocabulary:', len(minibatch_generator.entity_dictionary)
	print 'context vocabulary:', len(minibatch_generator.context_dictionary)

	print 'five'
	# Make a EntityEmbedder object, feed it the combined input
	entity_embedder = EntityEmbedder(
		combined_input,
		batch_size=BATCH_SIZE,
		entity_vocab_size=len(minibatch_generator.entity_dictionary),
		context_vocab_size=len(minibatch_generator.context_dictionary),
		num_embedding_dimensions = NUM_EMBEDDING_DIMENSIONS
	)

	print 'six'
	# Get the params and output from the word2vec embedder, feed that
	# back to the noise_contraster to get the training function
	combined_output = entity_embedder.get_output()
	params = entity_embedder.get_params()
	train = noise_contraster.get_train_func(combined_output, params)

	print 'seven'
	# Iterate over the corpus, training the embeddings
	training_start = time.time()
	for epoch in range(NUM_EPOCHS):
		print 'starting epoch %d' % epoch
		epoch_start = time.time()
		batch_num = -1
		for signal_batch, noise_batch in minibatch_generator.generate():
			batch_num += 1
			loss = train(signal_batch, noise_batch)
			if batch_num % 100 == 0:
				print '\tloss: %f' % loss

		epoch_elapsed = time.time() - epoch_start
		print (
			'\tFinished epoch %d.  Time elapsed %2.1f.' 
			% (epoch, epoch_elapsed)
		)

	print 'eight'
	# Save the model (the embeddings) if savedir was provided
	embeddings_filename = os.path.join(SAVEDIR, 'embeddings.npz')
	entity_embedder.save(embeddings_filename)

	print 'total training time: %f' % (time.time() - training_start)
	# Return the trained entity_embedder
	return entity_embedder



if __name__ == '__main__':
	if sys.argv[1] == 'prepare':

		start = time.time()
		prepare()
		elapsed = time.time() - start
		print 'Elapsed:', elapsed

	elif sys.argv[1] == 'train':
		train()
		print 'success'

	else:
		print 'usage: ./train_w2v.py [ prepare | train ]'

