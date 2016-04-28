#!/usr/bin/env python
import os
from theano import tensor as T
import time
import sys
import re
from word2vec import (
    Word2Vec as W, NoiseContraster, Word2VecEmbedder, MinibatchGenerator
)

DIRECTORIES = [
	'/home/rldata/gigaword-corenlp/cooccurrence',
	'/home/rldata/gigaword-corenlp/cooccurrence/0'
]
SKIP = [
	re.compile(r'README\.txt'),
]
BATCH_SIZE=10000
THRESHOLD = 1 # This means there will be no discarding
NOISE_RATIO = 15
SAVEDIR = '/home/2012/enewel3/entity-embeddings/data/word2vec'
MIN_FREQUENCY = 5
NUM_EMBEDDING_DIMENSIONS = 500
NUM_EPOCHS = 1

def parse(filename):

	tokenized_sentences = []
	for line in open(filename):
		tokenized_sentences.append(line.split('\t')[1].split())

	return tokenized_sentences


def prepare():
	minibatch_generator = MinibatchGenerator(
		directories=DIRECTORIES,
		skip=SKIP,
		batch_size=BATCH_SIZE,
		t=THRESHOLD,
		parse=parse
	)
	minibatch_generator.prepare(
		savedir=SAVEDIR
	)


def train():

	# Define the input theano variables
	signal_input = T.imatrix('query_input')
	noise_input = T.imatrix('noise_input')

	# Make a NoiseContraster, and get the combined input
	noise_contraster = NoiseContraster(signal_input, noise_input)
	combined_input = noise_contraster.get_combined_input()

	# Make a MinibatchGenerator
	minibatch_generator = MinibatchGenerator(
		directories=DIRECTORIES,
		skip=SKIP,
		noise_ratio=NOISE_RATIO,
		t=THRESHOLD,
		batch_size=BATCH_SIZE, 
		parse=parse,
		verbose=True
	)

	# load the minibatch generator.  Prune very rare tokens.
	minibatch_generator.load(SAVEDIR)
	minibatch_generator.prune(min_frequency=MIN_FREQUENCY)

	# Make a Word2VecEmbedder object, feed it the combined input
	word2vec_embedder = Word2VecEmbedder(
		combined_input,
		batch_size=BATCH_SIZE,
		vocabulary_size=len(minibatch_generator.unigram_dictionary),
		num_embedding_dimensions = NUM_EMBEDDING_DIMENSIONS
	)

	# Get the params and output from the word2vec embedder, feed that
	# back to the noise_contraster to get the training function
	combined_output = word2vec_embedder.get_output()
	params = word2vec_embedder.get_params()
	train = noise_contraster.get_train_func(combined_output, params)


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

	# Save the model (the embeddings) if savedir was provided
	print 'Saving model...'
	embeddings_filename = os.path.join(SAVEDIR, 'embeddings.npz')
	word2vec_embedder.save(embeddings_filename)

	print 'total training time: %f' % (time.time() - training_start)
	# Return the trained word2vec_embedder
	return word2vec_embedder



if __name__ == '__main__':
	if sys.argv[1] == 'prepare':

		start = time.time()
		prepare()
		elapsed = time.time() - start
		print 'Elapsed:', elapsed

	elif sys.argv[1] == 'train':
		train()

	else:
		print 'usage: ./train_w2v.py [ prepare | train ]'

