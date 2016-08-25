import sys
import os

import numpy as np
from word2vec import (
	get_noise_contrastive_loss, Word2VecEmbedder as Embedder
)
from dataset_reader import (
	EntityPair2VecDatasetReader as DatasetReader
)
from relation2vec_embedder import Relation2VecEmbedder
from theano_minibatcher import NoiseContrastiveTheanoMinibatcher
from r2v import read_context_embeddings, read_entity_embeddings


# Possibly import theano and lasagne
exclude_theano_set = 'EXCLUDE_THEANO' in os.environ 
if exclude_theano_set and int(os.environ['EXCLUDE_THEANO']) == 1:
	pass
else:
	from theano import function
	import theano
	from lasagne.init import Normal
	from lasagne.updates import nesterov_momentum


def entity_pair2vec(

	# Input / output options
	files=[],
	directories=[],
	skip=[],
	save_dir=None,
	read_data_async=True,
	num_processes=3,
	max_queue_size=0,

	# Batching options
	num_epochs=5,
	batch_size = 1000,  # Number of *signal* examples per batch
	macrobatch_size = 100000,

	# Dictionary options
	query_dictionary=None,
	context_dictionary=None,
	load_dictionary_dir=None,
	min_query_frequency=10,
	min_context_frequency=10,

	# Sampling options
	noise_ratio=15,

	# Embeddings options
	query_embeddings_fname=None,
	context_embeddings_fname=None,
	num_embedding_dimensions=500,
	query_embedding_init=None,
	context_embedding_init=None,

	# Learning rate options
	learning_rate=0.1,
	momentum=0.9,
	freeze_context=False,

	# Verbosity option
	verbose=True,
):
	'''
	Helper function that handles all concerns involved in training
	A word2vec model using the approach of Mikolov et al.  It surfaces
	all of the options.

	For customizations going beyond simply tweeking existing options and
	hyperparameters, substitute this function by writing your own training
	routine using the provided classes.  This function would be a starting
	point for you.
	'''

	# Make a DatasetReader, pass through parameters sent by 
	# caller
	reader = DatasetReader(
		files=files,
		directories=directories,
		skip=skip,
		macrobatch_size = macrobatch_size,
		max_queue_size = max_queue_size,
		noise_ratio=noise_ratio,
		num_processes=num_processes,
		query_dictionary=query_dictionary,
		context_dictionary=context_dictionary,
		load_dictionary_dir=load_dictionary_dir,
		verbose=verbose
	)

	# Prepare the dataset reader (this produces unigram stats)
	both_dictionaries_supplied = context_dictionary and query_dictionary
	if load_dictionary_dir is None and not both_dictionaries_supplied:
		if verbose:
			print 'preparing dictionaries...'
		reader.prepare(save_dir=save_dir)

	# If any min frequencies were specified, prune the dictionaries
	if min_query_frequency is not None or min_context_frequency is not None:
		if verbose:
			print 'pruning dictionaries...'
		reader.prune(
			min_query_frequency=min_query_frequency, 
			min_context_frequency=min_context_frequency
		)

	# Make a symbolic minibatcher 
	minibatcher = NoiseContrastiveTheanoMinibatcher(
		batch_size=batch_size,
		noise_ratio=noise_ratio,
		dtype="int32",
		num_dims=2
	)

	# Make an Embedder, feed it the combined input.
	# Note that the full batch includes noise examples and signal_examples
	# so is larger than batch_size, which is the number of signal_examples
	# only per batch.
	full_batch_size = batch_size * (1 + noise_ratio)
	embedder = Embedder(
		input_var=minibatcher.get_batch(),
		batch_size=full_batch_size,
		query_vocabulary_size=reader.query_vocab_size(),
		context_vocabulary_size=reader.context_vocab_size(),
		num_embedding_dimensions=num_embedding_dimensions,
		query_embedding_init=query_embedding_init,
		context_embedding_init=context_embedding_init,
	)

	# If a file for pre-trained context word embeddings was given, read it
	if context_embeddings_fname is not None:
		if verbose:
			print 'reading context embeddings'
		read_context_embeddings(context_embeddings_fname, reader, embedder)

	# If a file for pre-trained entity embeddings was given, read it
	if query_embeddings_fname is not None:
		raise NotImplementedError(
			"Pretrained embeddings for entity pairs don't exist. "
			"Maybe this could be implemented by taking the average or "
			"product of pretrained entity embeddings, but that's not "
			"implemented."
		)
		if verbose:
			print 'reading context embeddings'
		read_entity_embeddings(query_embeddings_fname, reader, embedder)

	# Architectue is ready.  Make the loss function, and use it to create 
	# the parameter updates responsible for learning
	loss = get_noise_contrastive_loss(embedder.get_output(), batch_size)

	# Optionally exclude the context embeddings (in params[1]) from the
	# parameters to be learnt.  This freezes them to their starting value
	params = embedder.get_params()
	if freeze_context:
		params = params[:1] + params[2:]

	updates = nesterov_momentum(
		loss, params, learning_rate, momentum
	)

	# Include minibatcher updates, which cause the symbolic batch to move
	# through the dataset like a sliding window
	updates.update(minibatcher.get_updates())

	# Use the loss function and the updates to compile a training function.
	# Note that it takes no inputs because the dataset is fully loaded using
	# theano shared variables
	train = function([], loss, updates=updates)

	# Iterate through the dataset, training the embeddings
	for epoch in range(num_epochs):

		if verbose:
			print 'starting epoch %d' % epoch

		if read_data_async:
			macrobatches = reader.generate_dataset_parallel()
		else:
			macrobatches = reader.generate_dataset_serial()

		macrobatch_num = 0
		for signal_macrobatch, noise_macrobatch in macrobatches:

			macrobatch_num += 1
			if verbose:
				print 'running macrobatch %d' % (macrobatch_num - 1)

			minibatcher.load_dataset(signal_macrobatch, noise_macrobatch)
			losses = []
			for batch_num in range(minibatcher.get_num_batches()):
				if verbose:
					print 'running minibatch', batch_num
				losses.append(train())
			if verbose:
				print '\taverage loss: %f' % np.mean(losses)

	# Save the model (the embeddings) if save_dir was provided
	if save_dir is not None:
		embedder.save(save_dir)

	# Return the trained embedder and the dictionary mapping tokens
	# to ids
	return embedder, reader


