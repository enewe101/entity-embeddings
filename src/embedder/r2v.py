import sys
import os

import numpy as np
from word2vec import get_noise_contrastive_loss
from dataset_reader import (
	Relation2VecDatasetReader as DatasetReader, RANDOM_SINGLE_CHOICE,
	FULL_CONTEXT
)
from relation2vec_embedder import Relation2VecEmbedder
from theano_minibatcher import NoiseContrastiveTheanoMinibatcher


# Possibly import theano and lasagne
exclude_theano_set = 'EXCLUDE_THEANO' in os.environ 
if exclude_theano_set and int(os.environ['EXCLUDE_THEANO']) == 1:
	pass
else:
	from theano import function
	import theano
	from lasagne.init import Normal
	from lasagne.updates import nesterov_momentum


def read_entity_embeddings(embeddings_fname, reader, embedder):
	raise NotImplementedError()


def read_context_embeddings(embeddings_fname, reader, embedder):
	# We'll read the embeddings from file and write them into the 
	# existing embeddings matrix. First, get the existing entity 
	# embeddings.  

	context_embedding_params = embedder.get_params()[1]
	context_embeddings = context_embedding_params.get_value()

	# Now read the embeddings file.  We will match the embeddings in 
	# the file based on the word that appears in the first position
	# on each line.  To do that, we need to already have a entity
	# dictionary.
	if not reader.is_prepared():
		raise ValueError(
			'Trying to load context embeddings from token-based embeddings '
			'file without first preparing the reader.  The reader needs '
			'to have built a context dictionary so that it can map the '
			'embedding tokens onto indices in the embeddings matrix.'
		)

	context_dictionary = reader.context_dictionary

	first_line = True
	for line in open(embeddings_fname):

		# skip first line
		if first_line:
			first_line = False
			continue

		# Parse the line
		fields = line.strip().split()
		token, vector = fields[0], fields[1:]

		# Get the position of this embedding
		token_id = context_dictionary.get_id(token)

		# Overwrite the embedding using the one from file
		context_embeddings[token_id] = vector

	context_embedding_params.set_value(context_embeddings)


def relation2vec(

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
	entity_dictionary=None,
	context_dictionary=None,
	entity_pair_dictionary=None,
	load_dictionary_dir=None,
	min_query_frequency=0,
	min_context_frequency=0,
	min_entity_pair_frequency=0,

	# Sampling options
	noise_ratio=15,
	entity_noise_ratio=0.0,
	signal_sample_mode=RANDOM_SINGLE_CHOICE,
	len_context=1,

	# Embeddings options
	context_embeddings_fname=None,
	entity_embeddings_fname=None,
	num_embedding_dimensions=500,
	entity_embedding_init=None,
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

	# For convenience, we allow signal_sample_mode to be a string
	if signal_sample_mode == 'RANDOM_SINGLE_CHOICE':
		signal_sample_mode = RANDOM_SINGLE_CHOICE
	elif signal_sample_mode == 'FULL_CONTEXT':
		signal_sample_mode = FULL_CONTEXT

	# Make a Relation2VecDatasetReader, pass through parameters sent by 
	# caller
	reader = DatasetReader(
		files=files,
		directories=directories,
		skip=skip,
		macrobatch_size = macrobatch_size,
		max_queue_size = max_queue_size,
		noise_ratio=noise_ratio,
		entity_noise_ratio=entity_noise_ratio,
		num_processes=num_processes,
		entity_dictionary=entity_dictionary,
		context_dictionary=context_dictionary,
		entity_pair_dictionary=entity_pair_dictionary,
		min_query_frequency=min_query_frequency,
		min_context_frequency=min_context_frequency,
		min_entity_pair_frequency=min_entity_pair_frequency,
		load_dictionary_dir=load_dictionary_dir,
		signal_sample_mode=signal_sample_mode,
		len_context=len_context,
		verbose=verbose
	)

	# Prepare the dataset reader (this produces unigram stats)
	if not reader.is_prepared():
		if verbose:
			print 'preparing dictionaries...'
		reader.prepare(save_dir=save_dir)

	# Make a symbolic minibatcher 
	minibatcher = NoiseContrastiveTheanoMinibatcher(
		batch_size=batch_size,
		noise_ratio=noise_ratio,
		dtype="int32",
		num_dims=2
	)

	# Make a Word2VecEmbedder object, feed it the combined input.
	# Note that the full batch includes noise examples and signal_examples
	# so is larger than batch_size, which is the number of signal_examples
	# only per batch.
	full_batch_size = batch_size * (1 + noise_ratio)
	embedder = Relation2VecEmbedder(
		input_var=minibatcher.get_batch(),
		batch_size=full_batch_size,
		entity_vocab_size=reader.entity_vocab_size(),
		context_vocab_size=reader.context_vocab_size(),
		num_embedding_dimensions=num_embedding_dimensions,
		entity_embedding_init=entity_embedding_init,
		context_embedding_init=context_embedding_init,
		len_context=len_context,
	)

	# If a file for pre-trained context word embeddings was given, read it
	if context_embeddings_fname is not None:
		if verbose:
			print 'reading context embeddings'
		read_context_embeddings(context_embeddings_fname, reader, embedder)

	# TODO: implement this!
	# If a file for pre-trained entity embeddings was given, read it
	if entity_embeddings_fname is not None:
		if verbose:
			print 'reading context embeddings'
		read_entity_embeddings(entity_embeddings_fname, reader, embedder)

	# Architectue is ready.  Make the loss function, and use it to create 
	# the parameter updates responsible for learning
	loss = get_noise_contrastive_loss(embedder.get_output(), batch_size)

	# Optionally exclude the context embeddings (in params[1]) from the
	# parameters to be learnt.  This freezes them to their starting value
	params = embedder.get_params()
	if freeze_context:
		params = [params[0], params[2], params[3]]

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
		reader.save_dictionary(save_dir)

	# Return the trained embedder and the dictionary mapping tokens
	# to ids
	return embedder, reader


def show_computation_graph(
	files=[],
	directories=[],
	skip=[],
	save_dir=None,
	num_epochs=5,
	entity_dictionary=None,
	context_dictionary=None,
	load_dictionary_dir=None,
	min_query_frequency=0,
	min_context_frequency=0,
	noise_ratio=15,
	batch_size = 1000,  # Number of *signal* examples per batch
	macrobatch_size = 100000,
	max_queue_size=0,
	num_embedding_dimensions=500,
	entity_embedding_init=None,
	context_embedding_init=None,
	learning_rate=0.1,
	momentum=0.9,
	verbose=True,
	read_data_async=True,
	num_processes=3
):
	'''
	Builds the same architecture as `relation2vec()`, but does no training,
	instead, it just prints the theano computation graph of the 
	architecture.  This is to help with debugging and optimization.
	'''

	# Make a Relation2VecDatasetReader, pass through parameters sent by 
	# caller
	reader = DatasetReader(
		files=files,
		directories=directories,
		skip=skip,
		macrobatch_size = macrobatch_size,
		max_queue_size = max_queue_size,
		noise_ratio=noise_ratio,
		num_processes=num_processes,
		entity_dictionary=entity_dictionary,
		context_dictionary=context_dictionary,
		load_dictionary_dir=load_dictionary_dir,
		verbose=verbose
	)

	# Make a symbolic minibatcher Note that the full batch includes 
	# noise_ratio number of noise examples for every signal example, and 
	# the parameter "batch_size" here is interpreted as just the number of 
	# signal examples per batch; the full batch size is:
	minibatcher = NoiseContrastiveTheanoMinibatcher(
		batch_size=batch_size,
		noise_ratio=noise_ratio,
		dtype="int32",
		num_dims=2
	)

	# Make a Word2VecEmbedder object, feed it the combined input
	embedder = Relation2VecEmbedder(
		input_var=minibatcher.get_batch(),
		batch_size=batch_size,
		entity_vocab_size=reader.entity_vocab_size(),
		context_vocab_size=reader.context_vocab_size(),
		num_embedding_dimensions=num_embedding_dimensions,
		entity_embedding_init=entity_embedding_init,
		context_embedding_init=context_embedding_init
	)

	# Architectue is ready.  Make the loss function, and use it to create 
	# the parameter updates responsible for learning
	loss = get_noise_contrastive_loss(embedder.get_output(), batch_size)
	updates = nesterov_momentum(
		loss, embedder.get_params(), learning_rate, momentum
	)

	# Include minibatcher updates, which cause the symbolic batch to move
	# through the dataset like a sliding window
	updates.update(minibatcher.get_updates())

	# Use the loss function and the updates to compile a training function.
	# Note that it takes no inputs because the dataset is fully loaded using
	# theano shared variables
	train = function([], loss, updates=updates)

	theano.printing.debugprint(train)
