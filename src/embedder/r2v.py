from word2vec import get_noise_contrastive_loss
from dataset_reader import Relation2VecDatasetReader as DatasetReader
from relation2vec_embedder import Relation2VecEmbedder
from theano_minibatcher import NoiseContrastiveTheanoMinibatcher
from lasagne.init import Normal
from lasagne.updates import nesterov_momentum
from theano import function
import os

def relation2vec(
	files=[],
	directories=[],
	skip=[],
	save_dir=None,
	num_epochs=5,
	entity_dictionary=None,
	context_dictionary=None,
	noise_ratio=15,
	batch_size = 1000,  # Number of *signal* examples per batch
	num_embedding_dimensions=500,
	word_embedding_init=Normal(),
	context_embedding_init=Normal(),
	learning_rate=0.1,
	momentum=0.9,
	verbose=True,
	num_processes=3
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

	# Make a Relation2VecDatasetReader, pass through parameters sent by caller
	reader = DatasetReader(
		files=files,
		directories=directories,
		skip=skip,
		batch_size=batch_size, # number of *signal_examples* per batch
		noise_ratio=noise_ratio,
		num_processes=num_processes,
		entity_dictionary=None,
		context_dictionary=None,
		verbose=verbose
	)

	# Prpare the minibatch generator
	# (this produces the counter_sampler stats)
	reader.prepare(save_dir=save_dir)

	# Make a symbolic minibatcher
	# Note that the full batch includes noise_ratio number of noise examples#
	# for every signal example, and the parameter "batch_size" here is interpreted
	# as just the number of signal examples per batch; the full batch size is:
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
		word_embedding_init=word_embedding_init,
		context_embedding_init=context_embedding_init
	)

	# Architectue is ready.  Make the loss function, and use it to create the
	# parameter updates responsible for learning
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

	# Generate the full dataset, and load it onto the GPU
	dataset = reader.generate_dataset_parallel(save_dir=save_dir)
	minibatcher.load_dataset(*dataset)

	# Iterate through the dataset, training the embeddings
	for epoch in range(num_epochs):
		if verbose:
			print 'starting epoch %d' % epoch
		losses = []
		minibatcher.reset()
		for batch_num in range(minibatcher.get_num_batches()):
			losses.append(train())
		if verbose:
			print '\tAverage loss: %f' % np.mean(losses)

	# Save the model (the embeddings) if save_dir was provided
	# TODO: this should save to a subdir and should make save_dir if necessary
	if save_dir is not None:
		embeddings_filename = os.path.join(save_dir, 'embeddings.npz')
		embedder.save(embeddings_filename)

	# Return the trained embedder and the dictionary mapping tokens
	# to ids
	return embedder, reader
