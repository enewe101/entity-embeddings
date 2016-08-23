import os
import numpy as np
from word2vec import get_noise_contrastive_loss, row_dot, sigmoid


# Possibly import theano and lasagne
exclude_theano_set = 'EXCLUDE_THEANO' in os.environ 
if exclude_theano_set and int(os.environ['EXCLUDE_THEANO']) == 1:
	pass
else:
	from lasagne import layers
	from lasagne.layers import (
		get_output, get_all_params, get_all_param_values
	)
	from lasagne.init import Normal
	from theano import tensor as T, function


class Relation2VecEmbedder(object):

	def __init__(
		self,
		input_var=None,
		batch_size=None,
		entity_vocab_size=10000,
		context_vocab_size=1000000,
		num_embedding_dimensions=500,
		entity_embedding_init=None,
		context_embedding_init=None,
		len_context=1
	):

		# Register most parameters to self namespace
		self.input_var = input_var
		if input_var is None:
			self.input_var = T.imatrix()

		if entity_embedding_init is None:
			entity_embedding_init = Normal()

		if context_embedding_init is None:
			context_embedding_init = Normal()

		self.batch_size = batch_size
		self.entity_vocab_size = entity_vocab_size
		self.context_vocab_size = context_vocab_size
		self.num_embedding_dimensions = num_embedding_dimensions
		self.len_context = len_context

		# Slice the input var into relevant aspects
		self.entity1 = self.input_var[:,0]
		self.entity2 = self.input_var[:,1]
		self.context = self.input_var[:,2:]

		# Take in and embed entity1
		self.l_in_entity1 = layers.InputLayer(
			shape=(batch_size,), input_var=self.entity1
		)
		self.l_embed_entity1 = layers.EmbeddingLayer(
			incoming=self.l_in_entity1, 
			input_size=entity_vocab_size, 
			output_size=num_embedding_dimensions,
			W=entity_embedding_init
		)

		# Take in and embed entity2 -- note this uses same parameters as
		# embedding for entity1
		self.l_in_entity2 = layers.InputLayer(
			shape=(batch_size,), input_var=self.entity2
		)
		self.l_embed_entity2 = layers.EmbeddingLayer(
			incoming=self.l_in_entity2,
			input_size=entity_vocab_size, 
			output_size=num_embedding_dimensions,
			W=self.l_embed_entity1.W
		)

		# Merge the entities and embed their relationship
		self.l_merge_entities = layers.ConcatLayer(
			(self.l_embed_entity1, self.l_embed_entity2), axis=1
		)
		self.l_relation = layers.DenseLayer(
			self.l_merge_entities, num_embedding_dimensions
		)
		self.relation_embedding = get_output(self.l_relation)

		# Take in and embed the context
		self.l_in_context = layers.InputLayer(
			shape=(batch_size,len_context), input_var=self.context
		)
		self.l_embed_context = layers.EmbeddingLayer(
			self.l_in_context, context_vocab_size, num_embedding_dimensions,
			W=context_embedding_init
		)
		self.prelim_context_embedding = get_output(self.l_embed_context)

		# Average the embedding accros the context
		self.context_embedding = self.prelim_context_embedding.sum(
			axis=1) / float(len_context)

		# Compute the network output
		self.output = sigmoid(row_dot(
			self.relation_embedding, self.context_embedding
		))

		self._embed_relationship = None


	def embed_relationship(self, entity_pairs):
		'''
		Get the embedding(s) for the entity_pair(s) by calling the
		compiled theano function.  The function has to be compiled the
		first time this is called.
		'''

		if self._embed_relationship is None:
			self._compile_embed_relationship()
		return self._embed_relationship(entity_pairs)


	def _compile_embed_relationship(self):
		'''
		Compiles a function that computes the relationship embedding
		given two entities (i.e. it does a partial forward pass,
		not including the part of the network dedicated to the context
		'''

		input_var = T.imatrix('entities')
		entity1 = input_var[:,0]
		entity2 = input_var[:,1]

		# Take in and embed entity1
		l_in_entity1 = layers.InputLayer(
			shape=(self.batch_size,), input_var=entity1
		)
		l_embed_entity1 = layers.EmbeddingLayer(
			l_in_entity1,
			self.entity_vocab_size,
			self.num_embedding_dimensions,
			W=self.l_embed_entity1.W
		)

		# Take in and embed entity2 -- note this uses same parameters as
		# embedding for entity1
		l_in_entity2 = layers.InputLayer(
			shape=(self.batch_size,), input_var=entity2
		)
		l_embed_entity2 = layers.EmbeddingLayer(
			l_in_entity2,
			self.entity_vocab_size,
			self.num_embedding_dimensions,
			W=l_embed_entity1.W	# recall, the entity embedders share params
		)

		# Merge the entities and embed their relationship
		l_merge_entities = layers.ConcatLayer(
			(l_embed_entity1, l_embed_entity2), axis=1
		)
		l_relation = layers.DenseLayer(
			l_merge_entities, self.num_embedding_dimensions,
			W=self.l_relation.W, b=self.l_relation.b
		)
		relation_embedding = get_output(l_relation)

		# Compile the function
		self._embed_relationship = function([input_var], relation_embedding)


	def get_params(self):
		# Note that parameters for l_embed_entity1 are the same as
		# for l_embed_entity2, so we only need to fetch params for one
		# Return value looks like this:
		#
		#	[
		#		entity-embedding params,
		#		context-embedding params,
		#		relation-embedding weights,
		#		relation-embedding biases
		#	]
		return (
			self.l_embed_entity1.get_params(trainable=True)
			+ self.l_embed_context.get_params(trainable=True)
			+ self.l_relation.get_params(trainable=True)
		)


	def get_param_values(self):
		# Note that parameters for l_embed_entity1 are the same as
		# for l_embed_entity2, so we only need to fetch params for one
		# Return value looks like this:
		#
		#	[
		#		entity-embedding params,
		#		context-embedding params,
		#		relation-embedding weights,
		#		relation-embedding biases
		#	]

		return [p.get_value() for p in self.get_params()]


	def get_output(self):
		return self.output


	#TODO: test
	def save(self, directory):
		'''
		Saves the model parameters (embeddings) to disk, in a file called
		"embeddings.npz" under the directory given.
		'''

		# We are willing to create the directory given if it doesn't exist
		if not os.path.exists(directory):
			os.mkdir(directory)

		# Save under the directory given in a file called "embeddings.npz'
		save_path = os.path.join(directory, "embeddings.npz")

		# Get the parameters and save them to disk
		W_entity,W_context,W_relation,b_relation = self.get_param_values()
		np.savez(
			save_path,
			W_entity=W_entity,
			W_context=W_context,
			W_relation=W_relation,
			b_relation=b_relation
		)


	#TODO: test
	def load(self, directory):
		'''
		Loads the model parameter values (embeddings) stored in the
		directory given.  Expects to find the parameters in a file called
		"embeddings.npz" within the directory given.
		'''

		# By default, we save to a file called "embeddings.npz" within the
		# directory given to the save function.
		save_path = os.path.join(directory, "embeddings.npz")

		# Load the parameters
		npfile = np.load(save_path)
		W_entity_ = npfile['W_entity']
		W_context_ = npfile['W_context']
		W_relation_ = npfile['W_relation']
		b_relation_ = npfile['b_relation']

		# Set the shared variables of the model (which reside on GPU)
		W_entity, W_context, W_relation, b_relation = self.get_params()
		W_entity.set_value(W_entity_)
		W_context.set_value(W_context_)
		W_relation.set_value(W_relation_)
		b_relation.set_value(b_relation_)

