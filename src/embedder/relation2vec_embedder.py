import numpy as np
from lasagne import layers
from lasagne.layers import get_output, get_all_params, get_all_param_values
from lasagne.init import Normal
from theano import tensor as T, function
from word2vec import NoiseContraster, row_dot, sigmoid


class Relation2VecEmbedder(object):
	def __init__(
		self,
		input_var=None,
		batch_size=None,
		entity_vocab_size=10000,
		context_vocab_size=1000000,
		num_embedding_dimensions=500,
		word_embedding_init=Normal(),
		context_embedding_init=Normal()
	):

		# Register most parameters to self namespace
		self.input_var = input_var
		if input_var is None:
			self.input_var = T.imatrix()

		self.batch_size = batch_size
		self.entity_vocab_size = entity_vocab_size
		self.context_vocab_size = context_vocab_size
		self.num_embedding_dimensions = num_embedding_dimensions

		# Slice the input var into relevant aspects
		self.entity1 = self.input_var[:,0]
		self.entity2 = self.input_var[:,1]
		self.context = self.input_var[:,2]

		# Take in and embed entity1
		self.l_in_entity1 = layers.InputLayer(
			shape=(batch_size,), input_var=self.entity1
		)
		self.l_embed_entity1 = layers.EmbeddingLayer(
			self.l_in_entity1, entity_vocab_size, num_embedding_dimensions
		)

		# Take in and embed entity2 -- note this uses same parameters as
		# embedding for entity1
		self.l_in_entity2 = layers.InputLayer(
			shape=(batch_size,), input_var=self.entity2
		)
		self.l_embed_entity2 = layers.EmbeddingLayer(
			self.l_in_entity2, entity_vocab_size, num_embedding_dimensions,
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
			shape=(batch_size,), input_var=self.context
		)
		self.l_embed_context = layers.EmbeddingLayer(
			self.l_in_context, context_vocab_size, num_embedding_dimensions
		)
		self.context_embedding = get_output(self.l_embed_context)

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
	def save(self, filename):
		W_entity,W_context,W_relation,b_relation = self.get_param_values()
		np.savez(
			filename, 
			W_entity=W_entity,
			W_context=W_context,
			W_relation=W_relation,
			b_relation=b_relation
		)


	#TODO: test
	def load(self, filename):
		npfile = np.load(filename)
		W_entity_ = npfile['W_entity']
		W_context_ = npfile['W_context']
		W_relation_ = npfile['W_relation']
		b_relation_ = npfile['b_relation']

		W_entity, W_context, W_relation, b_relation = self.get_params()
		W_entity.set_value(W_entity_)
		W_context.set_value(W_context_)
		W_relation.set_value(W_relation_)
		b_relation.set_value(b_relation_)



