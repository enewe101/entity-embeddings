from lasagne import layers, get_output, get_all_params, get_all_param_values
from word2vec import NoiseContraster, row_dot, sigmoid

class EntityEmbedder(object):
	def __init__(
		self,
		input_var,
		batch_size,
		entity_vocab_size=10000,
		context_vocab_size=1000000,
		num_embedding_dimensions=500,
		word_embedding_init=Normal(),
		context_embedding_init=Normal()
	):

		self.entity1 = input_var[:,0]
		self.entity2 = input_var[:,1]
		self.context = input_var[:,2]

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
		self.l_merge_embeddings = layers.MergeLayer(
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


	def get_params(self):
		# Note that parameters for l_embed_entity1 are the same as
		# for l_embed_entity2, so we only need to fetch params for one
		return (
			get_all_params(self.l_embed_entity1, trainable=True)
			+ get_all_params(self.l_embed_context, trainable=True)
			+ get_all_params(self.l_relation, trainable=True)
		)


	def get_param_values(self):
		# Note that parameters for l_embed_entity1 are the same as
		# for l_embed_entity2, so we only need to fetch params for one
		return (
			get_all_param_values(self.l_embed_entity1, trainable=True) 
			+ get_all_param_values(self.l_embed_context, trainable=True)
			+ get_all_param_values(self.l_relation, trainable=True)
		)


	def get_output(self):
		return self.output


	def save(self, filename):
		W,C,D = self.get_param_values()
		np.savez(filename, W=W, C=C, D=D)


	def load(self, filename):
		npfile = np.load(filename)
		W_loaded, C_loaded, D_loaded = npfile['W'], npfile['C'], npfile['D']
		W_shared, C_shared, D_shared = self.get_params()
		W_shared.set_value(W_loaded)
		C_shared.set_value(C_loaded)
		D_shared.set_value(D_loaded)



