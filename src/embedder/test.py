from word2vec import UNK
from theano import tensor as T, function
import numpy as np
import os
import shutil
from collections import defaultdict, Counter
from relation2vec_embedder import Relation2VecEmbedder
from word2vec import get_noise_contrastive_loss, reseed
from lasagne.updates import nesterov_momentum
import itertools as itools
import unittest
from r2v import relation2vec, read_context_embeddings
from dataset_reader import (
	Relation2VecDatasetReader, DataSetReaderIllegalStateException,
	FULL_CONTEXT
)
from theano_minibatcher import NoiseContrastiveTheanoMinibatcher

#from minibatcher import (
#	Relation2VecMinibatcher, relation2vec_parse, word2vec_parse,
#	Word2VecMinibatcher
#)
from unittest import TestCase, main

@np.vectorize
def sigma(a):
	return 1 / (1 + np.exp(-a))


# TODO: make this a real test
class TestParse(TestCase):

	def test_parse(self):
		filename = 'test-data/test-corpus/004-raw.tsv'


class TestRelation2VecEmbedder(TestCase):

	def test_read_embeddings(self):

		# Some constants for the test
		files = ['test-data/test-corpus/a.tsv']
		num_embedding_dimensions = 300
		embeddings_fname = 'test-data/pre-trained-embeddings.txt'

		# Make a dataset reader
		reader = Relation2VecDatasetReader(
			files=files,
			verbose=False
		)
		reader.prepare()

		# Make a Word2VecEmbedder object
		embedder = Relation2VecEmbedder(
			entity_vocab_size=reader.entity_vocab_size(),
			context_vocab_size=reader.context_vocab_size(),
			num_embedding_dimensions=num_embedding_dimensions,
		)

		# Make a dictionary of the embeddings in the text file, which
		# we'll use to check whether the embeddings were loaded properly
		pretrained_embeddings = {}
		first_line = True
		for line in open(embeddings_fname):

			# skip the first line
			if first_line:
				first_line = False
				continue

			# Parse the line and store the vector in the dictionary
			fields = line.strip().split()
			token, vector = fields[0], np.array(fields[1:], dtype='float32')
			token_id = reader.context_dictionary.get_id(token)
			pretrained_embeddings[token_id] = vector

		# Get the original embeddings
		original_embeddings = embedder.get_params()[1].get_value()

		# Read in the embeddings from the text file
		read_context_embeddings(embeddings_fname, reader, embedder)

		# Get the embeddings after they've been read from file
		loaded_embeddings = embedder.get_params()[1].get_value()

		# Verify that, each vector in loaded embeddings either matches
		# the pretrained vector (if one existed for that token) or matches
		# the orignal embeddings (if there was not pretrained embedding)
		for token_id, vector in enumerate(loaded_embeddings):
			if token_id in pretrained_embeddings:
				self.assertTrue(
					np.allclose(pretrained_embeddings[token_id], vector)
				)
			else:
				self.assertTrue(
					np.allclose(original_embeddings[token_id], vector)
				)



	def test_save_load(self):

		# Make sure that the target directory exists, but delete any model
		# files left over from a previous test run.
		save_dir = 'test-data/test-embedder'
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)

		embedder = Relation2VecEmbedder(
			entity_vocab_size=10,
			context_vocab_size=50,
			num_embedding_dimensions=5
		)
		expected_params = embedder.get_param_values()
		embedder.save(save_dir)

		new_embedder = Relation2VecEmbedder(
			entity_vocab_size=10,
			context_vocab_size=50,
			num_embedding_dimensions=5
		)
		new_embedder.load(save_dir)
		found_params = new_embedder.get_param_values()

		for found, expected in zip(found_params, expected_params):
			self.assertTrue(np.array_equal(found, expected))

		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)


	def prepare_toy_dataset(self, noise_ratio):

		dataset = []
		dataset_repetitions = 1024 * 2
		for i in range(dataset_repetitions):
			signal_minibatch = [
				[1,2,2],
				[3,4,5],
				[5,6,8],
			]
			noise_minibatch = (
				[[1,2,i] for i in np.random.randint(0,10,noise_ratio)]
				+ [[3,4,i] for i in np.random.randint(0,10,noise_ratio)]
				+ [[5,6,i] for i in np.random.randint(0,10,noise_ratio)]
			)
			minibatch = signal_minibatch + noise_minibatch
			dataset.append(minibatch)

		return dataset


	def test_learning(self):

		np.random.seed(1)

		# Some constants for the test
		batch_size = 3
		noise_ratio = 15
		num_embedding_dimensions = 5
		num_epochs = 1
		num_replicates=10
		learning_rate = 0.01
		momentum = 0.9
		tolerance = 0.1
		save_dir = 'test-data/test-entity-embedder'

		# Seed randomness for a reproducible test.  Using 2 because
		# 1 was particularly bad by chance

		symbolic_batch = T.imatrix()

		# We'll make and train an Relation2VecEmbedder in a moment.  However,
		# first we will get the IDs for the entities that occur together
		# within the test corpus.  We'll be interested to see the
		# relationship embedding for them that is learnt during training
		expected_pairs_ids = [[1,2],[3,4],[5,6]]

		# We will repeatedly make a Relation2VecEmbedder, train it on the
		# test corpus, and then find its embedding for the entity-pairs
		# of interest.  We do it num_replicate # of times to average
		# results over enough trials that we can test if the learnt
		# embeddings have roughly the expected properties
		embedding_dot_products  = []
		for replicate in range(num_replicates):

			dataset = self.prepare_toy_dataset(noise_ratio)

			# Make a Word2VecEmbedder object, feed it the combined input
			embedder = Relation2VecEmbedder(
				input_var=symbolic_batch,
				batch_size=batch_size,
				entity_vocab_size=7,
				context_vocab_size=10,
				num_embedding_dimensions=num_embedding_dimensions,
			)
			# Get the parameters out of the trained model
			W_entity, W_context, W_relation, b_relation = (
				embedder.get_param_values()
			)

			# Architectue is ready.  Make the loss function, and use it to create the
			# parameter updates responsible for learning
			loss = get_noise_contrastive_loss(embedder.get_output(), batch_size)
			updates = nesterov_momentum(
				loss, embedder.get_params(), learning_rate, momentum
			)
			train = function([symbolic_batch], loss, updates=updates)

			# Train on the dataset, running through it num_epochs # of times
			for epoch in range(num_epochs):

				for minibatch in dataset:
					loss = train(minibatch)

			# Get the parameters out of the trained model
			W_entity, W_context, W_relation, b_relation = (
				embedder.get_param_values()
			)

			# Get the embeddings for the entity-pairs ("relationships")
			# occuring in the test corpus
			embedded_relationships = embedder.embed_relationship(
				expected_pairs_ids
			)

			# Take the dot product of the relationship embeddings
			# with the context-word embeddings, and then process this
			# through the sigmoid function.  This yields a
			# relationship-context "fit-score", being larger if they
			# have better fit according to the model.  The score has
			# an interpretation as a probablity, see "Noise-Contrastive
			# Estimation of Unnormalized Statistical Models, with
			# Applications to Natural Image Statistics".
			embedding_dot_product = np.round(sigma(np.dot(
				embedded_relationships, W_context.T
			)),2)

			# Accumulate fit scores over the replicates
			embedding_dot_products.append(embedding_dot_product)

		# Average the fit-scores over the replicates
		avg_embedding_product = np.mean(embedding_dot_products, axis=0)

		# Find the context words that fit best with each entity-pair.
		# These should be equal to the ids for the contexts actually
		# occuring along with those entity pairs in the coropus.  To see
		# where these numbers come from, compare the contexts cooccurring
		# with entity pairs in the test-corpus with their ids in the
		# token_map saved at <savedir>
		expected_best_fitting_context_ids = [2,5,8]
		best_fitting_context_ids = np.argmax(avg_embedding_product, axis=1)
		self.assertTrue(np.array_equal(
			best_fitting_context_ids, expected_best_fitting_context_ids
		))

		# Get the probability accorded to each entity pair and the
		# context that they fit best with
		predicted_fit = np.max(avg_embedding_product, axis=1)

		# The optimal probability accorded to entity_pairs and contexts
		# should be 0.375 for the entity pairs with their best fitting
		# contexts (this is according to the
		# formulation of the loss function as part of Noise Contrastive
		# Estimation, specifically, this is p(C=1 | (e1,e2,c)),
		# see "Noise-Contrastive Estimation of Unnormalized Statistical
		# Models, with Applications to Natural Image Statistics".
		expected_fit = 0.375

		# The predictions should be "close" to the expected values
		differences = [
			abs(predicted - expected_fit)
			for predicted in predicted_fit
		]
		self.assertTrue(all([diff < tolerance for diff in differences]))


	def test_run_e2e(self):

		# Seed randomness for a reproducible test.  Using 2 because
		# 1 was particularly bad by chance
		np.random.seed(1)

		# Some constants for the test
		files = ['test-data/test-corpus/c.tsv']
		batch_size = 3
		macrobatch_size = 3102 #
		noise_ratio = 15
		num_embedding_dimensions = 5
		num_epochs = 1
		num_replicates=10
		learning_rate = 0.01
		momentum = 0.9
		tolerance = 0.25
		save_dir = 'test-data/test-entity-embedder'

		# Make a minibatcher to yield training batches from test corpus
		reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size=macrobatch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		reader.prepare(save_dir=save_dir)

		# We'll make and train an Relation2VecEmbedder in a moment. 
		# However, first we will get the IDs for the entities that occur 
		# together within the test corpus.  We'll be interested to see the
		# relationship embedding for them that is learnt during training
		edict = reader.entity_dictionary
		expected_pairs = [
			('A','B'), ('C','D'), ('E','F')
		]
		expected_pairs_ids = [
			(edict.get_id(e1), edict.get_id(e2))
			for e1, e2 in expected_pairs
		]

		# We will repeatedly make a Relation2VecEmbedder, train it on the
		# test corpus, and then find its embedding for the entity-pairs
		# of interest.  We do it num_replicate # of times to average
		# results over enough trials that we can test if the learnt
		# embeddings have roughly the expected properties
		embedding_dot_products  = []

		# Make a symbolic minibatcher
		# Note that the full batch includes noise_ratio number of noise 
		# examples for every signal example, and the parameter "batch_size"
		# here is interpreted as just the number of signal examples per 
		# batch; the full batch size is:
		minibatcher = NoiseContrastiveTheanoMinibatcher(
			batch_size=batch_size,
			noise_ratio=noise_ratio,
			dtype="int32",
			num_dims=2
		)

		for replicate in range(num_replicates):

			# Make a Word2VecEmbedder object, feed it the combined input
			embedder = Relation2VecEmbedder(
				input_var=minibatcher.get_batch(),
				batch_size=batch_size,
				entity_vocab_size=reader.entity_vocab_size(),
				context_vocab_size=reader.context_vocab_size(),
				num_embedding_dimensions=num_embedding_dimensions,
			)

			# Architectue is ready. Make the loss function, and use it 
			# to create the parameter updates responsible for learning
			loss = get_noise_contrastive_loss(
				embedder.get_output(), batch_size)
			updates = nesterov_momentum(
				loss, embedder.get_params(), learning_rate, momentum
			)
			# Include minibatcher updates, which cause the symbolic 
			# batch to move through the dataset like a sliding window
			updates.update(minibatcher.get_updates())

			# Use the loss function and the updates to compile a training 
			# function. Note that it takes no inputs because the dataset is
			# fully loaded using theano shared variables
			train = function([], loss, updates=updates)

			# Train on the dataset, running through it num_epochs 
			# number of times
			for epoch in range(num_epochs):

				macrobatches = reader.generate_dataset_parallel()
				for signal_data, noise_data in macrobatches:
					num_batches = minibatcher.load_dataset(
						signal_data, noise_data)
					for batch_num in range(num_batches):
						loss = train()

			# Get the parameters out of the trained model
			W_entity, W_context, W_relation, b_relation = (
				embedder.get_param_values()
			)

			# Get the embeddings for the entity-pairs ("relationships")
			# occuring in the test corpus
			embedded_relationships = embedder.embed_relationship(
				expected_pairs_ids
			)

			# Take the dot product of the relationship embeddings
			# with the context-word embeddings, and then process this
			# through the sigmoid function.  This yields a
			# relationship-context "fit-score", being larger if they
			# have better fit according to the model.  The score has
			# an interpretation as a probablity, see "Noise-Contrastive
			# Estimation of Unnormalized Statistical Models, with
			# Applications to Natural Image Statistics".
			embedding_dot_product = np.round(sigma(np.dot(
				embedded_relationships, W_context.T
			)),2)

			# Accumulate fit scores over the replicates
			embedding_dot_products.append(embedding_dot_product)

		# Average the fit-scores over the replicates
		avg_embedding_product = np.mean(embedding_dot_products, axis=0)

		# Find the context words that fit best with each entity-pair.
		# These should be equal to the ids for the contexts actually
		# occuring along with those entity pairs in the coropus.  To see
		# where these numbers come from, compare the contexts cooccurring
		# with entity pairs in the test-corpus with their ids in the
		# token_map saved at <savedir>
		expected_best_fitting_context_ids = [2,5,8]

		# The UNK token generally has high affinity for every relation 
		# embedding because there is no strong gradient affecting it.  We 
		# will mask the UNK before looking for best-fitting learned context
		# for each relationship.
		avg_embedding_product[:,0] = 0
		best_fitting_context_ids = np.argmax(avg_embedding_product, axis=1)
		self.assertTrue(np.array_equal(
			best_fitting_context_ids, expected_best_fitting_context_ids
		))

		# Get the probability accorded to each entity pair and the
		# context that they fit best with
		predicted_fit = np.max(avg_embedding_product, axis=1)

		# The optimal probability accorded to entity_pairs and contexts
		# should be 0.375 for the entity pairs with their best fitting
		# contexts (this is according to the
		# formulation of the loss function as part of Noise Contrastive
		# Estimation, specifically, this is p(C=1 | (e1,e2,c)),
		# see "Noise-Contrastive Estimation of Unnormalized Statistical
		# Models, with Applications to Natural Image Statistics".
		expected_fit = 0.375

		# The predictions should be "close" to the expected values
		differences = [
			abs(predicted - expected_fit)
			for predicted in predicted_fit
		]
		self.assertTrue(all([diff < tolerance for diff in differences]))


	def test_learning_function(self):

		# Seed randomness for a reproducible test.  Using 2 because
		# 1 was particularly bad by chance
		np.random.seed(3)

		# Some constants for the test
		files = ['test-data/test-corpus/c.tsv']
		batch_size = 3
		macrobatch_size = 1551
		noise_ratio = 15
		num_embedding_dimensions = 5
		num_epochs = 2
		num_replicates = 5
		learning_rate = 0.01
		momentum = 0.9
		tolerance = 0.25
		save_dir = 'test-data/test-entity-embedder'

		# Train the embedder using the convenience function
		embedder, reader = relation2vec(
			files=files,
			save_dir=save_dir,
			num_epochs=num_epochs,
			noise_ratio=noise_ratio,
			batch_size = batch_size,
			macrobatch_size = macrobatch_size,
			num_embedding_dimensions=num_embedding_dimensions,
			learning_rate=learning_rate,
			momentum=momentum,
			verbose=False
		)

		# Get the IDs for the entities that occur together
		# within the test corpus.  We'll be interested to see the
		# relationship embedding for them that is learnt during training
		edict = reader.entity_dictionary
		expected_pairs = [
			('A','B'), ('C','D'), ('E','F')
		]
		expected_pairs_ids = [
			(edict.get_id(e1), edict.get_id(e2))
			for e1, e2 in expected_pairs
		]

		# Get the parameters out of the trained model
		W_entity, W_context, W_relation, b_relation = (
			embedder.get_param_values()
		)

		# Get the embeddings for the entity-pairs ("relationships")
		# occuring in the test corpus
		embedded_relationships = embedder.embed_relationship(
			expected_pairs_ids
		)

		# Take the dot product of the relationship embeddings
		# with the context-word embeddings, and then process this
		# through the sigmoid function.  This yields a
		# relationship-context "fit-score", being larger if they
		# have better fit according to the model.  The score has
		# an interpretation as a probablity, see "Noise-Contrastive
		# Estimation of Unnormalized Statistical Models, with
		# Applications to Natural Image Statistics".
		embedding_product = np.round(sigma(np.dot(
			embedded_relationships, W_context.T
		)),2)

		# Find the context words that fit best with each entity-pair.
		# These should be equal to the ids for the contexts actually
		# occuring along with those entity pairs in the coropus.  To see
		# where these numbers come from, compare the contexts cooccurring
		# with entity pairs in the test-corpus with their ids in the
		# token_map saved at <savedir>
		expected_best_fitting_context_ids = [2,5,8]
		# The UNK token generally has high affinity for every relation 
		# embedding because there is no strong gradient affecting it.  We 
		# will mask the UNK before looking for best-fitting learned context
		# for each relationship.
		embedding_product[:,0] = 0
		best_fitting_context_ids = np.argmax(embedding_product, axis=1)
		self.assertTrue(np.array_equal(
			best_fitting_context_ids, expected_best_fitting_context_ids
		))


class TestNoiseContrastiveTheanoMinibatcher(TestCase):

	def test_symbolic_minibatches(self):
		'''
		The symbolic minibatching mechanism should yield the exact same
		set of examples as the non-symbolic mechanism.  Here we compare
		them to make sure that is the case.  In turn, the non-symbolic
		minibatching method is tested to ensure it has the expected content
		as should be produced from the test data.  So this is not a "unit"
		test in the sense that it has dependency on other tests in this
		TestCase.
		'''
		# Ensure reproducibility in this stochastic test
		np.random.seed(1)

		# Make a minibatcher
		batch_size = 5
		noise_ratio = 9
		minibatcher = NoiseContrastiveTheanoMinibatcher(
			batch_size=batch_size,
			noise_ratio=noise_ratio,
			dtype="int32",
			num_dims=2
		)

		# mock a dataset
		num_signal_examples = 50
		signal_examples = np.array([
			[i,i,i] for i in range(num_signal_examples)
		], dtype='int32')
		noise_examples = np.array([
			[j,j,j] for j in range(num_signal_examples * noise_ratio)
		], dtype='int32')

		# Get the symbolic minibatch
		num_batches = minibatcher.load_dataset(signal_examples, noise_examples)
		minibatch = minibatcher.get_batch()
		updates = minibatcher.get_updates()

		self.assertEqual(num_batches, num_signal_examples / batch_size)

		f = function([], minibatch, updates=updates)
		for batch_num in range(num_batches):
			expected_batch = np.array(
				[[j,j,j] for j in range(batch_num*batch_size, (batch_num+1) * batch_size)]
				+ [[j,j,j] for j in range(batch_num*batch_size*noise_ratio, (batch_num+1)*batch_size*noise_ratio)]
				, dtype='int32'
			)
			found_batch = f()
			self.assertTrue(np.array_equal(found_batch, expected_batch))



	def test_minibatches_from_corpus(self):
		'''
		Test that the symbolic minibatching mechanism, used in tandem with 
		the DatsetReader, produces the expected training examples in its
		minibatches.
		'''
		# Ensure reproducibility in this stochastic test
		np.random.seed(1)

		# Make a DatasetReader
		files = ['test-data/test-corpus/a.tsv']
		batch_size = 5
		noise_ratio = 9
		reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size = 140,
			noise_ratio=noise_ratio,
			signal_sample_mode=FULL_CONTEXT,
			verbose=False
		)
		reader.prepare()

		# Read through the file and determine all of the entity pairs that
		# should be found, and the valid context tokens that go with them
		valid_contexts = defaultdict(set)
		filename='test-data/test-corpus/a.tsv'
		for line in reader.parse(filename):
			context_tokens, entity_spans = line

			context_ids = reader.context_dictionary.get_ids(
				context_tokens
			)

			for e1, e2 in itools.combinations(entity_spans, 2):

				# Give a strict order to make assertions easier
				e1_id, e2_id = reader.entity_dictionary.get_ids(
					[e1,e2])

				filtered_context_ids = reader.eliminate_spans(
					context_ids, entity_spans[e1]+entity_spans[e2]
				)

				# Adopt ordering convention of entities in entity pairs 
				# when using them as dictionary keys.  (Makes it easier to 
				# write assertions.)
				if e1_id > e2_id:
					e1, e2 = e2, e1
					e1_id, e2_id = e2_id, e1_id

				valid_contexts[(e1_id, e2_id)].update(filtered_context_ids)

		seen_contexts = defaultdict(set)
		macrobatches = reader.generate_dataset_serial()

		# Make a minibatcher
		batch_size = 5
		noise_ratio = 9
		minibatcher = NoiseContrastiveTheanoMinibatcher(
			batch_size=batch_size,
			noise_ratio=noise_ratio,
			dtype="int32",
			num_dims=2
		)

		minibatch = minibatcher.get_batch()
		updates = minibatcher.get_updates()

		# Define a theano function that simply extracts the minibatch 
		# and advances the minibatch pointer
		f = function([], minibatch, updates=updates)
		for signal_examples, noise_examples in macrobatches:

			# Verify the relative size of the signal and noise macrobatches
			self.assertEqual(
				len(noise_examples), len(signal_examples) * noise_ratio
			)

			# Get the symbolic minibatch
			num_batches = minibatcher.load_dataset(
				signal_examples, noise_examples)

			for batch_num in range(num_batches):
				minibatch = f()
				signal_minibatch = minibatch[:batch_size,]
				for e1_id, e2_id, context_id in signal_minibatch:

					# Adopt ordering convention of entities in entity 
					# pairs when using them as dictionary keys.  (Makes 
					# it easier to write assertions.)
					if e1_id > e2_id:
						e1_id, e2_id = e2_id, e1_id

					seen_contexts[(e1_id, e2_id)].add(context_id)
	
		del seen_contexts[(UNK,UNK)]
		self.assertEqual(seen_contexts, valid_contexts)



class TestDatasetReader(TestCase):

	def test_sample_between_entities(self):
		reader = Relation2VecDatasetReader()
		fname = 'test-data/test-corpus/003-raw.tsv'
		entity_pairs_to_test = [
			('YAGO:Hong_Kong', 'YAGO:Po_Sang_Bank'),
			None,
			('YAGO:Israel', 'YAGO:Lebanon'),
			('YAGO:Hezbollah', 'YAGO:Bint_Jbeil'),
			('YAGO:Israel', 'YAGO:United_Nations'),
			('YAGO:China', 'YAGO:Asian_Games'),
			None,
			None,
			None,
			('YAGO:United_Arab_Emirates', 'YAGO:South_Korea')
		]
		expected_tokens = [
			[',', 'one', 'of', 'the', 'major', 'gold', 'dealers', 'in'],
			None,
			['fighter', 'bombers', 'visiting'],
			['armed', 'wing', ',', 'the', 'Islamic', 'Resistance', ',', 
				'fired', 'on', 'the', 'warplanes', 'as', 'they', 
				'overflew', 'the', 'southern', 'region', 'of'],
			['drew', 'between', 'the', 'two', 'countries', 'after'],
			['continued', 'to', 'flaunt', 'its', 'all-round', 'shooting', 
				'power', 'as', 'it', 'surged', 'into', 'the', 'semifinals',
				'of', 'the', 'men', "'s", 'basketball', 'competition', 
				'with', 'a', '106-55', 'defeat', 'of', 'Kazakstan', 'at', 
				'the'],
			None,
			None,
			None,
			[',', 'which', 'won', 'its', 'Pool', 'B', 'game', '106-51', 
				'against']
		]

		i = -1
		for line, pair in zip(reader.parse(fname),entity_pairs_to_test):
			i += 1

			if pair is None:
				continue

			tokens, entity_spans = line
			e1, e2 = pair
			e_spans1, e_spans2 = entity_spans[e1], entity_spans[e2]
			intervening_tokens = reader.find_tokens_between_closest_pair(
				e_spans1, e_spans2)
			intervening_tokens = [tokens[j] for j in intervening_tokens]
			self.assertEqual(intervening_tokens, expected_tokens[i])
			print expected_tokens[i]



	def test_raises_not_implemented(self):
		'''
		Ensure that the reader raises an error if `get_vocab_size()` is 
		called.'
		'''

		# Some constants for the test
		files = ['test-data/test-corpus/c.tsv']
		batch_size = 3
		noise_ratio = 15

		# Make a minibatcher to yield training batches from test corpus
		reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size=batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		reader.prepare()

		# Ensure that the reader raises an error if `get_vocab_size() is 
		# called`
		with self.assertRaises(NotImplementedError):
			reader.get_vocab_size()


	def test_generate_entity_noise_from_corpus(self):
		'''
		Test that the noise generated from signal examples introduces
		random entities, in addition to random context tokens, when 
		the DatasetReader is configured to do so by constructor arg
		'''
		np.random.seed(1)
		tolerance = 0.1
		noise_ratio = 20
		entity_noise_ratio = 0.5
		files = ['test-data/test-corpus/c.tsv']
		reader = Relation2VecDatasetReader(
			files=files,
			noise_ratio=noise_ratio,
			entity_noise_ratio=entity_noise_ratio,
			verbose=False
		)


		# We need the reader to have dictionaries before we can generate
		# noise.
		reader.prepare()

		# Get the size of the vocabularies.  We are testing for the fact
		# that noise examples have random selections of entities and
		# contexts, and this lets us know how prevalent any given choice
		# should be
		entity_vocab_size = reader.entity_vocab_size()
		context_vocab_size = reader.context_vocab_size()

		num_contexts_differ = 0
		num_e1_differ = 0
		num_e2_differ = 0
		num_signal_examples = 0
		for macrobatch in reader.generate_dataset_serial():
			signal_macrobatch, noise_macrobatch = macrobatch
			chunked_by_signal_example = [
				noise_macrobatch[i:i+noise_ratio] 
				for i in range(0, len(noise_macrobatch), noise_ratio)
			]

			# Go through all the generated noise examples.  See how much 
			# they # differ from signal examples from which they were 
			# generated. Noise examples should only differ in one of either
			# entity1, entity2, or context (verify that this is the case). 
			# Keep track of how often each of these has been substituted by
			# a random choice, and check that it is close to the expected 
			# num.
			for i in range(len(signal_macrobatch)):

				signal_example = signal_macrobatch[i]
				noise_examples = chunked_by_signal_example[i]

				# Skip examples added to bad macrobatch
				if np.array_equal(signal_example, [UNK,UNK,UNK]):
					continue

				num_signal_examples += 1

				for noise_example in noise_examples:

					# Check if entity 1 was changed
					e1_diff = False
					if noise_example[0] != signal_example[0]:
						e1_diff = True
						num_e1_differ += 1

					# Check if entity 2 was changed
					e2_diff = False
					if noise_example[1] != signal_example[1]:
						e2_diff = True
						num_e2_differ += 1

					# Check if the context token was changed
					context_diff = False
					if noise_example[2] != signal_example[2]:
						context_diff = True
						num_contexts_differ += 1

					# Verify that only one of the entities or context is 
					# different from the signal example
					num_different = sum([e1_diff, e2_diff, context_diff])
					self.assertTrue(num_different < 2)

		# The number of contexts that should differ is almost equal to the
		# number of contexts that get randomly selected
		expected_num_contexts_differ = (
			num_signal_examples * noise_ratio * (1-entity_noise_ratio) 
		)
		# but is slightly smaller because the original context gets chosen
		# by chance sometimes
		expected_num_contexts_differ = (
			expected_num_contexts_differ 
			* (1 - 1 / float(context_vocab_size))
		)

		# Similarly for expected num of entities that differ, but 
		# the number of times entities differ is divided between the 
		# two possible choices for entity
		expected_num_entities_differ = (
			num_signal_examples * noise_ratio * entity_noise_ratio
			* (1 - 1 / float(context_vocab_size)) * 0.5
		)

		# Verify that close to the expected number of contexts got 
		# replaced by random contexts
		contexts_diff = abs(
			num_contexts_differ - expected_num_contexts_differ
		) / expected_num_contexts_differ
		self.assertTrue(contexts_diff < tolerance)

		# Verify that close to the expected number of first-entities got
		# replaced by random contexts
		entities_1_diff = abs(
			num_e1_differ - expected_num_entities_differ
		) / expected_num_entities_differ
		self.assertTrue(entities_1_diff < tolerance)

		# Verify that close to the expected number of first-entities got
		# replaced by random contexts
		entities_2_diff = abs(
			num_e2_differ - expected_num_entities_differ
		) / expected_num_entities_differ
		self.assertTrue(entities_2_diff < tolerance)


	def test_generate_entity_noise(self):
		'''
		Test that the noise generated from signal examples introduces
		random entities, in addition to random context tokens, when 
		the DatasetReader is configured to do so by constructor arg
		'''
		np.random.seed(1)
		tolerance = 0.1
		noise_ratio = 2000
		entity_noise_ratio = 0.5
		files = ['test-data/test-corpus/c.tsv']
		signal_examples = [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]
		reader = Relation2VecDatasetReader(
			files=files,
			noise_ratio=noise_ratio,
			entity_noise_ratio=entity_noise_ratio,
			verbose=False
		)

		# We need the reader to have dictionaries before we can generate
		# noise.
		reader.prepare()

		# Get the size of the vocabularies.  We are testing for the fact
		# that noise examples have random selections of entities and
		# contexts, and this lets us know how prevalent any given choice
		# should be
		entity_vocab_size = reader.entity_vocab_size()
		context_vocab_size = reader.context_vocab_size()

		# Generate noise examples.  Chunk them according to the signal
		# example that they were generated from
		noise_examples = reader.generate_noise_examples(signal_examples)
		chunked_by_signal_example = [
			noise_examples[i:i+noise_ratio] 
			for i in range(0,len(noise_examples),noise_ratio)
		]

		# The number of contexts that should differ is almost equal to the
		# number of contexts that get randomly selected
		expected_num_contexts_differ = (
			len(signal_examples) * noise_ratio * (1-entity_noise_ratio) 
		)
		# but is slightly smaller because the original context gets chosen
		# by chance sometimes
		expected_num_contexts_differ = (
			expected_num_contexts_differ 
			* (1 - 1 / float(context_vocab_size))
		)

		# Similarly for expected num of entities that differ, but 
		# the number of times entities differ is divided between the 
		# two possible choices for entity
		expected_num_entities_differ = (
			len(signal_examples) * noise_ratio * entity_noise_ratio
			* (1 - 1 / float(context_vocab_size)) * 0.5
		)

		# Go through all the generated noise examples.  See how much they
		# differ from signal examples from which they were generated.
		# Noise examples should only differ in one of either entity1, 
		# entity2, or context (verify that this is the case).  Keep
		# track of how often each of these has been substituted by
		# a random choice, and check that it is close to the expected num.
		num_contexts_differ = 0
		num_e1_differ = 0
		num_e2_differ = 0
		for i in range(len(signal_examples)):
			signal_example = signal_examples[i]
			noise_examples = chunked_by_signal_example[i]
			for noise_example in noise_examples:

				# Check if entity 1 was changed
				e1_diff = False
				if noise_example[0] != signal_example[0]:
					e1_diff = True
					num_e1_differ += 1

				# Check if entity 2 was changed
				e2_diff = False
				if noise_example[1] != signal_example[1]:
					e2_diff = True
					num_e2_differ += 1

				# Check if the context token was changed
				context_diff = False
				if noise_example[2] != signal_example[2]:
					context_diff = True
					num_contexts_differ += 1

				# Verify that only one of the entities or context is 
				# different from the signal example
				num_different = sum([e1_diff, e2_diff, context_diff])
				self.assertTrue(num_different < 2)

		# Verify that close to the expected number of contexts got 
		# replaced by random contexts
		contexts_diff = abs(
			num_contexts_differ - expected_num_contexts_differ
		) / expected_num_contexts_differ
		self.assertTrue(contexts_diff < tolerance)

		# Verify that close to the expected number of first-entities got
		# replaced by random contexts
		entities_1_diff = abs(
			num_e1_differ - expected_num_entities_differ
		) / expected_num_entities_differ
		self.assertTrue(entities_1_diff < tolerance)

		# Verify that close to the expected number of first-entities got
		# replaced by random contexts
		entities_2_diff = abs(
			num_e2_differ - expected_num_entities_differ
		) / expected_num_entities_differ
		self.assertTrue(entities_2_diff < tolerance)


	def test_generate_dataset_serial(self):
		'''
		Make sure that the correct entities and contexts are found
		together in batches
		'''

		# Ensure reproducibility in this stochastic test
		np.random.seed(1)

		# Make a DatasetReader
		files = ['test-data/test-corpus/a.tsv']
		batch_size = 5
		noise_ratio = 9
		reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size = 350,
			noise_ratio=noise_ratio,
			signal_sample_mode=FULL_CONTEXT,
			verbose=False
		)
		reader.prepare()

		# Read through the file and determine all of the entity pairs that
		# should be found, and the valid context tokens that go with them
		valid_contexts = defaultdict(set)
		filename='test-data/test-corpus/a.tsv'
		for line in reader.parse(filename):
			context_tokens, entity_spans = line

			context_ids = reader.context_dictionary.get_ids(
				context_tokens
			)

			for e1, e2 in itools.combinations(entity_spans, 2):

				# Give a strict order to make assertions easier
				e1_id, e2_id = reader.entity_dictionary.get_ids(
					[e1,e2])

				filtered_context_ids = reader.eliminate_spans(
					context_ids, entity_spans[e1]+entity_spans[e2]
				)

				# Adopt ordering convention of entities in entity pairs 
				# when using them as dictionary keys.  (Makes it easier 
				# to write assertions.)
				if e1_id > e2_id:
					e1, e2 = e2, e1
					e1_id, e2_id = e2_id, e1_id

				valid_contexts[(e1_id, e2_id)].update(filtered_context_ids)

		seen_contexts = defaultdict(set)
		macrobatches = reader.generate_dataset_serial()
		for signal_examples, noise_examples in macrobatches:
			self.assertEqual(
				len(noise_examples), len(signal_examples) * noise_ratio)
			for e1_id, e2_id, context_id in signal_examples:
				# Adopt ordering convention of entities in entity pairs 
				# when using them as dictionary keys.  (Makes it easier to 
				# write assertions.)
				if e1_id > e2_id:
					e1_id, e2_id = e2_id, e1_id

				seen_contexts[(e1_id, e2_id)].add(context_id)

		# elminiate entry in seen_contexts that is due to padding
		del seen_contexts[(UNK,UNK)]
		self.assertEqual(seen_contexts, valid_contexts)


	def test_generate_dataset_parallel(self):
		'''
		Relation2VecDatasetReader can produce minibatches 
		asynchronously (meaning that it generates future minibatches 
		before they are requested and stores them in a queue) or like an 
		ordinary generator as the consumer requests them.  Both methods 
		should give the same results.
		'''

		# Ensure reproducibility in this stochastic test
		np.random.seed(1)

		# Make a DatasetReader
		files = [
			'test-data/test-corpus/a1.tsv',
			'test-data/test-corpus/a2.tsv',
			'test-data/test-corpus/a3.tsv'
		]
		batch_size = 5
		noise_ratio = 9
		reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size = 350,
			noise_ratio=noise_ratio,
			signal_sample_mode = FULL_CONTEXT,
			verbose=False
		)
		reader.prepare()

		# Read through the file and determine all of the entity pairs that
		# should be found, and the valid context tokens that go with them
		valid_contexts = defaultdict(set)
		for fname in files:
			for line in reader.parse(fname):
				context_tokens, entity_spans = line

				context_ids = reader.context_dictionary.get_ids(
					context_tokens
				)

				for e1, e2 in itools.combinations(entity_spans, 2):

					# Give a strict order to make assertions easier
					e1_id, e2_id = reader.entity_dictionary.get_ids(
						[e1,e2])

					filtered_context_ids = reader.eliminate_spans(
						context_ids, entity_spans[e1]+entity_spans[e2]
					)

					# Adopt ordering convention of entities in entity pairs when using
					# them as dictionary keys.  (Makes it easier to write assertions.)
					if e1_id > e2_id:
						e1, e2 = e2, e1
						e1_id, e2_id = e2_id, e1_id

					valid_contexts[(e1_id, e2_id)].update(
						filtered_context_ids)

		seen_contexts = defaultdict(set)
		macrobatches = reader.generate_dataset_parallel()
		for signal_examples, noise_examples in macrobatches:
			self.assertEqual(
				len(noise_examples),
				len(signal_examples) * noise_ratio
			)
			for e1_id, e2_id, context_id in signal_examples:

				# Adopt ordering convention of entities in entity pairs 
				# when using them as dictionary keys.  (Makes it easier to 
				# write assertions.)
				if e1_id > e2_id:
					e1_id, e2_id = e2_id, e1_id

				seen_contexts[(e1_id, e2_id)].add(context_id)

		# Eliminate elements present due to padding
		del seen_contexts[(UNK,UNK)]

		self.assertEqual(seen_contexts, valid_contexts)



	def test_entity_span_skip(self):
		'''
		Tests the function that returns the sentence tokens after
		removing the spans belonging to entities.
		'''

		files = ['test-data/test-corpus/a.tsv']
		batch_size = 5
		noise_ratio = 9
		reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size = batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		reader.prepare()

		sentence = (
			"China and United States draw 1-1 BEIJING , Feb 1 ( AFP ) - "
			"China and the United States drew 1-1 in a soccer friendly at "
			"Guangzhou Saturday , the last of a two-match series , the "
			"Xinhua news agency reported ."
		).split()
		entity_spans = {
			'YAGO:United_States':[(15,19), (1,4)],
			'YAGO:Xinhua_News_Agency':[(38,38)],
			'YAGO:Beijing':[(7,7)],
			'YAGO:Guangzhou':[(27,27)]
		}
		expected_tokens = {
			('YAGO:Beijing', 'YAGO:United_States'): (
				"draw 1-1 , Feb 1 ( AFP ) - drew 1-1 in a soccer friendly "
				"at Guangzhou Saturday , the last of a two-match series , "
				"the Xinhua news agency reported ."
			).split(),

			('YAGO:Beijing', 'YAGO:Xinhua_News_Agency'): (
				"China and United States draw 1-1 , Feb 1 ( AFP ) - China "
				"and the United States drew 1-1 in a soccer friendly at "
				"Guangzhou Saturday , the last of a two-match series , the "
				"news agency reported ."
			).split(),

			('YAGO:Beijing', 'YAGO:Guangzhou'): (
				"China and United States draw 1-1 , Feb 1 ( AFP ) - China "
				"and the United States drew 1-1 in a soccer friendly at "
				"Saturday , the last of a two-match series , the Xinhua "
				"news agency reported ."
			).split(),

			('YAGO:United_States', 'YAGO:Xinhua_News_Agency'): (
				"draw 1-1 BEIJING , Feb 1 ( AFP ) - drew 1-1 in a soccer "
				"friendly at Guangzhou Saturday , the last of a two-match "
				"series , the news agency reported ."
			).split(),

			('YAGO:Guangzhou', 'YAGO:United_States'): (
				"draw 1-1 BEIJING , Feb 1 ( AFP ) - drew 1-1 in a soccer "
				"friendly at Saturday , the last of a two-match series , "
				"the Xinhua news agency reported ."
			).split(),

			('YAGO:Guangzhou', 'YAGO:Xinhua_News_Agency'): (
				"China and United States draw 1-1 BEIJING , Feb 1 ( AFP ) "
				"- China and the United States drew 1-1 in a soccer "
				"friendly at Saturday , the last of a two-match series , "
				"the news agency reported ."
			).split()
		}


		for e1, e2 in itools.combinations(entity_spans, 2):

			# put entities into definite ordering, which makes it easier
			# to make the right comparisons within assertions
			if e1 > e2:
				e1, e2 = e2, e1

			found_tokens_no_spans = reader.eliminate_spans(
				sentence, entity_spans[e1] + entity_spans[e2]
			)
			self.assertEqual(found_tokens_no_spans, expected_tokens[e1,e2])


	def test_save_load_dictionaries(self):
		'''
		Try saving and reloading a Relation2VecMinibatcher's dictionaries,
		and ensure they are unchanged by the saving and loading.
		'''
		save_dir = 'test-data/test-dataset-reader'
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)

		files = ['test-data/test-corpus/a.tsv']
		batch_size = 5
		noise_ratio = 15
		reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size = batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)

		reader.prepare(save_dir='test-data/test-minibatch-generator')

		new_reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size = batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		new_reader.load_dictionary('test-data/test-minibatch-generator')

		# check that the underlying data in the dictionaries and unigram
		# are the same
		self.assertEqual(
			reader.entity_dictionary.token_map.tokens,
			new_reader.entity_dictionary.token_map.tokens
		)
		self.assertEqual(
			reader.context_dictionary.token_map.tokens,
			new_reader.context_dictionary.token_map.tokens
		)
		self.assertEqual(
			reader.entity_dictionary.counter_sampler.counts,
			new_reader.entity_dictionary.counter_sampler.counts
		)
		self.assertEqual(
			reader.context_dictionary.counter_sampler.counts,
			new_reader.context_dictionary.counter_sampler.counts
		)

		# Now we'll try using the manual call to save
		shutil.rmtree('test-data/test-minibatch-generator')

		# These functions don't automatically make the parent directory
		# if it doesn't exist, so we need to make it
		reader.save_dictionary('test-data/test-minibatch-generator')
		new_reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size=batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)

		new_reader.load_dictionary('test-data/test-minibatch-generator')

		# check that the underlying data in the dictionaries and unigram
		# are the same
		self.assertEqual(
			reader.entity_dictionary.token_map.tokens,
			new_reader.entity_dictionary.token_map.tokens
		)
		self.assertEqual(
			reader.context_dictionary.token_map.tokens,
			new_reader.context_dictionary.token_map.tokens
		)
		self.assertEqual(
			reader.entity_dictionary.counter_sampler.counts,
			new_reader.entity_dictionary.counter_sampler.counts
		)
		self.assertEqual(
			reader.context_dictionary.counter_sampler.counts,
			new_reader.context_dictionary.counter_sampler.counts
		)


	# TODO: this test deactivated for now because the saving and loading
	# of pre-compiled datasets wasn't migrated to the new macrobatching
	# approach to dataset iteration.  Should we enable saving and loading
	# again?
	def _test_save_load_examples_serial(self):
		'''
		Try saving and reloading a Relation2VecDatasetReader
		'''
		save_dir = 'test-data/test-dataset-reader'
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)

		files = ['test-data/test-corpus/a.tsv']
		batch_size = 5
		noise_ratio = 15
		reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size = batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		reader.prepare()
		reader.generate_dataset_serial(save_dir)

		new_reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size = batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		new_reader.load_data(save_dir)

		# check that the underlying data in the dictionaries and unigram
		# are the same
		self.assertTrue(np.array_equal(new_reader.signal_examples, reader.signal_examples))
		self.assertTrue(np.array_equal(new_reader.noise_examples, reader.noise_examples))

		# Next, test manually calling the save function.  Clear the data from disk.
		shutil.rmtree(save_dir)

		# Call the save function on the existing reader
		reader.save_data(save_dir)

		# Make a fresh reader in which we will load the data just saved.
		new_reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size=batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		new_reader.load_data(save_dir)

		# check that the underlying data in the dictionaries and unigram
		# are the same
		self.assertTrue(np.array_equal(new_reader.signal_examples, reader.signal_examples))
		self.assertTrue(np.array_equal(new_reader.noise_examples, reader.noise_examples))

		# Now we'll try using the manual call to save
		shutil.rmtree(save_dir)


	def _test_save_load_examples_parallel(self):
		'''
		Try saving and reloading a Relation2VecDatasetReader
		'''
		save_dir = 'test-data/test-dataset-reader'
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)

		files = ['test-data/test-corpus/a.tsv']
		batch_size = 5
		noise_ratio = 15
		reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size = batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		reader.prepare()
		reader.generate_dataset_parallel(save_dir)

		new_reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size = batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		new_reader.load_data(save_dir)

		# check that the underlying data in the dictionaries and unigram
		# are the same
		self.assertTrue(np.array_equal(new_reader.signal_examples, reader.signal_examples))
		self.assertTrue(np.array_equal(new_reader.noise_examples, reader.noise_examples))

		# Next, test manually calling the save function.  Clear the data from disk.
		shutil.rmtree(save_dir)

		# Call the save function on the existing reader
		reader.save_data(save_dir)

		# Make a fresh reader in which we will load the data just saved.
		new_reader = Relation2VecDatasetReader(
			files=files,
			macrobatch_size=batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		new_reader.load_data(save_dir)

		# check that the underlying data in the dictionaries and unigram
		# are the same
		self.assertTrue(np.array_equal(new_reader.signal_examples, reader.signal_examples))
		self.assertTrue(np.array_equal(new_reader.noise_examples, reader.noise_examples))

		# Now we'll try using the manual call to save
		shutil.rmtree(save_dir)






if __name__ == '__main__':
	main()
