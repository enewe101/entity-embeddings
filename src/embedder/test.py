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
from r2v import relation2vec
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


#class TestWord2VecOnCorpus(TestCase):
#	'''
#	This tests the Word2Vec end-to-end functionality applied to a text
#	corpus.
#	'''
#
#	def test_word2vec_on_corpus_(self):
#
#		files=['test-data/test-corpus/numbers-long-raw.txt']
#		directories=[]
#		verbose = False
#		batch_size = 10
#		t=1
#		verbose=False
#		num_epochs=1
#		num_embedding_dimensions=5
#
#		# Make a Minibatcher
#		minibatcher = Word2VecMinibatcher(
#			files=files, t=t,
#			batch_size=batch_size,
#			verbose=False
#		)
#
#		# Prpare the minibatch generator
#		# (this produces the counter_sampler stats)
#		minibatcher.prepare()
#
#		# Define the input theano variables
#		signal_input = T.imatrix('query_input')
#		noise_input = T.imatrix('noise_input')
#
#		# Make a NoiseContraster, and get the combined input
#		noise_contraster = NoiseContraster(signal_input, noise_input)
#		combined_input = noise_contraster.get_combined_input()
#
#		# Make a Word2VecEmbedder object, feed it the combined input
#		word2vec_embedder = Word2VecEmbedder(
#			input_var=combined_input,
#			batch_size=batch_size,
#			vocabulary_size=minibatcher.get_vocab_size(),
#			num_embedding_dimensions=num_embedding_dimensions
#		)
#
#		# Get the params and output from the word2vec embedder, feed that
#		# back to the noise_contraster to get the training function
#		combined_output = word2vec_embedder.get_output()
#		params = word2vec_embedder.get_params()
#		train = noise_contraster.get_train_func(combined_output, params)
#
#		# Iterate over the corpus, training the embeddings
#		for epoch in range(num_epochs):
#			for signal_batch, noise_batch in minibatcher:
#				loss = train(signal_batch, noise_batch)
#
#		W, C = word2vec_embedder.get_param_values()
#		dots = sigma(np.dot(W,C.T))
#
#		# Based on the construction of the corpus, the following
#		# context embeddings should match the query at right and be
#		# the highest value in the product of the embedding matrices
#		# Note that token 0 is reserved for UNK.  It's embedding stays
#		# near the randomly initialized value, tending to yield of 0.5
#		# which is high overall, so it turns up as a "good match" to any
#		# word
#		expected_tops = [
#			[0,2,3], # these contexts are good match to query 1
#			[0,1,3], # these contexts are good match to query 2
#			[0,1,2], # these contexts are good match to query 3
#			[0,5,6], # these contexts are good match to query 4
#			[0,4,6], # these contexts are good match to query 5
#			[0,4,5], # these contexts are good match to query 6
#			[0,8,9], # these contexts are good match to query 7
#			[0,7,9], # these contexts are good match to query 8
#			[0,7,8], # these contexts are good match to query 9
#			[0,11,12], # these contexts are good match to query 10
#			[0,10,12], # these contexts are good match to query 11
#			[0,10,11]  # these contexts are good match to query 12
#		]
#
#		for i in range(1, 3*4+1):
#			top3 = sorted(
#				enumerate(dots[i]), key=lambda x: x[1], reverse=True
#			)[:3]
#			top3_positions = [t[0] for t in top3]
#			self.assertItemsEqual(top3_positions, expected_tops[i-1])


#class TestWord2VecMinibatcher(TestCase):
#
#	def setUp(self):
#
#		# Define some parameters to be used in construction
#		# Minibatcher
#		self.files = [
#			'test-data/test-corpus/003-raw.tsv',
#			'test-data/test-corpus/004-raw.tsv'
#		]
#		self.batch_size = 5
#		self.noise_ratio = 15
#		self.t = 0.03
#
#		# Make a minibatch generator
#		self.generator = Word2VecMinibatcher(
#			files=self.files,
#			t=self.t,
#			batch_size=self.batch_size,
#			noise_ratio=self.noise_ratio,
#			verbose=False
#		)
#
#		# Make another Word2VecMinibatcher, and pre-load this one with
#		# token_map and counter_sampler distribution information.
#		self.preloaded_generator = Word2VecMinibatcher(
#			files=self.files,
#			t=self.t,
#			batch_size=self.batch_size,
#			noise_ratio=self.noise_ratio,
#			verbose=False
#		)
#		self.preloaded_generator.load('test-data/word2vec-minibatcher-test')
#
#
#	def test_prepare(self):
#		'''
#		Check that Word2VecMinibatcher.prepare() properly makes a
#		UnigramDictionary that reflects the corpus.
#		'''
#		self.generator.prepare()
#		d = self.generator.unigram_dictionary
#
#		# Make sure that all of the expected tokens are found in the
#		# unigram_dictionary, and that their frequency in the is correct.
#		tokens = []
#		for filename in self.files:
#			for add_tokens in word2vec_parse(filename):
#				tokens.extend(add_tokens)
#
#		counts = Counter(tokens)
#		for token in tokens:
#			token_id = d.get_id(token)
#			count = d.get_frequency(token_id)
#			self.assertEqual(count, counts[token])
#
#
#	def test_minibatches(self):
#		'''
#		Make sure that the minibatches are the correct size, that
#		signal query- and contexts-words are always within 5 tokens of
#		one another and come from the same sentence.
#		'''
#		# Ensure reproducibility in this stochastic test
#		np.random.seed(1)
#
#		# Before looking at the minibatches, we need to determine what
#		# query-context pairs are possible.
#		# To do that, first read in the corpus, and break it into lines
#		# and tokens
#		lines = []
#
#		# Go through the corpus and get all the token ids as a list
#		tokenized_lines = []
#		for filename in self.files:
#			for tokens in word2vec_parse(filename):
#				tokenized_lines.append(tokens)
#
#		# Now iterate through the lines, noting what tokens arise within
#		# one another's contexts.  Build a lookup table providing the set
#		# of token_ids that arose in the context of each given token_id
#		legal_pairs = defaultdict(set)
#		d = self.preloaded_generator.unigram_dictionary
#		for line in tokenized_lines:
#			token_ids = d.get_ids(line)
#			for i, token_id in enumerate(token_ids):
#				low = max(0, i-5)
#				legal_context = token_ids[low:i] + token_ids[i+1:i+6]
#				legal_pairs[token_id].update(legal_context)
#
#		# finally, add UNK to the legal pairs
#		legal_pairs[0] = set([0])
#
#		for signal_batch, noise_batch in self.preloaded_generator:
#
#			self.assertEqual(len(signal_batch), self.batch_size)
#			self.assertEqual(
#				len(noise_batch), self.batch_size * self.noise_ratio
#			)
#
#			# Ensure that all of the signal examples are actually valid
#			# samples from the corpus
#			for query_token_id, context_token_id in signal_batch:
#				self.assertTrue(
#					context_token_id in legal_pairs[query_token_id]
#				)
#
#
#	def test_token_discarding(self):
#
#		# Ensure reproducibility for the test
#		np.random.seed(1)
#
#		# Get the preloaded generator and its unigram_dictionary
#		self.preloaded_generator
#		d = self.preloaded_generator.unigram_dictionary
#
#		# Go through the corpus and get all the token ids as a list
#		token_ids = []
#		for filename in self.files:
#			for tokens in word2vec_parse(filename):
#				token_ids.extend(d.get_ids(tokens))
#
#		# Run through the tokens, evaluating
#		# Word2VecMinibatcher.do_discard() on each.  Keep track of all
#		# "discarded" tokens for which do_discard() returns True
#		discarded = []
#		num_replicates = 100
#		for replicates in range(num_replicates):
#			for token_id in token_ids:
#				if self.preloaded_generator.do_discard(token_id):
#					discarded.append(token_id)
#
#		# Count the tokens, and the discarded tokens.
#		discarded_counts = Counter(discarded)
#		token_counts = Counter(token_ids)
#
#		# Compute the frequency of the word "the", and the frequency
#		# with which it was discarded
#		the_id = d.get_id('the')
#		num_the_appears = token_counts[the_id]
#		the_frequency = num_the_appears/float(len(token_ids))
#		num_the_discarded = discarded_counts[the_id]
#		frequency_of_discard = (
#			num_the_discarded / float(num_the_appears * num_replicates)
#		)
#
#		# What was actually the most discarded token?  It should be "the"
#		most_discarded_id, num_most_discarded = (
#			discarded_counts.most_common()[0]
#		)
#		self.assertEqual(most_discarded_id, the_id)
#
#		# What was the expected frequency with which "the" would be
#		# discarded?  Assert it is close to the actual discard rate.
#		expected_frequency = 1 - np.sqrt(self.t / the_frequency)
#		tolerance = 0.015
#		self.assertTrue(
#			abs(expected_frequency - frequency_of_discard) < tolerance
#		)


class TestRelation2VecEmbedder(TestCase):

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

	def test_raises_not_implemented(self):
		'''
		Ensure that the reader raises an error if `get_vocab_size()` is called.'
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

		# Ensure that the reader raises an error if `get_vocab_size() is called`
		with self.assertRaises(NotImplementedError):
			reader.get_vocab_size()


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
