from theano import tensor as T, function
import numpy as np
import os
import shutil
from collections import defaultdict, Counter
from relation2vec_embedder import Relation2VecEmbedder
from word2vec import NoiseContraster, Word2VecEmbedder
import itertools as itools
import unittest
from minibatcher import (
	Relation2VecMinibatcher, relation2vec_parse, word2vec_parse,
	Word2VecMinibatcher
)
from unittest import TestCase, main

@np.vectorize
def sigma(a):
	return 1 / (1 + np.exp(-a))


# TODO: make this a real test
class TestParse(TestCase):

	def test_parse(self):
		filename = 'test-data/test-corpus/004-raw.tsv'


class TestWord2VecOnCorpus(TestCase):
	'''
	This tests the Word2Vec end-to-end functionality applied to a text 
	corpus.
	'''

	def test_word2vec_on_corpus_(self):

		files=['test-data/test-corpus/numbers-long-raw.txt']
		directories=[]
		verbose = False
		batch_size = 10
		t=1
		verbose=False
		num_epochs=1
		num_embedding_dimensions=5

		# Make a Minibatcher
		minibatcher = Word2VecMinibatcher(
			files=files, t=t,
			batch_size=batch_size,
			verbose=False
		)

		# Prpare the minibatch generator 
		# (this produces the counter_sampler stats)
		minibatcher.prepare()

		# Define the input theano variables
		signal_input = T.imatrix('query_input')
		noise_input = T.imatrix('noise_input')

		# Make a NoiseContraster, and get the combined input
		noise_contraster = NoiseContraster(signal_input, noise_input)
		combined_input = noise_contraster.get_combined_input()

		# Make a Word2VecEmbedder object, feed it the combined input
		word2vec_embedder = Word2VecEmbedder(
			input_var=combined_input,
			batch_size=batch_size,
			vocabulary_size=minibatcher.get_vocab_size(),
			num_embedding_dimensions=num_embedding_dimensions
		)

		# Get the params and output from the word2vec embedder, feed that
		# back to the noise_contraster to get the training function
		combined_output = word2vec_embedder.get_output()
		params = word2vec_embedder.get_params()
		train = noise_contraster.get_train_func(combined_output, params)

		# Iterate over the corpus, training the embeddings
		for epoch in range(num_epochs):
			# print 'starting epoch %d' % epoch
			for signal_batch, noise_batch in minibatcher:
				loss = train(signal_batch, noise_batch)

		W, C = word2vec_embedder.get_param_values()
		dots = sigma(np.dot(W,C.T))

		# Based on the construction of the corpus, the following 
		# context embeddings should match the query at right and be 
		# the highest value in the product of the embedding matrices
		# Note that token 0 is reserved for UNK.  It's embedding stays
		# near the randomly initialized value, tending to yield of 0.5
		# which is high overall, so it turns up as a "good match" to any
		# word
		expected_tops = [
			[0,2,3], # these contexts are good match to query 1
			[0,1,3], # these contexts are good match to query 2 
			[0,1,2], # these contexts are good match to query 3 
			[0,5,6], # these contexts are good match to query 4 
			[0,4,6], # these contexts are good match to query 5 
			[0,4,5], # these contexts are good match to query 6 
			[0,8,9], # these contexts are good match to query 7 
			[0,7,9], # these contexts are good match to query 8 
			[0,7,8], # these contexts are good match to query 9 
			[0,11,12], # these contexts are good match to query 10
			[0,10,12], # these contexts are good match to query 11 
			[0,10,11]  # these contexts are good match to query 12 
		]

		for i in range(1, 3*4+1):
			top3 = sorted(
				enumerate(dots[i]), key=lambda x: x[1], reverse=True
			)[:3]
			top3_positions = [t[0] for t in top3]
			self.assertItemsEqual(top3_positions, expected_tops[i-1])


class TestWord2VecMinibatcher(TestCase):

	def setUp(self):

		# Define some parameters to be used in construction 
		# Minibatcher
		self.files = [
			'test-data/test-corpus/003-raw.tsv',
			'test-data/test-corpus/004-raw.tsv'
		]
		self.batch_size = 5
		self.noise_ratio = 15
		self.t = 0.03

		# Make a minibatch generator
		self.generator = Word2VecMinibatcher(
			files=self.files,
			t=self.t,
			batch_size=self.batch_size,
			noise_ratio=self.noise_ratio,
			verbose=False
		)

		# Make another Word2VecMinibatcher, and pre-load this one with 
		# token_map and counter_sampler distribution information.
		self.preloaded_generator = Word2VecMinibatcher(
			files=self.files,
			t=self.t,
			batch_size=self.batch_size,
			noise_ratio=self.noise_ratio,
			verbose=False
		)
		self.preloaded_generator.load('test-data/word2vec-minibatcher-test')


	def test_prepare(self):
		'''
		Check that Word2VecMinibatcher.prepare() properly makes a 
		UnigramDictionary that reflects the corpus.
		'''
		self.generator.prepare()
		d = self.generator.unigram_dictionary

		# Make sure that all of the expected tokens are found in the 
		# unigram_dictionary, and that their frequency in the is correct.
		tokens = []
		for filename in self.files:
			for add_tokens in word2vec_parse(filename):
				tokens.extend(add_tokens)

		counts = Counter(tokens)
		for token in tokens:
			token_id = d.get_id(token)
			count = d.get_frequency(token_id)
			self.assertEqual(count, counts[token])


	def test_minibatches(self):
		'''
		Make sure that the minibatches are the correct size, that 
		signal query- and contexts-words are always within 5 tokens of
		one another and come from the same sentence.
		'''
		# Ensure reproducibility in this stochastic test
		np.random.seed(1)

		# Before looking at the minibatches, we need to determine what 
		# query-context pairs are possible.
		# To do that, first read in the corpus, and break it into lines
		# and tokens
		lines = []

		# Go through the corpus and get all the token ids as a list
		tokenized_lines = []
		for filename in self.files:
			for tokens in word2vec_parse(filename):
				tokenized_lines.append(tokens)

		# Now iterate through the lines, noting what tokens arise within
		# one another's contexts.  Build a lookup table providing the set
		# of token_ids that arose in the context of each given token_id
		legal_pairs = defaultdict(set)
		d = self.preloaded_generator.unigram_dictionary
		for line in tokenized_lines:
			token_ids = d.get_ids(line)
			for i, token_id in enumerate(token_ids):
				low = max(0, i-5)
				legal_context = token_ids[low:i] + token_ids[i+1:i+6]
				legal_pairs[token_id].update(legal_context)

		# finally, add UNK to the legal pairs
		legal_pairs[0] = set([0])

		for signal_batch, noise_batch in self.preloaded_generator:

			self.assertEqual(len(signal_batch), self.batch_size)
			self.assertEqual(
				len(noise_batch), self.batch_size * self.noise_ratio
			)

			# Ensure that all of the signal examples are actually valid
			# samples from the corpus
			for query_token_id, context_token_id in signal_batch:
				self.assertTrue(
					context_token_id in legal_pairs[query_token_id]
				)

	def test_token_discarding(self):

		# Ensure reproducibility for the test
		np.random.seed(1)

		# Get the preloaded generator and its unigram_dictionary
		self.preloaded_generator
		d = self.preloaded_generator.unigram_dictionary

		# Go through the corpus and get all the token ids as a list
		token_ids = []
		for filename in self.files:
			for tokens in word2vec_parse(filename):
				token_ids.extend(d.get_ids(tokens))

		# Run through the tokens, evaluating 
		# Word2VecMinibatcher.do_discard() on each.  Keep track of all 
		# "discarded" tokens for which do_discard() returns True
		discarded = []
		num_replicates = 100
		for replicates in range(num_replicates):
			for token_id in token_ids:
				if self.preloaded_generator.do_discard(token_id):
					discarded.append(token_id)

		# Count the tokens, and the discarded tokens.  
		discarded_counts = Counter(discarded)
		token_counts = Counter(token_ids)

		# Compute the frequency of the word "the", and the frequency
		# with which it was discarded
		the_id = d.get_id('the')
		num_the_appears = token_counts[the_id]
		the_frequency = num_the_appears/float(len(token_ids))
		num_the_discarded = discarded_counts[the_id]
		frequency_of_discard = (
			num_the_discarded / float(num_the_appears * num_replicates)
		)

		# What was actually the most discarded token?  It should be "the"
		most_discarded_id, num_most_discarded = (
			discarded_counts.most_common()[0]
		)
		self.assertEqual(most_discarded_id, the_id)

		# What was the expected frequency with which "the" would be 
		# discarded?  Assert it is close to the actual discard rate.
		expected_frequency = 1 - np.sqrt(self.t / the_frequency)
		tolerance = 0.015
		self.assertTrue(
			abs(expected_frequency - frequency_of_discard) < tolerance
		)




class TestRelation2VecEmbedder(TestCase):

	def test_save_load(self):

		# Make sure that the target directory exists, but delete any model
		# files left over from a previous test run.
		if not os.path.exists('test-data/test-embedder'):
			os.makedirs('test-data/test-embedder')
		if os.path.exists('test-data/test-embedder/embeddings.npz'):
			os.remove('test-data/test-embedder/embeddings.npz')

		embedder = Relation2VecEmbedder(
			entity_vocab_size=10,
			context_vocab_size=50,
			num_embedding_dimensions=5
		)
		expected_params = embedder.get_param_values()
		embedder.save('test-data/test-embedder/embeddings.npz')

		new_embedder = Relation2VecEmbedder(
			entity_vocab_size=10,
			context_vocab_size=50,
			num_embedding_dimensions=5
		)
		new_embedder.load('test-data/test-embedder/embeddings.npz')
		found_params = new_embedder.get_param_values()

		for found, expected in zip(found_params, expected_params):
			self.assertTrue(np.array_equal(found, expected))

		if os.path.exists('test-data/test-embedder/embeddings.npz'):
			os.remove('test-data/test-embedder/embeddings.npz')


	def test_learning(self):

		# Seed randomness for a reproducible test.  Using 2 because
		# 1 was particularly bad by chance
		np.random.seed(2)

		# Some constants for the test
		files = ['test-data/test-corpus/c.tsv']
		batch_size = 50
		noise_ratio = 15
		num_embedding_dimensions = 5
		num_epochs = 5
		num_replicates=5
		learning_rate = 0.01
		tolerance = 0.025
		savedir = 'test-data/test-entity-embedder'

		# Make a minibatcher to yield training batches from test corpus
		minibatcher = Relation2VecMinibatcher(
			files=files,
			batch_size=batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		minibatcher.prepare(savedir=savedir)

		print 'finished making minibatcher'

		# Make signal and noise channels and prepared the noise contraster
		signal_input = T.imatrix('signal')
		noise_input = T.imatrix('noise')
		noise_contraster = NoiseContraster(
			signal_input, noise_input, learning_rate=learning_rate
		)
		combined_input = noise_contraster.get_combined_input()
		
		print 'made noise contraster'

		# We'll make and train an Relation2VecEmbedder in a moment.  However,
		# first we will get the IDs for the entities that occur together
		# within the test corpus.  We'll be interested to see the 
		# relationship embedding for them that is learnt during training
		edict = minibatcher.entity_dictionary
		expected_pairs = [
			('A','B'), ('C','D'), ('E','F')
		]
		expected_pairs_ids = [
			(edict.get_id(e1), edict.get_id(e2)) 
			for e1, e2 in expected_pairs
		]

		print 'about to start main loop'

		# We will repeatedly make an Relation2VecEmbedder, train it on the 
		# test corpus, and then find its embedding for the entity-pairs
		# of interest.  We do it num_replicate # of times to average
		# results over enough trials that we can test if the learnt
		# embeddings have roughly the expected properties
		embedding_dot_products  = []
		for replicate in range(num_replicates):

			print 'making r2v embedder'

			# Make an Relation2VecEmbedder
			entity_embedder = Relation2VecEmbedder(
				combined_input,
				batch_size,
				minibatcher.entity_vocab_size(),
				minibatcher.context_vocab_size(),
				num_embedding_dimensions
			)

			print 'made r2v embedder'

			# Get its output and make a theano training function
			output = entity_embedder.get_output()
			params = entity_embedder.get_params()
			train = noise_contraster.get_train_func(output, params)

			print 'made training func'

			# Train on the dataset, running through it num_epochs # of times
			for epoch in range(num_epochs):
				#print 'Epoch %d' % epoch

				for signal_batch, noise_batch in minibatcher:
					loss = train(signal_batch, noise_batch)
					#print loss

			print 'Done training in one pass of main loop'

			# Get the parameters out of the trained model
			W_entity, W_context, W_relation, b_relation = (
				entity_embedder.get_param_values()
			)

			# Get the embeddings for the entity-pairs ("relationships") 
			# occuring in the test corpus
			embedded_relationships = entity_embedder.embed_relationship(
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
			#print embedding_dot_product

			# Accumulate fit scores over the replicates
			embedding_dot_products.append(embedding_dot_product)

		# Average the fit-scores over the replicates
		avg_embedding_product = np.mean(embedding_dot_products, axis=0)
		#print avg_embedding_product

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



class TestRelation2VecMinibatcher(TestCase):

	def setUp(self):
		'''
		Prepare a Relation2VecMinibatcher so it is available in tests.
		'''
		self.files = ['test-data/test-corpus/a.tsv']
		self.batch_size = 5
		self.noise_ratio = 15
		self.mini_gen = Relation2VecMinibatcher(
			files=self.files,
			batch_size=self.batch_size, 
			noise_ratio=self.noise_ratio,
			verbose=False
		)
		self.mini_gen.prepare()


	def tearDown(self):
		# Remove test files if they exist
		if os.path.exists('test-data/test-minibatch-generator'):
			shutil.rmtree('test-data/test-minibatch-generator')


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

		# Get the symbolic minibatch
		batch_spec = self.mini_gen.get_symbolic_minibatch()
		symbolic_signal, symbolic_noise, updates = batch_spec

		# Make a batching function.  This is a somewhat trivial function:
		# all it does is pull out the minibatch.  But it simulates 
		# accessing the minibatch in a theano compiled function which 
		# incorporates incrementing the minibatch iteration number as an
		# update
		get_batch = function(
			[],[symbolic_signal, symbolic_noise],
			updates=updates
		)

		#
		num_batches = self.mini_gen.load_dataset()

		# Get every signal and noise example, and keep track of them
		# all.  We'll check that we get the exact same examples from
		# the symbolic and non-symbolic methods of minibatching
		symbolic_signal_counter = Counter()
		symbolic_noise_counter = Counter()
		for batch_num in range(num_batches):
			signal, noise = get_batch()
			for row in signal:
				symbolic_signal_counter[tuple(row)] += 1

			for row in noise:
				symbolic_noise_counter[tuple(row)] += 1

		# We will now generate the minibatches, but serve them 
		# non-symbolically.  Ensure the same randomness seeds this process
		# so that we can expect the same examples to be chosen.
		np.random.seed(1)

		# Generate minibatches using the minibatcher's non-symbolic
		# asynchronous minibatching approach
		signal_counter = Counter()
		noise_counter = Counter()
		for signal_batch, noise_batch in self.mini_gen:
			for row in signal_batch:
				signal_counter[tuple(row)] += 1
			for row in noise_batch:
				noise_counter[tuple(row)] += 1

		# now check that we have all of the same examples
		self.assertEqual(symbolic_signal_counter, signal_counter)
		self.assertEqual(symbolic_noise_counter, noise_counter)


	def test_save_load(self):
		'''
		Try saving and reloading a Relation2VecMinibatcher
		'''
		if os.path.exists('test-data/test-minibatch-generator'):
			shutil.rmtree('test-data/test-minibatch-generator')
		
		files = ['test-data/test-corpus/a.tsv']
		batch_size = 5
		noise_ratio = 15
		mini_gen = Relation2VecMinibatcher(
			files=self.files,
			batch_size=self.batch_size, 
			noise_ratio=self.noise_ratio,
			verbose=False
		)
		mini_gen.prepare(savedir='test-data/test-minibatch-generator')

		new_mini_gen = Relation2VecMinibatcher(
			files=self.files,
			batch_size=self.batch_size, 
			noise_ratio=self.noise_ratio,
			verbose=False
		)
		new_mini_gen.load('test-data/test-minibatch-generator')

		# check that the underlying data in the dictionaries and unigram
		# are the same
		self.assertEqual(
			mini_gen.entity_dictionary.token_map.tokens,
			new_mini_gen.entity_dictionary.token_map.tokens
		)
		self.assertEqual(
			mini_gen.context_dictionary.token_map.tokens,
			new_mini_gen.context_dictionary.token_map.tokens
		)
		self.assertEqual(
			mini_gen.entity_dictionary.counter_sampler.counts,
			new_mini_gen.entity_dictionary.counter_sampler.counts
		)
		self.assertEqual(
			mini_gen.context_dictionary.counter_sampler.counts,
			new_mini_gen.context_dictionary.counter_sampler.counts
		)

		# Now we'll try using the component save and load functions
		# First remove the saved files
		shutil.rmtree('test-data/test-minibatch-generator')

		# These functions don't automatically make the parent directory
		# if it doesn't exist, so we need to make it
		os.mkdir('test-data/test-minibatch-generator')
		mini_gen.save_entity_dictionary(
			'test-data/test-minibatch-generator/entity-dictionary'
		)
		mini_gen.save_context_dictionary(
			'test-data/test-minibatch-generator/context-dictionary'
		)

		new_new_mini_gen = Relation2VecMinibatcher(
			files=self.files,
			batch_size=self.batch_size, 
			noise_ratio=self.noise_ratio,
			verbose=False
		)

		new_new_mini_gen.load_entity_dictionary(
			'test-data/test-minibatch-generator/entity-dictionary'
		)
		new_new_mini_gen.load_context_dictionary(
			'test-data/test-minibatch-generator/context-dictionary'
		)

		# check that the underlying data in the dictionaries and unigram
		# are the same
		self.assertEqual(
			mini_gen.entity_dictionary.token_map.tokens,
			new_new_mini_gen.entity_dictionary.token_map.tokens
		)
		self.assertEqual(
			mini_gen.context_dictionary.token_map.tokens,
			new_new_mini_gen.context_dictionary.token_map.tokens
		)
		self.assertEqual(
			mini_gen.entity_dictionary.counter_sampler.counts,
			new_new_mini_gen.entity_dictionary.counter_sampler.counts
		)
		self.assertEqual(
			mini_gen.context_dictionary.counter_sampler.counts,
			new_new_mini_gen.context_dictionary.counter_sampler.counts
		)


	def test_generate_async_batch(self):
		'''
		Relation2VecMinibatcher can produce minibatches asynchronously (meaning
		that it generates future minibatches before they are requested and
		stores them in a queue) or like an ordinary generator as the consumer 
		requests them.  Both methods should give the same results.
		'''

		generator_batches = []
		async_batches = []

		# We will generate the minibatches using three alternative methods
		# provided by the minibatch generator.  Provided we seed numpy's 
		# randomness before generating, each should yield exactly the
		# same minibatches.

		# Collect the minibatches by treating the Relation2VecMinibatcher
		# as an iterator.  This leads to "asynchronous" multiprocessing
		# batch generation
		np.random.seed(1)
		for minibatch in self.mini_gen:
			async_batches.append(minibatch)

		# Collect the minibatches by calling 
		# `Relation2VecMinibatcher.generate()`, which produces minibatches
		# using the "generator" construct, without any multiprocessing
		np.random.seed(1)
		for minibatch in self.mini_gen.generate_minibatches():
			generator_batches.append(minibatch)

		# Collect the minibatches by calling 
		# `Relation2VecMinibatcher.get_minibatches()`, which returns a simple
		# list of all minibatches, all generated "upfront"
		np.random.seed(1)
		upfront_batches = self.mini_gen.get_minibatches()

		# Test that async and generate are the same
		zipped_batches = zip(generator_batches, async_batches)
		for (gen_batch, async_batch) in zipped_batches:
			async_sig, async_noise = async_batch
			gen_sig, gen_noise = gen_batch
			self.assertTrue(np.equal(async_sig, gen_sig).all())
			self.assertTrue(np.equal(async_noise, gen_noise).all())

		# Test that generate and upfront are the same
		zipped_batches = zip(generator_batches, upfront_batches)
		for (gen_batch, upfront_batch) in zipped_batches:
			upfront_sig, upfront_noise = upfront_batch
			gen_sig, gen_noise = gen_batch
			self.assertTrue(np.equal(upfront_sig, gen_sig).all())
			self.assertTrue(np.equal(upfront_noise, gen_noise).all())


	def test_batch_contents(self):
		'''
		Make sure that the correct entities and contexts are found
		together in batches
		'''
		valid_tokens = defaultdict(set)
		filename='test-data/test-corpus/a.tsv'

		for line in relation2vec_parse(filename):
			context_tokens, entity_spans = line

			context_ids = self.mini_gen.context_dictionary.get_ids(
				context_tokens
			)

			for e1, e2 in itools.combinations(entity_spans, 2):

				# Give a strict order to make assertions easier
				e1_id, e2_id = self.mini_gen.entity_dictionary.get_ids(
					[e1,e2])

				filtered_context_ids = self.mini_gen.eliminate_spans(
					context_ids, entity_spans[e1]+entity_spans[e2]
				)

				if e1_id > e2_id:
					e1, e2 = e2, e1
					e1_id, e2_id = e2_id, e1_id

				valid_tokens[(e1_id, e2_id)].update(filtered_context_ids)

		seen_entity_pairs = []
 		for signal_batch, noise_batch in self.mini_gen:
			for e1_id, e2_id, context_id in signal_batch:

				if e1_id > e2_id:
					e1_id, e2_id = e2_id, e1_id

				self.assertTrue(context_id in valid_tokens[e1_id, e2_id])
				seen_entity_pairs.append((e1_id, e2_id))

		self.assertItemsEqual(seen_entity_pairs, valid_tokens.keys())
		

	def test_batch_shape(self):
		count_batches = 0
		for signal_batch, noise_batch in self.mini_gen:
			self.assertEqual(len(signal_batch), self.batch_size)
			self.assertEqual(
				len(noise_batch), self.batch_size * self.noise_ratio
			)
			count_batches += 1

		expected_number_of_batches = (4*3/2 + 3*2/2 + 2*1/2 + 0 + 0) / 5

		self.assertEqual(count_batches, expected_number_of_batches)


	def test_entity_span_skip(self):
		'''
		Tests the function that returns the sentence tokens after
		removing the spans belonging to entities.
		'''
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

			found_tokens_no_spans = self.mini_gen.eliminate_spans(
				sentence, entity_spans[e1] + entity_spans[e2]
			)
			self.assertEqual(found_tokens_no_spans, expected_tokens[e1,e2])


		


if __name__ == '__main__':
	main()
