from theano import tensor as T, function
import numpy as np
import os
import shutil
from collections import defaultdict
from entity_embedder import EntityEmbedder
from word2vec import CorpusReader, NoiseContraster
import itertools as itools
import unittest
from minibatch_generator import MinibatchGenerator, parse
from unittest import TestCase, main

@np.vectorize
def sigma(a):
	return 1 / (1 + np.exp(-a))

# TODO: make this test.  It should know what entity pairs and contexts
# have a high probability of arising with the data, following NCE form
class TestEntityEmbedder(TestCase):

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
		minibatch_generator = MinibatchGenerator(
			files=files,
			batch_size=batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		minibatch_generator.prepare(savedir=savedir)

		# Make signal and noise channels and prepared the noise contraster
		signal_input = T.imatrix('signal')
		noise_input = T.imatrix('noise')
		noise_contraster = NoiseContraster(
			signal_input, noise_input, learning_rate=learning_rate
		)
		combined_input = noise_contraster.get_combined_input()

		# We'll make and train an EntityEmbedder in a moment.  However,
		# first we will get the IDs for the entities that occur together
		# within the test corpus.  We'll be interested to see the 
		# relationship embedding for them that is learnt during training
		edict = minibatch_generator.entity_dictionary
		expected_pairs = [
			('A','B'), ('C','D'), ('E','F')
		]
		expected_pairs_ids = [
			(edict.get_id(e1), edict.get_id(e2)) 
			for e1, e2 in expected_pairs
		]

		# We will repeatedly make an EntityEmbedder, train it on the 
		# test corpus, and then find its embedding for the entity-pairs
		# of interest.  We do it num_replicate # of times to average
		# results over enough trials that we can test if the learnt
		# embeddings have roughly the expected properties
		embedding_dot_products  = []
		for replicate in range(num_replicates):

			# Make an EntityEmbedder
			entity_embedder = EntityEmbedder(
				combined_input,
				batch_size,
				minibatch_generator.entity_vocab_size(),
				minibatch_generator.context_vocab_size(),
				num_embedding_dimensions
			)

			# Get its output and make a theano training function
			output = entity_embedder.get_output()
			params = entity_embedder.get_params()
			train = noise_contraster.get_train_func(output, params)

			# Train on the dataset, running through it num_epochs # of times
			for epoch in range(num_epochs):
				#print 'Epoch %d' % epoch

				for signal_batch, noise_batch in minibatch_generator:
					loss = train(signal_batch, noise_batch)
					#print loss

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



class TestMinibatchGenerator(TestCase):

	def setUp(self):
		'''
		Prepare a MinibatchGenerator so it is available in tests.
		'''
		self.files = ['test-data/test-corpus/a.tsv']
		self.batch_size = 5
		self.noise_ratio = 15
		self.mini_gen = MinibatchGenerator(
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


	def test_save_load(self):
		'''
		Try saving and reloading a MinibatchGenerator
		'''
		if os.path.exists('test-data/test-minibatch-generator'):
			shutil.rmtree('test-data/test-minibatch-generator')
		
		files = ['test-data/test-corpus/a.tsv']
		batch_size = 5
		noise_ratio = 15
		mini_gen = MinibatchGenerator(
			files=self.files,
			batch_size=self.batch_size, 
			noise_ratio=self.noise_ratio,
			verbose=False
		)
		mini_gen.prepare(savedir='test-data/test-minibatch-generator')

		new_mini_gen = MinibatchGenerator(
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

		new_new_mini_gen = MinibatchGenerator(
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
		MinibatchGenerator can produce minibatches asynchronously (meaning
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

		# Collect the minibatches by treating the MinibatchGenerator
		# as an iterator.  This leads to "asynchronous" multiprocessing
		# batch generation
		np.random.seed(1)
		for minibatch in self.mini_gen:
			async_batches.append(minibatch)

		# Collect the minibatches by calling 
		# `MinibatchGenerator.generate()`, which produces minibatches
		# using the "generator" construct, without any multiprocessing
		np.random.seed(1)
		for minibatch in self.mini_gen.generate_minibatches():
			generator_batches.append(minibatch)

		# Collect the minibatches by calling 
		# `MinibatchGenerator.get_minibatches()`, which returns a simple
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
		# Construct the expected entities and contexts
		corpus_reader = CorpusReader(
			files=['test-data/test-corpus/a.tsv'],
			parse=parse
		)
		valid_tokens = defaultdict(set)
		for line in corpus_reader.read_no_q():
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
