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


# TODO: make this test.  It should know what entity pairs and contexts
# have a high probability of arising with the data, following NCE form
class TestEntityEmbedder(TestCase):

	def test_learning(self):

		files = ['test-data/test-corpus/b.tsv']
		batch_size = 5
		noise_ratio = 15
		num_embedding_dimensions = 5
		num_epochs

		minibatch_generator = MinibatchGenerator(
			files=files,
			batch_size=batch_size,
			noise_ratio=noise_ratio,
			verbose=False
		)
		minibatch_generator.prepare()
		signal_input = T.matrix('signal')
		noise_input = T.matrix('noise')
		noise_contraster = NoiseContraster(signal_input, noise_input)
		combined_input = noise_contraster.get_combined_input()
		entity_embedder = EntityEmbedder(
			combined_input,
			batch_size,
			minibatch_generator.entity_vocab_size(),
			minibatch_generator.context_vocab_size(),
			num_embedding_dimensions
		)
		output = entity_embedder.get_output()
		params = entity_embedder.get_params()
		train = noise_contraster.get_train_func(output, params)

		for epoch in range(num_epochs):
			print 'Epoch %d' % epoch

			for signal_batch, noise_batch in minibatch_generator:
				loss = train(signal_batch, noise_batch)
				print loss

		expected_pairs = [(A,B),(C,D),(E,F),(E,G),(E,H),(F,G),(F,H),(G,H)]
		relationship_embedding = 
		for e1, e2 in expected_pairs:

		




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
			mini_gen.entity_dictionary.tokens,
			new_mini_gen.entity_dictionary.tokens
		)
		self.assertEqual(
			mini_gen.context_dictionary.tokens,
			new_mini_gen.context_dictionary.tokens
		)
		self.assertEqual(
			mini_gen.unigram.counts,
			new_mini_gen.unigram.counts
		)

		# Now we'll try using the component save and load functions
		# First remove the saved files
		shutil.rmtree('test-data/test-minibatch-generator')

		# These functions don't automatically make the parent directory
		# if it doesn't exist, so we need to make it
		os.mkdir('test-data/test-minibatch-generator')
		mini_gen.save_entity_dictionary(
			'test-data/test-minibatch-generator/entity-dictionary.gz'
		)
		mini_gen.save_context_dictionary(
			'test-data/test-minibatch-generator/context-dictionary.gz'
		)
		mini_gen.save_unigram(
			'test-data/test-minibatch-generator/unigram.gz'
		)

		new_new_mini_gen = MinibatchGenerator(
			files=self.files,
			batch_size=self.batch_size, 
			noise_ratio=self.noise_ratio,
			verbose=False
		)

		mini_gen.load_entity_dictionary(
			'test-data/test-minibatch-generator/entity-dictionary.gz'
		)
		mini_gen.load_context_dictionary(
			'test-data/test-minibatch-generator/context-dictionary.gz'
		)
		mini_gen.load_unigram(
			'test-data/test-minibatch-generator/unigram.gz'
		)

		# And again, check that the underlying data in the dictionaries and 
		# unigram are the same
		self.assertEqual(
			mini_gen.entity_dictionary.tokens,
			new_mini_gen.entity_dictionary.tokens
		)
		self.assertEqual(
			mini_gen.context_dictionary.tokens,
			new_mini_gen.context_dictionary.tokens
		)
		self.assertEqual(
			mini_gen.unigram.counts,
			new_mini_gen.unigram.counts
		)


	def test_generate_vs_async(self):
		'''
		MinibatchGenerator can produce minibatches asynchronously (meaning
		that it generates future minibatches before they are requested and
		stores them in a queue) or like an ordinary generator as the consumer 
		requests them.  Both methods should give the same results.
		'''
		generator_batches = []
		async_batches = []

		np.random.seed(1)
		for signal_batch, noise_batch in self.mini_gen:
			async_batches.append((signal_batch, noise_batch))

		np.random.seed(1)
		for signal_batch, noise_batch in self.mini_gen.generate():
			generator_batches.append((signal_batch, noise_batch))

		zipped_batches = zip(generator_batches, async_batches)
		for ((async_sig, async_noise), (gen_sig, gen_noise)) in zipped_batches:
			self.assertTrue(np.equal(async_sig, gen_sig).all())
			self.assertTrue(np.equal(async_noise, gen_noise).all())


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
