import numpy as np
import kernels as k
from nltk.corpus import wordnet, wordnet_ic
import classifier
import extract_features
from unittest import TestCase, main

class TestFeatureAccumulator(TestCase):

    def test_get_dep_tree_features(self):
        # Make a mock (empty) dictionary (does not affect test, but needed to 
        # create the feature accumulator).
        dictionary = set()

        # Make a mock dependency tree
        F = {
            'parents':[],
            'children':[],
            'pos':'pos_F'
        }
        E = {
            'parents':[('rel_F', F)],
            'children':[],
            'pos':'pos_E'
        }
        D = {
            'parents':[],
            'children':[],
            'pos':'pos_D'
        }
        C = {
            'parents':[('rel_E', E)],
            'children':[('rel_D', D)],
            'pos':'pos_C'
        }
        B = {
            'parents':[],
            'children':[],
            'pos':'pos_B'
        }
        BB = {
            'parents':[],
            'children':[],
            'pos':'pos_BB'
        }
        A = {
            'parents':[('rel_C', C)],
            'children':[('rel_B', B), ('rel_BB', BB)],
            'pos':'pos_A'
        }

        accumulator = extract_features.FeatureAccumulator(dictionary)
        features = accumulator.get_dep_tree_features_recurse(A, depth=2)

        # Note that because we called it with depth=2, no feature is made for 
        # token F
        expected_features = [
            'parent:rel_C:pos_C', 'parent:rel_C:pos_C-parent:rel_E:pos_E',
            'parent:rel_C:pos_C-child:rel_D:pos_D', 'child:rel_B:pos_B',
            'child:rel_BB:pos_BB'
        ]

        self.assertItemsEqual(features, expected_features)


class TestKernel(TestCase):

	def test_k(self):
		features = classifier.load_features()
		dictionary = classifier.get_dictionary(features)
		information_content = wordnet_ic.ic('ic-treebank-resnik-add1.dat')

		test_tokens = ['ceo','coach', 'manager','boss', 'brother','sister']
		test_ids = [[dictionary.get_id(token)] for token in test_tokens]
		print test_ids

		# First we test the syntactic kernel
		kernel = k.bind_kernel(
			features,
			dictionary,
			syntactic_similarity=True,
			syntactic_multiplier=1.0
		)
		found_results = kernel(test_ids, test_ids)
		expected_results = self.get_expected_results(
			test_tokens, 
			lambda x,y: 1.0 * k.dict_dot(features[x], features[y])
		)
		self.assertEqual(found_results, expected_results)

		print '\n' + '-'*70 + '\n'
		print 'syntactic'
		print np.round(np.array(expected_results), 3)

		# Next test each of the semantic similarities
		for similarity_type in k.LEGAL_SIMILARITIES:

			kernel = k.bind_kernel(
				features,
				dictionary,
				syntactic_similarity=False,
				semantic_similarity=similarity_type,
				semantic_multiplier=1.0
			)
			found_results = kernel(test_ids, test_ids)
			expected_results = self.get_expected_results(
				test_tokens, 
				lambda x,y: 1.0 * k.max_similarity(
					similarity_type,
					k.nouns_only(wordnet.synsets(x)),
					k.nouns_only(wordnet.synsets(y)), 
					information_content
				)
			)
			print '\n' + '-'*70 + '\n'
			print similarity_type
			print np.round(np.array(expected_results), 3)
			self.assertEqual(found_results, expected_results)

	def get_expected_results(self, token_list, func):

		expected_results = []
		for token_a in token_list:
			expected_result_row = []
			expected_results.append(expected_result_row)
			for token_b in token_list:
				expected_result_row.append(
					func(token_a, token_b)
				)

		return expected_results




if __name__ == '__main__':
	main()
