import numpy as np
import kernels as k
from nltk.corpus import wordnet, wordnet_ic
import classifier
from unittest import TestCase, main


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

			# Skip the "None" similarity type
			if similarity_type is None:
				continue

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
