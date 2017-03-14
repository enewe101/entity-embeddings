from functools import partial
import json
import os
import sys
sys.path.append('..')
from SETTINGS import DATA_DIR
from word2vec import UnigramDictionary, SILENT
import classifier as c
DICTIONARY_DIR = os.path.join(DATA_DIR, 'dictionary')
from utils import (
	read_seed_file, get_train_sets, filter_seeds, ensure_unicode,
	get_test_sets
)
from nltk.stem import WordNetLemmatizer

UNRECOGNIZED_TOKENS_PATH = os.path.join(DATA_DIR, 'unrecognized-tokens.txt')


def load_dictionary(dictionary_dir=DICTIONARY_DIR):
	dictionary = UnigramDictionary(on_unk=SILENT)
	dictionary.load(dictionary_dir)
	return dictionary


def get_MAP(n, ranks):
	'''
	Get ther Mean Average Precision for the first n relevant documents.
	`ranks` should give the rank for each of the first n documents, 
	zero-indexed.
	'''

	# Define the MAP for zero results to be 1
	if n == 0:
		return 1.0

	# Look at the rank for the first n relevant documents (or as many are
	# available, if less than n) and accumulate the contributions each 
	# makes to the MAP
	numerator = 0
	precision_sum = 0
	average_precision_sum = 0
	for k in range(n):

		# Calculate the contribution of the kth relevant document 
		# to the MAP numerator.  If the k'th relevant doc didn't exist,
		# then it contributes nothing
		try:
			precision_sum += (k+1) / float(ranks[k]+1)
		except IndexError:
			pass

		average_precision_sum += precision_sum / float(k+1)

	mean_average_precision = average_precision_sum / float(n)

	return mean_average_precision


def get_top(
	n=50,
	kind='svm',
	on_unk=False,

	# SVM options
	syntactic_similarity=True,
	semantic_similarity=None,
	syntactic_multiplier=1.0,
	semantic_multiplier=1.0,

	# KNN options
	k=3,
):
	'''
	Build a classifier based on the given settings, and then return the
	n highest-scoring words
	'''
	evaluator = get_map_evaluator(
		kind=kind,
		on_unk=on_unk,
		syntactic_similarity=syntactic_similarity,
		semantic_similarity=semantic_similarity,
		syntactic_multiplier=syntactic_multiplier,
		semantic_multiplier=semantic_multiplier,
		k=k,
	)
	evaluator.get_scores()
	print '\n'.join([s[1] for s in evaluator.scores[:n]])


def diagnose_map_evaluators(
	classifier_definitions=[],
	map_evaluators=None,
	out_path=UNRECOGNIZED_TOKENS_PATH,
	n=100
):
	'''
	Given a set of classifier definitions (which should be a list of
	dictionaries containing keyword arguments for the function 
	get_map_evaluator), create the classifier for each, and find the
	unrecognized tokens (i.e. not in the test set) for each of the 
	classifier's n top-scoring tokens
	'''
	out_file = open(UNRECOGNIZED_TOKENS_PATH, 'w')

	# Create the map evaluators, if they weren't supplied
	if map_evaluators is None:
		map_evaluators = [
			(cdef, get_map_evaluator(**cdef)) 
			for cdef in classifier_definitions
		]

	# For each classifier, find the unrecognized tokens among the top n
	# scoring tokens, and write them to file
	for cdef, map_evaluator in map_evaluators:
		unrecognized_tokens = map_evaluator.diagnose_MAP(n)
		out_file.write(json.dumps(cdef)+'\n')
		out_file.write('\n'.join(unrecognized_tokens) + '\n\n')

	return map_evaluators


def get_map_evaluator(
	kind='svm',
	on_unk=False,
	dictionary_dir=DICTIONARY_DIR,
	min_frequency=5,
	syntactic_similarity=True,
	semantic_similarity=None,
	syntactic_multiplier=1.0,
	semantic_multiplier=1.0,
	k=3,
):
	classifier = c.make_classifier(
		kind=kind,
		on_unk=on_unk,
		syntactic_similarity=syntactic_similarity,
		semantic_similarity=semantic_similarity,
		syntactic_multiplier=syntactic_multiplier,
		semantic_multiplier=semantic_multiplier,
		k=k,
	)
	train_positives, train_negatives = get_train_sets()
	test_positives, test_negatives = get_test_sets()
	evaluator = RelationalNounMapEvaluator(
		classifier, train_positives, train_negatives, test_positives,
		test_negatives, dictionary_dir=dictionary_dir,
		min_frequency=min_frequency
	)

	return evaluator


def get_descriminant_func(test_positives=None, test_negatives=None):

	if test_positives is None or test_negatives is None:
		test_positives, test_negatives = get_test_sets()
	
	def descriminant_func(result):
		try:
			if result in test_positives:
				return True
			elif result in test_negatives:
				return False
			else:
				raise ValueError(
					'Cannot decide if %s is relevant' % str(result)
				)
		except TypeError:
			print type(result), repr(result)

	return descriminant_func



class RelationalNounMapEvaluator(object):

	def __init__(
		self, 
		classifier,
		train_positives, 
		train_negatives,
		test_positives,
		test_negatives,
		dictionary_dir=DICTIONARY_DIR, 
		min_frequency=5
	):

		# Register parameters
		self.classifier = classifier
		self.training_data = train_positives | train_negatives
		self.testing_data = test_positives | test_negatives
		self.test_positives = test_positives
		self.test_negatives = test_negatives
		self.dictionary = load_dictionary(dictionary_dir)
		self.min_frequency = min_frequency

		# Prune the dictionary to `min_frequency` if it isn't Falsey
		if min_frequency:
			self.dictionary.prune(min_frequency)

		# Make a MAP evaluator
		descriminant_func = get_descriminant_func(
			test_positives, test_negatives
		)
		self.evaluator = MapEvaluator(descriminant_func)

		# This will hold the "relational-nounishness-scores" to rank the
		# nouns that seem most relational
		self.scores = None

		# NOTE: this can be removed once we have a pre-lemmatized dictionary
		self.lemmatizer = WordNetLemmatizer()


	def diagnose_MAP(self, n):
		# Get classifier's scores for all tokens (if not already done)
		if self.scores is None:
			self.get_scores()

		# Go through the top n tokens and collect all unrecognized ones
		print 'checking...'
		num_seen = 0
		unrecognized_tokens = []
		top_tokens = (s[1] for s in self.scores)
		for token in top_tokens:

			# Check if this token is recognized
			if token not in self.testing_data:
				unrecognized_tokens.append(token)

			# Only look at the top n tokens
			num_seen += 1
			if num_seen >= n:
				break

		return unrecognized_tokens


	def get_MAP(self, n):
		if self.scores is None:
			self.get_scores()

		top_tokens = (s[1] for s in self.scores)
		return self.evaluator.get_MAP(n, top_tokens)


	def get_scores(self):
		# Get the scores for each token in the dictionary
		# Skip any tokens that were in the training_data!
		# NOTE: this can be removed once I make a lematized dictionary 
		print 'lemmatizing...'
		token_list = set([
			self.lemmatizer.lemmatize(ensure_unicode(token).lower())
			for token in self.dictionary.get_token_list()
		])
		print 'scoring...'
		self.scores = [
			(self.classifier.score(token), token)
			for token in token_list
			if token not in self.training_data
		]
		print 'sorting...'
		self.scores.sort(reverse=True)



class MapEvaluator(object):
	'''
	Class for evaluating Mean Average Precision for some result-iterator.
	It requires a `descriminant_funct` -- a function that can distinguish 
	which results are relevant and which are irrelevant.
	'''

	def __init__(self, descriminant_func):
		self.descriminant_func = descriminant_func


	def get_MAP(self, n, results_iterator):
		'''
		Given some iterator that yields results (`results_iterator`), 
		calculate it's Mean Average Precision. 
		'''
		ranks = self.get_ranks(n, results_iterator)
		return get_MAP(n, ranks)


	def get_ranks(self, n, results_iterator):
		'''
		Given some iterator that yields results (`results_iterator`), 
		find the ranks for the first n relevant results.  
		'''

		# If n is 0, we don't have to report any ranks at all
		if n == 0:
			return []

		i = 0		# indexes relevant results
		ranks = []	# stores ranks

		# Get the ranks for the first `n` relevant
		for rank, result in enumerate(results_iterator):

			# If a result is positive, save it's rank
			if self.descriminant_func(result):
				ranks.append(rank)
				i += 1

			# Stop if we've found `n` relevant results
			if i >= n:
				break

		return ranks


def cross_val_positives(classifier='svm', clf_kwargs={}, use_wordnet=False):
	positive_seeds, negative_seeds = get_train_sets()
	features = load_features()
	dictionary = get_dictionary(features)
	positive_seeds = filter_seeds(positive_seeds, dictionary)
	negative_seeds = filter_seeds(negative_seeds, dictionary)

	num_correct = 0
	num_tested = 0
	for test_item in positive_seeds:
		positive_seeds_filtered = [
			p for p in positive_seeds if p is not test_item
		]

		args = (features,dictionary,positive_seeds_filtered,negative_seeds)
		if classifier == 'svm':
			clf = make_svm_classifier(
				*args, clf_kwargs=clf_kwargs, use_wordnet=use_wordnet)
		elif classifier == 'knn':
			clf = make_knn_classifier(*args, clf_kwargs=clf_kwargs)
		else:
			raise ValueError('Unexpected classifier type: %s.' % classifier)

		num_tested += 1
		prediction = clf.predict(test_item)[0]
		if prediction:
			num_correct += 1

		padding = 40 - len(test_item)
		print test_item, ' ' * padding, 'correct' if prediction else '-'

	print (
		'\n' + '-'*70 + '\n\n' +
		'true positives / positives = %f' 
		% (num_correct / float(num_tested))
	)




def cross_val_negatives(classifier='svm', clf_kwargs={}, use_wordnet=False):
	positive_seeds, negative_seeds = get_train_sets()
	features = load_features()
	dictionary = get_dictionary(features)
	positive_seeds = filter_seeds(positive_seeds, dictionary)
	negative_seeds = filter_seeds(negative_seeds, dictionary)

	num_correct = 0
	num_tested = 0
	for test_item in negative_seeds:
		negative_seeds_filtered = [
			n for n in negative_seeds if n is not test_item
		]

		args = (features,dictionary,positive_seeds,negative_seeds_filtered)
		if classifier == 'svm':
			clf = make_svm_classifier(
				*args, clf_kwargs=clf_kwargs, use_wordnet=use_wordnet)
		elif classifier == 'knn':
			clf = make_knn_classifier(*args, clf_kwargs=clf_kwargs)
		else:
			raise ValueError('Unexpected classifier type: %s.' % classifier)

		num_tested += 1
		prediction = clf.predict(test_item)[0]
		if not prediction:
			num_correct += 1

		padding = 40 - len(test_item)
		print test_item, ' ' * padding, '-' if prediction else 'correct'

	print (
		'\n' + '-'*70 + '\n\n' +
		'true negatives / negatives = %f' 
		% (num_correct / float(num_tested))
	)

if __name__ == '__main__':
	diagnose_map_evaluators([{}])
