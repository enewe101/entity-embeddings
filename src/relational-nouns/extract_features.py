import numpy as np
import json
from sklearn import svm, neighbors
from iterable_queue import IterableQueue
import t4k
from multiprocessing import Process
import time
import os
from corenlp_xml_reader import AnnotatedText
import sys
sys.path.append('..')
from subprocess import check_output
from word2vec import UnigramDictionary
from collections import defaultdict, Counter
from SETTINGS import (
	RELATIONAL_NOUN_SEEDS_PATH, NONRELATIONAL_NOUN_SEEDS_PATH, FEATURES_PATH
)
from nltk.corpus import wordnet


NUM_ARTICLE_LOADING_PROCESSES = 12
	

def get_seeds():
	positive_seeds = open(RELATIONAL_NOUN_SEEDS_PATH).read().split()
	positive_seeds = set([
		s for s in positive_seeds if not s.startswith('?')])

	negative_seeds = open(NONRELATIONAL_NOUN_SEEDS_PATH).read().split()
	negative_seeds = set([
		s for s in negative_seeds if not s.startswith('?')])

	return positive_seeds, negative_seeds


class WordNetFeatures(object):

	def __init__(self):
		pass

	def get_wordnet_features(self, lemma):
		synsets = wordnet.synsets(lemma)
		



def get_fnames():

	# Get the absolute path to all the corenlp files under batch '9fd'
	path = '/home/ndg/dataset/gigaword-corenlp/data/9fd/CoreNLP'
	fnames = check_output(['ls %s' % path], shell=True).split()
	fnames = [os.path.join(path, fname) for fname in fnames]

	return fnames


def extract_and_save_features(limit=100, path=FEATURES_PATH):
	features = extract_all_features(limit)
	open(path, 'w').write(json.dumps(features))


def extract_all_features(limit=100):

	start = time.time()

	# First, make an iterable queue and load all the article fnames onto it
	fnames_q = IterableQueue()
	fnames_producer = fnames_q.get_producer()
	for fname in get_fnames()[:limit]:
		fnames_producer.put(fname)
	fnames_producer.close()

	# Make a queue to hold feature stats (results), and a consumer to 
	# receive completed feature stats objects from workers
	features_q = IterableQueue()
	features_consumer = features_q.get_consumer()

	# Create workers that consume filenames and produce feature counts.
	for p in range(NUM_ARTICLE_LOADING_PROCESSES):
		fnames_consumer = fnames_q.get_consumer()
		features_producer = features_q.get_producer()
		process = Process(
			target=extract_features_from_articles,
			args=(fnames_consumer, features_producer)
		)
		process.start()

	# Close the iterable queues
	fnames_q.close()
	features_q.close()

	# Accumulate the results.  This blocks until workers are finished
	features = defaultdict(Counter)

	for add_features in features_consumer:
		for key in add_features:
			features[key] += add_features[key]

	#articles = [
	#	AnnotatedText(open(os.path.join(path, fname)).read()) 
	#	for fname in fnames[:100]
	#]

	elapsed = time.time() - start
	print 'elapsed', elapsed

	return features


def load_features(path=FEATURES_PATH):
	return json.loads(open(path).read())


def get_dictionary(features):
	counts = Counter({token:features[token]['count'] for token in features})
	dictionary = UnigramDictionary()
	dictionary.update(counts.elements())
	return dictionary


def filter_seeds(words, dictionary):
	'''
	Filters out any words in the list `words` that are not found in 
	the dictionary (and are hence mapped to UNK, which has id 0
	'''
	return [w for w in words if dictionary.get_id(w) != 0]


def get_classifier():
	positive_seeds, negative_seeds = get_seeds()
	features = load_features()
	dictionary = get_dictionary(features)
	positive_seeds = filter_seeds(positive_seeds, dictionary)
	negative_seeds = filter_seeds(negative_seeds, dictionary)
	classifier = make_svm_classifier(
		features, dictionary, positive_seeds, negative_seeds)
	return classifier, dictionary


def cross_val_positives(classifier='svm', clf_kwargs={}):
	positive_seeds, negative_seeds = get_seeds()
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
			clf = make_svm_classifier(*args, clf_kwargs=clf_kwargs)
		elif classifier == 'knn':
			clf = make_knn_classifier(*args, clf_kwargs=clf_kwargs)
		else:
			raise ValueError('Unexpected classifier type: %s.' % classifier)

		num_tested += 1
		prediction = clf.predict(dictionary.get_id(test_item))[0]
		if prediction:
			num_correct += 1

		padding = 40 - len(test_item)
		print test_item, ' ' * padding, 'correct' if prediction else '-'

	print (
		'\n' + '-'*70 + '\n\n' +
		'true positives / positives = %f' 
		% (num_correct / float(num_tested))
	)


def cross_val_negatives(classifier='svm', clf_kwargs={}):
	positive_seeds, negative_seeds = get_seeds()
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
			clf = make_svm_classifier(*args, clf_kwargs=clf_kwargs)
		elif classifier == 'knn':
			clf = make_knn_classifier(*args, clf_kwargs=clf_kwargs)
		else:
			raise ValueError('Unexpected classifier type: %s.' % classifier)


		num_tested += 1
		prediction = clf.predict(dictionary.get_id(test_item))[0]
		if not prediction:
			num_correct += 1

		padding = 40 - len(test_item)
		print test_item, ' ' * padding, '-' if prediction else 'correct'

	print (
		'\n' + '-'*70 + '\n\n' +
		'true negatives / negatives = %f' 
		% (num_correct / float(num_tested))
	)


def make_svm_classifier(
	features, dictionary, positive_seeds, negative_seeds,
	clf_kwargs={}
):
	'''
	Make an SVM classifier
	'''
	X, Y = make_training_set(dictionary, positive_seeds, negative_seeds)
	classifier = svm.SVC(kernel=bind_kernel(features, dictionary))
	classifier.fit(X,Y)
	return classifier


def bind_dist(features, dictionary):
	def dist(X1, X2):
		token1, token2 = dictionary.get_tokens([X1[0], X2[0]])
		features1 = features[token1]
		features2 = features[token2]
		dot = dict_dot(features1, features2)
		if dot == 0:
			return np.inf
		return 1 / float(dot)
	return dist


def make_knn_classifier(
	features, dictionary, positive_seeds, negative_seeds,
	clf_kwargs={}
):
	'''
	Make a kNN classifier
	'''
	X, Y = make_training_set(dictionary, positive_seeds, negative_seeds)
	mydist = bind_dist(features, dictionary)
	classifier = KNN(metric=mydist, **clf_kwargs)
	classifier.fit(X,Y)
	return classifier


class KNN(object):

	def __init__(self, k=3, metric=None):
		self.k = k
		self.metric = metric
		if metric is None:
			self.metric = self.euclidean

	def euclidean(self, z,x):
		return np.linalg.norm(z-x)


	def fit(self, X, Y):
		self.X = X
		self.Y = Y


	def predict(self, z):

		if not isinstance(z, np.ndarray):
			z = np.array(z)
		if len(z.shape) == 0:
			z = np.array([[z]])
		elif len(z.shape) == 1:
			z = np.array([z])
		elif len(z.shape) > 2:
			raise ValueError(
				'Predict accepts a list of examples whose labels are '
				'to be predicted, or a single example.  Each examples '
				'should be a feature vector (or the type accepted by '
				'distance metric).'
			)

		# Make predictions for each example
		predictions = []
		for row in z:
			predictions.append(self._predict(row))

		return predictions


	def _predict(self, z):

		distances = [np.inf]*self.k
		labels = [None]*self.k

		# Iterate over all stored examples.  If the distance beats the
		# stored examples, keep it and the associated label
		for idx in range(len(self.X)):
			x, l = self.X[idx], self.Y[idx]
			d = self.metric(z,x)
			distances, labels = self.compare(d, l, distances, labels)

		# Now return the majority vote on the label
		return Counter(labels).most_common(1)[0][0]


	def compare(self, d, l, distances, labels):
		'''
		compare the distance d to the sorted (ascending) list of distances,
		starting from the end.  If d is less than any of the items, put it
		in its proper location (preserving sortedness), then truncate the
		list back to its original length.  Put l in the corresponding
		location in labels, and truncate it too.
		'''

		# First compare to the least element
		ptr = self.k
		while ptr > 0 and d < distances[ptr-1]:
			ptr -= 1

		if ptr < self.k:
			distances.insert(ptr, d)
			distances.pop()
			labels.insert(ptr, l)
			labels.pop()

		return distances, labels



def make_training_set(dictionary, positive_seeds, negative_seeds):
	# Make the training set.  Each "row" in the training set has a single 
	# "feature" -- it's the id which identifies the token.  
	# This will let us lookup the non-numeric features in kernel functions
	X = (
		[ [dictionary.get_id(s)] for s in positive_seeds]
		+ [ [dictionary.get_id(s)] for s in negative_seeds]
	)
	Y = [True] * len(positive_seeds) + [False] * len(negative_seeds)

	return X, Y


def bind_kernel(features, dictionary):
	'''
	Returns a kernel function that has a given dictionary and features
	lookup bound to its scope.
	'''

	def kernel(A,B):
		'''
		Custom kernel function.  This counts how often the links incident on
		two different words within their respective dependency trees are the 
		same, up to the dependency relation and the POS of the neighbour.

		Note that A references a set of words' dependency trees, and B
		references another set.  So that this function end up making
		len(A) * len(B) of such comparisons, and return the result as a 
		len(A) by len(B) matrix.
		'''
		result = []
		for a in A:
			token_a = dictionary.get_token(int(a[0]))
			features_a = features[token_a]
			result_row = []
			result.append(result_row)
			for b in B:
				token_b = dictionary.get_token(int(b[0]))
				features_b = features[token_b]
				result_row.append(dict_dot(features_a, features_b))

		return result

	return kernel


def dict_dot(a,b):
	result = 0
	for key in a:
		if key != 'count' and key in b:
			result += a[key] * b[key] / float(a['count'] * b['count'])
	return result
		

def extract_features_from_articles(fnames_consumer, features_producer):

	features = defaultdict(Counter)

	# Read articles named in fnames_consumer, and accumulate features from 
	# them
	for fname in fnames_consumer:
		article = AnnotatedText(open(fname).read())
		add_features = extract_features_from_article(article)
		for key in add_features:
			features[key] += add_features[key]

	# Put the accumulated features onto the producer queue then close it
	features_producer.put(features)
	features_producer.close()


def extract_features_from_article(article):
	features = defaultdict(Counter)

	for sentence in article.sentences:
		for token in sentence['tokens']:

			# We're only interested in non-proper nouns
			if token['pos'] not in ('NN', 'NNS'):
				continue

			# Add features seen for this instance.  We use the lemma as
			# the key around which to aggregate features for the same word
			add_features = extract_features_from_token(token)
			features[token['lemma']] += add_features

			# We also keep count of the number of times a given token is
			# seen which helps with normalization of the feature dict
			features[token['lemma']]['count'] += 1

	return features


def extract_features_from_token(token):
	'''
	Extracts a set of features based on the dependency tree relations
	for the given token.  Each feature describes one dependency tree 
	relation in terms of a "signature".  The signature of a dependency
	tree relation consists of whether it is a parent or child, what the 
	relation type is (e.g. nsubj, prep:for, etc), and what the pos of the 
	target is.
	'''
	add_features = Counter()

	# Record all parent signatures
	for relation, token in token['parents']:
		signature = '%s:%s:%s' % ('parent', relation, token['pos'])
		#signature = '%s:%s:%s' % ('parent', relation, token['ner'])
		add_features[signature] += 1

	# Record all 
	for relation, token in token['children']:
		signature = '%s:%s:%s' % ('child', relation, token['pos'])
		#signature = '%s:%s:%s' % ('child', relation, token['ner'])
		add_features[signature] += 1

	return add_features

	

