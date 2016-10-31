import t4k
import numpy as np
from sklearn import svm
import json
import sys
sys.path.append('..')
from word2vec import UnigramDictionary, UNK, SILENT
from collections import Counter, deque
from SETTINGS import DEPENDENCY_FEATURES_PATH, BASELINE_FEATURES_PATH
from kernels import bind_kernel, bind_dist
from utils import (
	read_seed_file, get_training_sets, filter_seeds, ensure_unicode
)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as w



def make_classifier(
	kind='svm',	# available: 'svm', 'knn', 'wordnet', 'basic_syntax'
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
	Convenience method to create a RelationalNounClassifier using
	default seed data and feature data
	'''
	positive_seeds, negative_seeds = get_training_sets()
	features = load_features()
	dictionary = get_dictionary(features)
	positive_seeds = filter_seeds(positive_seeds, dictionary)
	negative_seeds = filter_seeds(negative_seeds, dictionary)

	if kind == 'svm':
		return SvmNounClassifier(
			positive_seeds,
			negative_seeds,
			# SVM options
			features,
			dictionary,
			on_unk=False,
			syntactic_similarity=True,
			semantic_similarity=None,
			syntactic_multiplier=1.0,
			semantic_multiplier=1.0,
		)

	elif kind == 'knn':
		return KnnNounClassifier(
			positive_seeds,
			negative_seeds,
			# KNN options
			features,
			dictionary,
			on_unk=False,
			k=3
		)

	elif kind == 'wordnet':
		return WordnetClassifier(positive_seeds, negative_seeds):

	elif kind == 'basic_syntax':
		return BasicSyntaxNounClassifier(
			positive_seeds,
			negative_seeds,
			features,
			dictionary,
		)



class WordnetClassifier(object):

	def __init__(self, positive_seeds, negative_seeds):
		# Register the arguments locally
		self.positive_seeds = positive_seeds
		self.negative_seeds = negative_seeds
		self.fit()


	def fit(self):
		'''
		Get all the synsets that correspond to the positive and negative
		seeds
		'''
		self.positive_synsets = get_all_synsets(self.positive_seeds)
		self.negative_synsets = get_all_synsets(self.negative_seeds)


	def predict(self, tokens):

		# We expect an iterable of strings, but we can also accept a single
		# string.  If we got a single string, put it in a list.
		tokens = maybe_list_wrap(tokens)

		# Get the lemmas for the word to predict
		predictions = []
		for token in tokens:
			synset_deque = deque(w.synsets(token))
			predictions.append(self.predict_one(synset_deque))

		return predictions


	def predict_one(self, synset_deque):
		'''
		Do a breadth-first search, following the hypernyms of `lemma`
		in Wordnet, until one of the following conditionsn is met: 
			1) an synset corresponding to a positive seed is reached
			2) a synset corersponding to a negative seed is reached
			3) the hypernym root ("entity") is reached.
		if 1) occurs, return `True`, otherwise return `False`
		'''
		while len(synset_deque) > 0:

			# Pull the next synset of the queue (or in this case, deque)
			next_synset = synset_deque.popleft()

			# If we encountered a positive seed, return True
			if next_synset in self.positive_synsets:
				return True

			# If we encountered a negative seed, return True
			elif next_synset in self.negative_synsets:
				return False

			# Otherwise, add the parents (hypernyms), to be searched later
			else:
				synset_deque.extend(next_synset.hypernyms())

		# If we hit the hypernym root without finding any seeds, then
		# assume that the word is not relational (statistically most aren't)
		return False

	def score(self, tokens):
		'''
		The wordnet classifier inherently produces scores that are either
		1 or 0, so we just call to predict here.
		'''
		return self.predict(tokens)

			


class SvmNounClassifier(object):

	def __init__(
		self,
		positive_seeds,
		negative_seeds,

		# SVM options
		features,
		dictionary,
		on_unk=False,
		syntactic_similarity=True,
		semantic_similarity=None,
		syntactic_multiplier=1.0,
		semantic_multiplier=1.0,
	):
		'''
		on_unk [-1 || any]: (Stands for "on unknown token").  This
			controls behavior when a prediction is requested for a token 
			that did not appear in the dictionary:
				* -1 -- raise ValueError
				* anything else -- return that as the predicted class
		'''

		# Register parameters locally
		self.positive_seeds = positive_seeds
		self.negative_seeds = negative_seeds

		# Register SVM options
		self.features = features
		self.dictionary = dictionary
		self.on_unk = on_unk
		self.syntactic_similarity = syntactic_similarity 
		self.semantic_similarity = semantic_similarity  
		self.syntactic_multiplier = syntactic_multiplier 
		self.semantic_multiplier = semantic_multiplier  

		# Make the underlying classifier
		self.classifier = self.make_svm_classifier()

	def make_svm_classifier(self):
		'''
		Make an SVM classifier
		'''
		X, Y = make_training_set(
			self.positive_seeds, self.negative_seeds, self.dictionary)
		kernel = bind_kernel(
			self.features, self.dictionary, 
			syntactic_similarity=self.syntactic_similarity,
			semantic_similarity=self.semantic_similarity,
			syntactic_multiplier=self.syntactic_multiplier,
			semantic_multiplier=self.semantic_multiplier,
		)
		classifier = svm.SVC(kernel=kernel)
		classifier.fit(X,Y)
		return classifier


	def handle_unk(self, func, ids, lemmas):

		try:
			return func(ids)

		# If we get a ValueError, try to report the offending word
		except ValueError as e:
			try:
				offending_lemma = lemmas[e.offending_idx]
			except AttributeError:
				raise
			else:
				raise ValueError('Unrecognized word: %s' % offending_lemma)


	def predict(self, tokens):

		# and need special handling of UNK tokens
		ids, lemmas = self.convert_tokens(tokens)
		return self.handle_unk(self.predict_id, ids, lemmas)


	def convert_tokens(self, tokens):
		# Expects a list of lemmas, but can accept a single lemma too:
		# if that's what we got, put it in a list.
		if isinstance(tokens, basestring):
			tokens = [tokens]
		# Ensure tokens are unicode, lemmatize, and lowercased
		lemmas = lemmatize_many(tokens)

		# Convert lemma(s) to token_ids
		return self.dictionary.get_ids(lemmas), lemmas


	def score(self, tokens):
		ids, lemmas = self.convert_tokens(tokens)
		return self.handle_unk(self.score_id, ids, lemmas)
		

	def score_id(self, token_ids):
		scores = []
		for i, token_id in enumerate(token_ids):

			# Handle cases where the token_id corresponds to unknown token.
			if token_id == UNK:

				# Raise an error on unrecognized token_id (if desired)
				if self.on_unk < 0:
					e = ValueError('Unrecognized token_id: %d' % token_id)
					e.offending_idx = i
					raise e

				# if desired behavior for unrecognized tokens is to 
				# classify as False, then return a score of negative infinty
				elif self.on_unk == 0:
					scores.append(-np.inf)

				# if desired behavior for unrecognized tokens is to classify
				# as True, then return the smallest positive value.
				else:
					scores.append(np.finfo(float).eps)

				continue

			scores.append(
				self.classifier.decision_function([[token_id]])[0]
			)

		return scores


	def predict_id(self, token_ids):
		predictions = []
		for i, token_id in enumerate(token_ids):

			# Handle cases where the token_id corresponds to unknown token.
			if token_id == UNK:

				# Raise an error on unrecognized token_id (if desired)
				if self.on_unk < 0:
					e = ValueError('Unrecognized token_id: %d' % token_id)
					e.offending_idx = i
					raise e

				# Otherwise return the value assigned to on_unk 
				# as the class (default False)
				else:
					predictions.append(self.on_unk)

				continue

			predictions.append(self.classifier.predict([[token_id]])[0])

		return predictions


	def make_training_set(self):
		# Make the training set.  Each "row" in the training set has a 
		# single "feature" -- it's the id which identifies the token.  
		# This will let us lookup the non-numeric features in kernel 
		# functions
		X = (
			[ [self.dictionary.get_id(s)] for s in self.positive_seeds]
			+ [ [self.dictionary.get_id(s)] for s in self.negative_seeds]
		)
		Y = (
			[True] * len(self.positive_seeds) 
			+ [False] * len(self.negative_seeds)
		)

		return X, Y


class KnnNounClassifier(object):
	'''
	Class that wraps underlying classifiers and handles training, testing,
	and prediction logic that is specific to making a relational noun 
	classifier.
	'''

	def __init__(
		self,
		positive_seeds,
		negative_seeds,

		# KNN options
		features,
		dictionary,
		on_unk=False,
		k=3,
	):
		'''
		on_unk [-1 || any]: (Stands for "on unknown token").  This
			controls behavior when a prediction is requested for a token 
			that did not appear in the dictionary:
				* -1 -- raise ValueError
				* anything else -- return that as the predicted class
		'''

		# Register parameters locally
		self.positive_seeds = positive_seeds
		self.negative_seeds = negative_seeds

		# Register KNN options
		self.features = features
		self.dictionary = dictionary
		self.on_unk = on_unk
		self.k = k

		# Make the underlying classifier
		self.classifier = self.make_knn_classifier()


	def make_knn_classifier(self):
		'''
		Make a kNN classifier
		'''
		X, Y = make_training_set(
			self.positive_seeds, self.negative_seeds, self.dictionary)
		mydist = bind_dist(self.features, self.dictionary)
		classifier = KNN(metric=mydist, k=k)
		classifier.fit(X,Y)
		return classifier


	def handle_unk(self, func, ids, lemmas):

		try:
			return func(ids)

		# If we get a ValueError, try to report the offending word
		except ValueError as e:
			try:
				offending_lemma = lemmas[e.offending_idx]
			except AttributeError:
				raise
			else:
				raise ValueError('Unrecognized word: %s' % offending_lemma)


	def predict(self, tokens):
		ids, lemmas = self.convert_tokens(tokens)
		return self.handle_unk(self.predict_id, ids, lemmas)


	def convert_tokens(self, tokens):
		# Expects a list of lemmas, but can accept a single lemma too:
		# if that's what we got, put it in a list.
		if isinstance(tokens, basestring):
			tokens = [tokens]
		# Ensure tokens are unicode, lemmatize, and lowercased
		lemmas = lemmatize_many(tokens)

		# Convert lemma(s) to token_ids
		return self.dictionary.get_ids(lemmas), lemmas


	def score(self, tokens):
		ids, lemmas = self.convert_tokens(tokens)
		return self.handle_unk(self.score_id, ids, lemmas)
		

	def score_id(self, token_ids):
		scores = []
		for i, token_id in enumerate(token_ids):

			# Handle cases where the token_id corresponds to unknown token.
			if token_id == UNK:

				# Raise an error on unrecognized token_id (if desired)
				if self.on_unk < 0:
					e = ValueError('Unrecognized token_id: %d' % token_id)
					e.offending_idx = i
					raise e

				# if desired behavior for unrecognized tokens is to 
				# classify as False, then return a score of negative infinty
				elif self.on_unk == 0:
					scores.append(-np.inf)

				# if desired behavior for unrecognized tokens is to classify
				# as True, then return the smallest positive value.
				else:
					scores.append(np.finfo(float).eps)

				continue

			scores.append(
				self.classifier.decision_function([[token_id]])[0]
			)

		return scores


	def predict_id(self, token_ids):
		predictions = []
		for i, token_id in enumerate(token_ids):

			# Handle cases where the token_id corresponds to unknown token.
			if token_id == UNK:

				# Raise an error on unrecognized token_id (if desired)
				if self.on_unk < 0:
					e = ValueError('Unrecognized token_id: %d' % token_id)
					e.offending_idx = i
					raise e

				# Otherwise return the value assigned to on_unk 
				# as the class (default False)
				else:
					predictions.append(self.on_unk)

				continue

			predictions.append(self.classifier.predict([[token_id]])[0])

		return predictions


	def make_training_set(self):
		# Make the training set.  Each "row" in the training set has a 
		# single "feature" -- it's the id which identifies the token.  
		# This will let us lookup the non-numeric features in kernel 
		# functions
		X = (
			[ [self.dictionary.get_id(s)] for s in self.positive_seeds]
			+ [ [self.dictionary.get_id(s)] for s in self.negative_seeds]
		)
		Y = (
			[True] * len(self.positive_seeds) 
			+ [False] * len(self.negative_seeds)
		)

		return X, Y




class KNN(object):
	'''
	K-Nearest Neighbors classifier which accepts a custom distance function.
	I wrote this because the scikit KNN doesn't work well with custom
	distance functions that aren't true metrics, e.g. those based on 
	cosine distance
	'''

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



###	Helper functions


def load_features(path=DEPENDENCY_FEATURES_PATH):
	return json.loads(open(path).read())



def get_dictionary(features):
	counts = Counter({token:features[token]['count'] for token in features})
	dictionary = UnigramDictionary(on_unk=SILENT)
	dictionary.update(counts.elements())
	return dictionary

def get_all_synsets(tokens):
	return set(t4k.flatten([
		w.synsets(lemma) for lemma in lemmatize_many(tokens)
	]))


LEMMATIZER = WordNetLemmatizer()
def lemmatize(token):
	return LEMMATIZER.lemmatize(ensure_unicode(token).lower())

def lemmatize_many(tokens):
	return [lemmatize(token) for token in tokens ]


def maybe_list_wrap(tokens):
	'''
	Checks to see if `tokens` actually corresponds to a single token, in
	which case it gets put into a list.  Helps to overload methods so that
	they can accept a single token even though they are designed to accept
	an iterable of tokens.
	'''
	if isinstance(tokens, basestring):
		return [tokens]
	return tokens


def make_training_set(positive_seeds, negative_seeds, dictionary):
	'''
	Make the training set in the format expected by scikit's svm classifier,
	and my KNN classifier, based on positive and negative examples and
	a dictionary.  
	
	Each "row" in the training set has a 
	single "feature" -- it's the id which identifies the token.  
	This will let us lookup the non-numeric features in kernel 
	functions
	'''
	X = (
		[ [dictionary.get_id(s)] for s in positive_seeds]
		+ [ [dictionary.get_id(s)] for s in negative_seeds]
	)
	Y = (
		[True] * len(positive_seeds) 
		+ [False] * len(negative_seeds)
	)

	return X, Y
