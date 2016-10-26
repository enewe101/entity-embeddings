import numpy as np
from sklearn import svm
import json
import sys
sys.path.append('..')
from word2vec import UnigramDictionary, UNK, SILENT
from collections import Counter
from SETTINGS import FEATURES_PATH
from kernels import bind_kernel, bind_dist
from utils import (
	read_seed_file, get_training_sets, filter_seeds, ensure_unicode
)
from nltk.stem import WordNetLemmatizer



def make_classifier(
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
	Convenience method to create a RelationalNounClassifier using
	default seed data and feature data
	'''
	positive_seeds, negative_seeds = get_training_sets()
	features = load_features()
	dictionary = get_dictionary(features)
	positive_seeds = filter_seeds(positive_seeds, dictionary)
	negative_seeds = filter_seeds(negative_seeds, dictionary)

	return RelationalNounClassifier(
		features, dictionary, positive_seeds, negative_seeds,
		kind=kind, 
		on_unk=on_unk,

		# SVM options
		syntactic_similarity=syntactic_similarity,
		semantic_similarity=semantic_similarity,
		syntactic_multiplier=syntactic_multiplier,
		semantic_multiplier=semantic_multiplier,

		# KNN options
		k=k,
	)


class RelationalNounClassifier(object):
	'''
	Class that wraps underlying classifiers and handles training, testing,
	and prediction logic that is specific to making a relational noun 
	classifier.
	'''

	def __init__(
		self,
		features,
		dictionary,
		positive_seeds,
		negative_seeds,
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
		on_unk [-1 || any]: (Stands for "on unknown token").  This
			controls behavior when a prediction is requested for a token 
			that did not appear in the dictionary:
				* -1 -- raise ValueError
				* anything else -- return that as the predicted class
		'''

		# Register parameters locally
		self.features = features
		self.dictionary = dictionary
		self.positive_seeds = positive_seeds
		self.negative_seeds = negative_seeds
		self.kind = kind
		self.on_unk = on_unk

		# Register SVM options
		self.syntactic_similarity = syntactic_similarity 
		self.semantic_similarity = semantic_similarity  
		self.syntactic_multiplier = syntactic_multiplier 
		self.semantic_multiplier = semantic_multiplier  

		# Register KNN options
		self.k = k

		# Build and train the classifier, according to the `kind`
		if kind == 'svm':
			self.classifier = self.make_svm_classifier()
		if kind == 'knn':
			self.classifier = self.make_knn_classifier()

		# Get a lemmatizer, which is used on words when doing prediction
		self.lemmatizer = WordNetLemmatizer()


	def make_knn_classifier(self):
		'''
		Make a kNN classifier
		'''
		X, Y = self.make_training_set()
		mydist = bind_dist(self.features, self.dictionary)
		classifier = KNN(metric=mydist, k=k)
		classifier.fit(X,Y)
		return classifier


	def make_svm_classifier(self):
		'''
		Make an SVM classifier
		'''
		X, Y = self.make_training_set()
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
		ids, lemmas = self.convert_tokens(tokens)
		return self.handle_unk(self.predict_id, ids, lemmas)


	def convert_tokens(self, tokens):
		# Accept a single lemma being passed -- put it in a list
		if isinstance(tokens, basestring):
			tokens = [tokens]
		# Ensure tokens are unicode, lemmatize, and lowercased
		lemmas = [
			self.lemmatizer.lemmatize(ensure_unicode(token).lower())
			for token in tokens
		]

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


def load_features(path=FEATURES_PATH):
	return json.loads(open(path).read())



def get_dictionary(features):
	counts = Counter({token:features[token]['count'] for token in features})
	dictionary = UnigramDictionary(on_unk=SILENT)
	dictionary.update(counts.elements())
	return dictionary



