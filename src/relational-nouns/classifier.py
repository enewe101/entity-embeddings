import random
import t4k
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import json
import sys
sys.path.append('..')
from word2vec import UnigramDictionary, UNK, SILENT
from collections import Counter, deque
from SETTINGS import DEPENDENCY_FEATURES_PATH, BASELINE_FEATURES_PATH
from kernels import bind_kernel, bind_dist
from utils import (
	read_seed_file, get_training_sets, filter_seeds, ensure_unicode,
	get_dictionary, get_features
)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as w


def make_classifier(
	kind='svm',	# available: 'svm', 'knn', 'wordnet', 'basic_syntax'
	on_unk=False,

	# SVM options
	syntactic_similarity='dep_tree',
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
	features = get_features()
	dictionary = get_dictionary()
	dictionary.prune(min_frequency=5)
	positive_seeds = filter_seeds(positive_seeds, dictionary)
	negative_seeds = filter_seeds(negative_seeds, dictionary)


	# The proposed most-performant classifier
	if kind == 'svm':
		return SvmNounClassifier(
			positive_seeds=positive_seeds,
			negative_seeds=negative_seeds,
			# SVM options
			dictionary=dictionary,
			features=features,
			on_unk=False,
			syntactic_similarity=syntactic_similarity,
			semantic_similarity=semantic_similarity,
			syntactic_multiplier=syntactic_multiplier,
			semantic_multiplier=semantic_multiplier,
		)


	# A runner up, using KNN as the learner
	elif kind == 'knn':
		return KnnNounClassifier(
			positive_seeds,
			negative_seeds,
			# KNN options
			features['dep_tree'],
			dictionary,
			on_unk=False,
			k=3
		)


	# Simple rule: returns true if query is hyponym of known relational noun
	elif kind == 'wordnet':
		return WordnetClassifier(positive_seeds, negative_seeds)


	# Classifier using basic syntax cues
	elif kind == 'basic_syntax':
		get_features_func = arm_get_basic_syntax_features(
			features['baseline'])
		return BalancedLogisticClassifier(
			positive_seeds,
			negative_seeds,
			get_features=get_features_func
		)

def balance_samples(populations, target='largest'):

	# A target of largest means adjust all samples to be as big as the
	# largest.  Figure out the largest sample size
	largest = 0
	if target == 'largest':
		target = max([len(pop) for pop in populations])

	print 'target:', target

	resampled_populations = []
	for pop in populations:

		# If the sample size is off target, adjust it
		if len(pop) != target:
			new_pop = resample(pop, target)

		# If the sample size is on target, we still want to make our own 
		# copy so we don't encourage side effects downstream
		else:
			new_pop = list(pop)

		resampled_populations.append(new_pop)

	return resampled_populations



def resample(population, sample_size):

	# If the sample needs to be smaller, just subsample without replacement
	if len(population) > sample_size:
		return random.sample(population, sample_size)

	# If the sample needs to be bigger, add additional samples with 
	# replacement
	else:

		# Use a list (we don't know what kind of iterable population is)
		population = list(population)

		# Add samples drawn randomly with replacement
		population.extend([
			random.choice(population) 
			for i in range(sample_size - len(population))
		])
		return population



def arm_get_basic_syntax_features(features):
	'''
	Bind the features dictionary to a function that retrieves basic
	syntax features when given a token.  Suitable to be passed in as
	the `get_features` function for a BalancedLogisticClassifier
	'''

	def get_features(lemma):

		if lemma not in features:
			return [0, 0]

		count = features[lemma]['count']
		f1 = (
			features[lemma]['nmod:of:NNX'] / float(count)
			if 'nmod:of:NNX' in features[lemma] else 0
		)
		f2 = (
			features[lemma]['nmod:poss'] / float(count)
			if 'nmod:poss' in features[lemma] else 0
		)
		return [f1, f2]

	return get_features



class BalancedLogisticClassifier(object):
	'''
	Wraps a logistic regression classifier, making it able to operate
	on an artificially balanced dataset, but is adjusts the decision
	function based on the true prior.

	Assumes that the features are available for objects to be 
	classified using the get_features function.
	'''

	def __init__(
		self, 
		positive_seeds,
		negative_seeds,
		get_features,
		prior=None,
		balance=True,
		do_adjust=True
	):
		'''
		Note that if the prevalence of positives and negatives in the 
		dataset is not representative of their prior probabilities, you
		need to explicitly specifiy the prior probability of positive
		examples.

		If no prior is given, it will be assumed that the prior is

			len(positive_seens) / float(
				len(positive_seeds) + len(negative_seeds))

		Logistic regression is sensitive to data imbalances.  A better
		model can be achieved by training on a balanced dataset, and then
		adjusting the model intercept based on the true class prevalences.
		if `balance` is True, this is done automatically.
		'''
		self.positive_seeds = positive_seeds
		self.negative_seeds = list(negative_seeds)
		self.get_features = get_features

		self.prior = prior
		self.balance = balance
		self.do_adjust = do_adjust

		# Calculate the prior as it exists in the data.  The meaning and
		# usefulness of this value will depend on the settings of `balance`
		# and `prior`
		total = float(len(positive_seeds) + len(negative_seeds))
		self.data_prior = len(positive_seeds) / total

		# If prior is not given, we assume that it is given by the data
		# prior
		if self.prior is None:
			self.prior = self.data_prior

		self.classifier = LogisticRegression(
			solver='newton-cg',
		)
		self.fit()


	def fit(self):
		X,Y = self.make_training_set()
		print 'balance:', sum(Y) / float(len(Y))
		self.classifier.fit(X,Y)

		# We'll need to adjust the model's intercept if it was not trained
		# on data distributed according to it's natural prior.  This
		# can happen if an explicit value for the prior was given or if
		# we have artificially balanced the dataset during training
		intercept_needs_adjustment = self.prior is not None or self.balance
		if intercept_needs_adjustment and self.do_adjust:
			self.adjust_intercept()


	def adjust_intercept(self):
		print 'adjusting...'

		# Determine what the apparent prior (in the training data) was
		if self.balance:
			training_prior = 0.5
		else:
			training_prior = self.data_prior

		# Determine the adjustment needed based on what the real prior is
		adjustment = -np.log(
			(1-self.prior)/self.prior
			* training_prior/(1 - training_prior)
		)

		# Make the adjustment
		self.classifier.intercept_ += adjustment


	def predict(self, tokens):
		tokens = maybe_list_wrap(tokens)
		lemmas = lemmatize_many(tokens)
		features = [self.get_features(lemma) for lemma in lemmas]
		return self.classifier.predict(features)


	def score(self, tokens):
		tokens = maybe_list_wrap(tokens)
		lemmas = lemmatize_many(tokens)
		features = [self.get_features(l) for l in lemmas]
		return self.classifier.predict_proba(features)[:,1]


	def make_training_set(self):

		if self.balance:
			print 'balancing...'
			positive_seeds, negative_seeds = balance_samples(
				[self.positive_seeds, self.negative_seeds]
			)
		else:
			positive_seeds = self.positive_seeds
			negative_seeds = self.negative_seeds

		X = np.array(
			[self.get_features(s) for s in positive_seeds]
			+ [self.get_features(s) for s in negative_seeds]
		)

		Y = np.array(
			[1]*len(positive_seeds) + [0]*len(negative_seeds)
		)

		return X, Y



class BasicSyntaxNounClassifier(object):
	'''
	Uses a logistic regression classifier to classify nouns based on basic
	syntax statistics.
	'''

	def __init__(
		self, 
		positive_seeds,
		negative_seeds,
		features,
		dictionary,
		prior=None,
		balance=True,
		do_adjust=True
	):
		'''
		Note that if the prevalence of positives and negatives in the 
		dataset is not representative of their prior probabilities, you
		need to explicitly specifiy the prior probability of positive
		examples.

		If no prior is given, it will be assumed that the prior is

			len(positive_seens) / float(
				len(positive_seeds) + len(negative_seeds))

		Logistic regression is sensitive to data imbalances.  A better
		model can be achieved by training on a balanced dataset, and then
		adjusting the model intercept based on the true class prevalences.
		if `balance` is True, this is done automatically.
		'''
		self.positive_seeds = positive_seeds
		self.negative_seeds = list(negative_seeds)
		self.features = features
		self.dictionary = dictionary
		self.prior = prior
		self.balance = balance
		self.do_adjust = do_adjust

		# Calculate the prior as it exists in the data.  The meaning and
		# usefulness of this value will depend on the settings of `balance`
		# and `prior`
		total = float(len(positive_seeds) + len(negative_seeds))
		self.data_prior = len(positive_seeds) / total

		# If prior is not given, we assume that it is given by the data
		# prior
		if self.prior is None:
			self.prior = self.data_prior

		self.classifier = LogisticRegression(
			solver='newton-cg',
		)
		self.fit()


	def fit(self):
		X,Y = self.make_training_set()
		print 'balance:', sum(Y) / float(len(Y))
		self.classifier.fit(X,Y)

		# We'll need to adjust the model's intercept if it was not trained
		# on data distributed according to it's natural prior.  This
		# can happen if an explicit value for the prior was given or if
		# we have artificially balanced the dataset during training
		intercept_needs_adjustment = self.prior is not None or self.balance
		if intercept_needs_adjustment and self.do_adjust:
			self.adjust_intercept()


	def adjust_intercept(self):
		print 'adjusting...'

		# Determine what the apparent prior (in the training data) was
		if self.balance:
			training_prior = 0.5
		else:
			training_prior = self.data_prior

		# Determine the adjustment needed based on what the real prior is
		adjustment = -np.log(
			(1-self.prior)/self.prior
			* training_prior/(1 - training_prior)
		)

		# Make the adjustment
		self.classifier.intercept_ += adjustment


	def predict(self, tokens):
		tokens = maybe_list_wrap(tokens)
		lemmas = lemmatize_many(tokens)
		features = [self.get_features(lemma) for lemma in lemmas]
		return self.classifier.predict(features)


	def score(self, tokens):
		tokens = maybe_list_wrap(tokens)
		lemmas = lemmatize_many(tokens)
		features = [self.get_features(l) for l in lemmas]
		return self.classifier.predict_proba(features)[:,1]

	def get_features(self, lemma):

		if lemma not in self.features:
			return [0, 0]

		count = self.features[lemma]['count']
		f1 = (
			self.features[lemma]['nmod:of:NNX'] / float(count)
			if 'nmod:of:NNX' in self.features[lemma] else 0
		)
		f2 = (
			self.features[lemma]['nmod:poss'] / float(count)
			if 'nmod:poss' in self.features[lemma] else 0
		)
		return [f1, f2]


	def make_training_set(self):

		if self.balance:
			print 'balancing...'
			positive_seeds, negative_seeds = balance_samples(
				[self.positive_seeds, self.negative_seeds]
			)
		else:
			positive_seeds = self.positive_seeds
			negative_seeds = self.negative_seeds

		X = np.array(
			[self.get_features(s) for s in positive_seeds]
			+ [self.get_features(s) for s in negative_seeds]
		)

		Y = np.array(
			[1]*len(positive_seeds) + [0]*len(negative_seeds)
		)

		return X, Y




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
		dictionary,
		features=None,	# Must be provided if syntactic_similarity is True
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
			self.dictionary, self.features, 
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


#def load_features(path=DEPENDENCY_FEATURES_PATH):
#	return json.loads(open(path).read())



#def get_dictionary(features):
#	counts = Counter({token:features[token]['count'] for token in features})
#	dictionary = UnigramDictionary(on_unk=SILENT)
#	dictionary.update(counts.elements())
#	return dictionary

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
