import json
from sklearn import svm
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


NUM_ARTICLE_LOADING_PROCESSES = 12
	

def get_seeds():
	positive_seeds = open(RELATIONAL_NOUN_SEEDS_PATH).read().split()
	positive_seeds = set([
		s for s in positive_seeds if not s.startswith('?')])

	negative_seeds = open(NONRELATIONAL_NOUN_SEEDS_PATH).read().split()
	negative_seeds = set([
		s for s in negative_seeds if not s.startswith('?')])

	return positive_seeds, negative_seeds


def get_fnames():

	# Get the absolute path to all the corenlp files under batch '9fd'
	path = '/home/ndg/dataset/gigaword-corenlp/data/9fd/CoreNLP'
	fnames = check_output(['ls %s' % path], shell=True).split()
	fnames = [os.path.join(path, fname) for fname in fnames]

	return fnames


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


def load_features():
	return json.loads(open(FEATURES_PATH).read())


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
	print len(features)
	dictionary = get_dictionary(features)
	positive_seeds = filter_seeds(positive_seeds, dictionary)
	negative_seeds = filter_seeds(negative_seeds, dictionary)
	classifier = make_classifier(
		features, dictionary, positive_seeds, negative_seeds)
	return classifier, dictionary


def make_classifier(features, dictionary, positive_seeds, negative_seeds):

	# Make the training set.  We're going to use a custom kernel function
	# so each "row" in the training set has a single "feature" -- it's the
	# id which identifies the token.  This will let us lookup the 
	# non-numeric feature in the kernel function
	X = (
		[ [dictionary.get_id(s)] for s in positive_seeds]
		+ [ [dictionary.get_id(s)] for s in negative_seeds]
	)
	Y = [True] * len(positive_seeds) + [False] * len(negative_seeds)

	classifier = svm.SVC(kernel=bind_kernel(features, dictionary))
	classifier.fit(X,Y)

	return classifier


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

			# We're only interested in nouns
			if not token['pos'].startswith('N'):
				continue

			# Add features seen for this instance
			add_features = extract_features_from_token(token)
			features[token['word']] += add_features

			# We also keep count of the number of times a given token is
			# seen which helps with normalization of the feature dict
			features[token['word']]['count'] += 1

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

	



