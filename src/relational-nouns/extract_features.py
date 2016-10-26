#!/usr/bin/env python
from word2vec import UnigramDictionary
import json
from iterable_queue import IterableQueue
from multiprocessing import Process
import time
import os
from corenlp_xml_reader import AnnotatedText
import sys
sys.path.append('..')
from subprocess import check_output
from collections import defaultdict, Counter
from SETTINGS import GIGAWORD_DIR, FEATURES_PATH, DICTIONARY_DIR
from nltk.corpus import wordnet


NUM_ARTICLE_LOADING_PROCESSES = 12
	

def extract_and_save_features(
	limit=100,
	features_path=FEATURES_PATH,
	dictionary_dir=DICTIONARY_DIR
):
	features, dictionary = extract_all_features(limit)
	open(features_path, 'w').write(json.dumps(features))
	dictionary.save(dictionary_dir)


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
	dictionary = UnigramDictionary()
	for add_features, add_dictionary in features_consumer:
		dictionary.add_dictionary(add_dictionary)
		for key in add_features:
			features[key] += add_features[key]

	elapsed = time.time() - start
	print 'elapsed', elapsed

	return features, dictionary


def extract_features_from_articles(fnames_consumer, features_producer):

	features = defaultdict(Counter)
	dictionary = UnigramDictionary()

	# Read articles named in fnames_consumer, accumulate features from them
	for fname in fnames_consumer:

		# Get features from this article
		article = AnnotatedText(open(fname).read())
		add_features, add_dictionary = extract_features_from_article(
			article)

		# Accumulate the features
		dictionary.add_dictionary(add_dictionary)
		for key in add_features:
			features[key] += add_features[key]

	# Put the accumulated features onto the producer queue then close it
	features_producer.put((features, dictionary))
	features_producer.close()


def extract_features_from_article(article):

	features = defaultdict(Counter)
	dictionary = UnigramDictionary()

	for sentence in article.sentences:
		for token in sentence['tokens']:

			# We're only interested in non-proper nouns
			if token['pos'] not in ('NN', 'NNS'):
				continue

			# Add features seen for this instance.  We use the lemma as
			# the key around which to aggregate features for the same word
			add_features = extract_features_from_token(token)
			features[token['lemma']] += add_features
			dictionary.add(token['lemma'])

			# We also keep count of the number of times a given token is
			# seen which helps with normalization of the feature dict
			features[token['lemma']]['count'] += 1

	return features, dictionary


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

	
def get_fnames():

	# Get the absolute path to all the corenlp files under batch '9fd'
	path = os.path.join(GIGAWORD_DIR, 'data', '9fd', 'CoreNLP')
	fnames = check_output(['ls %s' % path], shell=True).split()
	fnames = [os.path.join(path, fname) for fname in fnames]

	return fnames


if __name__ == '__main__':

	# Accept the number of articles to process for feature extraction
	limit = int(sys.argv[1])

	# Optionally accept a path in which to save features, or use default
	features_path = FEATURES_PATH
	try:
		features_path = sys.argv[2]
	except IndexError:
		pass

	# Extract and save the features
	extract_and_save_features(limit, features_path)

