import os
import sys
sys.path.append('..')
import cjson
from word2vec import UNK, UnigramDictionary
from SETTINGS import (
	TRAIN_DIR, TEST_DIR,
	DICTIONARY_DIR, DEPENDENCY_FEATURES_PATH,
	BASELINE_FEATURES_PATH, HAND_PICKED_FEATURES_PATH
)


def read_seed_file(path):
	return set([
		s.strip() for s in open(path) 
		if not s.strip().startswith('#') and len(s.strip())
	])


def get_seed_set(path):
	'''
	Get a set of positive (relational) words and a set of negative 
	(non-relational) words, to be used as a training set
	'''
	positives = (
		read_seed_file(os.path.join(path, 'positives.txt'))
		| read_seed_file(os.path.join(path, 'aspectual.txt'))
	)
	negatives = (
		read_seed_file(os.path.join(path, 'negatives.txt'))
		| read_seed_file(os.path.join(path, 'role.txt'))
		| read_seed_file(os.path.join(path, 'relationship.txt'))
		| read_seed_file(os.path.join(path, 'verb-derived.txt'))
	)
	return positives, negatives


def get_train_sets():
	'''
	Get a set of positive (relational) words and a set of negative 
	(non-relational) words, to be used as a training set
	'''
	positives, negatives = get_seed_set(TRAIN_PATH)
	return positives, negatives


def get_test_sets():
	'''
	Get a set of positive (relational) words and a set of negative 
	(non-relational) words, to be used as a test set
	'''
	positives, negatives = get_seed_set(TEST_PATH)
	return positives, negatives


def get_dictionary(path=DICTIONARY_DIR):
	dictionary = UnigramDictionary()
	dictionary.load(path)
	return dictionary


def load_feature_file(path):
	return cjson.decode(open(path).read())


def get_features(path):
	return {
		'dep_tree': load_feature_file(os.path.join(
			path, 'dependency.json')),
		'baseline': load_feature_file(os.path.join(
			path, 'baseline.json')),
		'hand_picked': load_feature_file(os.path.join(
			path, 'hand_picked.json')),
		'dictionary': get_dictionary(os.path.join(
			path, 'lemmatized-noun-dictionary'))
	}


def filter_seeds(words, dictionary):
	'''
	Filters out any words in the list `words` that are not found in 
	the dictionary (and are hence mapped to UNK, which has id 0
	'''
	return [w for w in words if dictionary.get_id(w) != UNK]


def ensure_unicode(s):
	try:
		return s.decode('utf8')
	except UnicodeEncodeError:
		return s
