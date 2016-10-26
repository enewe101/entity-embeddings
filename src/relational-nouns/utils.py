import sys
sys.path.append('..')
from word2vec import UNK
from SETTINGS import (
	TRAIN_NEGATIVE_PATH, TRAIN_POSITIVE_PATH,
	TEST_NEGATIVE_PATH, TEST_POSITIVE_PATH
)


def read_seed_file(path):
	return set([
		s.strip() for s in open(path) 
		if not s.strip().startswith('#') and len(s.strip())
	])


def get_training_sets():
	'''
	Get a set of positive (relational) words and a set of negative 
	(non-relational) words, to be used as a training set
	'''
	positives = read_seed_file(TRAIN_POSITIVE_PATH)
	negatives = read_seed_file(TRAIN_NEGATIVE_PATH)
	return positives, negatives


def get_test_sets():
	'''
	Get a set of positive (relational) words and a set of negative 
	(non-relational) words, to be used as a training set
	'''
	positives = read_seed_file(TEST_POSITIVE_PATH)
	negatives = read_seed_file(TEST_NEGATIVE_PATH)
	return positives, negatives


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
