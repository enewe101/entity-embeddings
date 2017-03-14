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


def read_word(line):
	word, typestring = line.strip().split()
	original_typestring = typestring
	partial = False

	# Is the first character an "m", which stands for "mainly"
	if typestring.startswith('m'):
		partial = True
		typestring = typestring[1:]

	# Is the next character an "n" or a "p" (negative or positive)
	if typestring.startswith('p'):
		is_relational = True
	elif typestring.startswith('n'):
		is_relational = False
	else:
		raise ValueError(
			'No relational indicator: %s.' % original_typestring)

	typestring = typestring[1:]
	if len(typestring) == 0:
		subtype = None
	elif len(typestring) == 1:
		if typestring.startswith('b'):
			subtype = 'body-part'
		elif typestring.startswith('p'):
			subtype = 'portion'
		elif typestring.startswith('j'):
			subtype = 'adjectival'
		elif typestring.startswith('v'):
			subtype = 'deverbal'
		elif typestring.startswith('a'):
			subtype = 'aspectual'
		elif typestring.startswith('f'):
			subtype = 'functional'
		elif typestring.startswith('r'):
			subtype = 'link'
		else:
			raise ValueError(
				'Unrecognized noun subtype: %s.' % original_typestring
			)

	else:
		raise ValueError(
				'Trailing characters on typestring: %s.' 
				% original_typestring
			)

	return {
		'word':word,
		'is_relational':is_relational,
		'subtype':subtype
	}


def read_typed_seed_file(path):
	return [
		read_word(line) for line in open(path) 
		if not line.strip().startswith('#')
		and len(line.strip()) > 0
	]


def read_seed_file(path):
	return set([
		s.strip() for s in open(path) 
		if not s.strip().startswith('#') and len(s.strip())
	])


def get_seed_set(path, with_aspectual=False):
	'''
	Get a set of positive (relational) words and a set of negative 
	(non-relational) words, to be used as a training set
	'''
	if with_aspectual:
		positives = (
			read_seed_file(os.path.join(path, 'positive.txt'))
			| read_seed_file(os.path.join(path, 'aspectual.txt'))
		)
		negatives = (
			read_seed_file(os.path.join(path, 'negative.txt'))
			| read_seed_file(os.path.join(path, 'role.txt'))
			| read_seed_file(os.path.join(path, 'relationship.txt'))
			| read_seed_file(os.path.join(path, 'verb-derived.txt'))
		)
	else:
		positives = (
			read_seed_file(os.path.join(path, 'positive.txt'))
		)
		negatives = (
			read_seed_file(os.path.join(path, 'negative.txt'))
			| read_seed_file(os.path.join(path, 'role.txt'))
			| read_seed_file(os.path.join(path, 'relationship.txt'))
			| read_seed_file(os.path.join(path, 'verb-derived.txt'))
			| read_seed_file(os.path.join(path, 'aspectual.txt'))
		)
	return positives, negatives


def get_dictionary(path):
	dictionary = UnigramDictionary()
	dictionary.load(path)
	return dictionary


def get_train_sets():
	'''
	Get a set of positive (relational) words and a set of negative 
	(non-relational) words, to be used as a training set
	'''
	positives, negatives = get_seed_set(TRAIN_DIR)
	return positives, negatives


def get_test_sets():
	'''
	Get a set of positive (relational) words and a set of negative 
	(non-relational) words, to be used as a test set
	'''
	positives, negatives = get_seed_set(TEST_DIR)
	return positives, negatives


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
