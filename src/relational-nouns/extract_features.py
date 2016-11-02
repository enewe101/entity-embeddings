#!/usr/bin/env python
import tarfile
from word2vec import UnigramDictionary
import json
from iterable_queue import IterableQueue
from multiprocessing import Process
import time
import os
from corenlp_xml_reader import AnnotatedText, Token, Sentence
import sys
sys.path.append('..')
from subprocess import check_output
from collections import defaultdict, Counter, deque
from SETTINGS import (
	GIGAWORD_DIR, DICTIONARY_DIR, DATA_DIR, DEPENDENCY_FEATURES_PATH,
	BASELINE_FEATURES_PATH, HAND_PICKED_FEATURES_PATH,
	RELATIONAL_NOUN_FEATURES_DIR
)
from nltk.corpus import wordnet

NUM_ARTICLE_LOADING_PROCESSES = 12
GAZETTEER_DIR = os.path.join(DATA_DIR, 'gazetteers')
GAZETTEER_FILES = [
	'country', 'city', 'us-state', 'continent', 'subcontinent'
]
	

def load_gazetteers():
	# Read a the gazetteer files, and save each as a set.
	# We have place names and demonyms for coutries, cities, etc.
	gazetteer = {'names': set(), 'demonyms': set()}
	for gazetteer_fname_prefix in GAZETTEER_FILES:
		for pos in ['names', 'demonyms']:

			# Work out the path
			gazetteer_type = gazetteer_fname_prefix + '-' + pos
			gazetteer_fname = gazetteer_type + '.txt'
			gazetteer_path = os.path.join(GAZETTEER_DIR, gazetteer_fname)

			# Read the file into a set
			gazetteer[gazetteer_type] = set([
				line.strip() for line in open(gazetteer_path)
			])

			# Pool all names and separately pool all demonyms
			gazetteer[pos] |= gazetteer[gazetteer_type]

	return gazetteer

# Keep a global gazetteer for easy access
GAZETTEERS = load_gazetteers()


def extract_and_save_features(
	limit=100,
	dependency_features_path=DEPENDENCY_FEATURES_PATH,
	dictionary_dir=DICTIONARY_DIR,
	baseline_features_path=BASELINE_FEATURES_PATH,
	hand_picked_features_path=HAND_PICKED_FEATURES_PATH
):
	# Extract all the features and a dictionary
	extract = extract_all_features(limit)

	# Save each of the features
	open(dependency_features_path, 'w').write(json.dumps(
		extract['dep_tree_features']))
	open(baseline_features_path, 'w').write(json.dumps(
		extract['baseline_features']))
	open(hand_picked_features_path, 'w').write(json.dumps(
		extract['hand_picked_features']))

	# Save the dictionary
	extract['dictionary'].save(dictionary_dir)


def extract_and_save_features_from_archive(archive_path):
	extract = extract_all_features_from_archive(archive_path)
	this_archive = os.path.basename(archive_path)[:-4]

	# Save each of the features
	dependency_features_path = os.path.join(
		RELATIONAL_NOUN_FEATURES_DIR, this_archive, 'dependency.json')
	open(dependency_features_path, 'w').write(json.dumps(
		extract['dep_tree_features']))

	baseline_features_path = os.path.join(
		RELATIONAL_NOUN_FEATURES_DIR, this_archive, 'baseline.json')
	open(baseline_features_path, 'w').write(json.dumps(
		extract['baseline_features']))

	hand_picked_features_path = os.path.join(
		RELATIONAL_NOUN_FEATURES_DIR, this_archive, 'hand-picked.json')
	open(hand_picked_features_path, 'w').write(json.dumps(
		extract['hand_picked_features']))

	# Save the dictionary
	dictionary_dir = os.path.join(
		RELATIONAL_NOUN_FEATURES_DIR, this_archive, 
		'lemmatized-noun-dictionary'
	)
	extract['dictionary'].save(dictionary_dir)


def extract_all_features_from_archive(archive_path):

	start = time.time()

	# First, make an iterable queue.  Extract all the corenlp files from the
	# archive and load them onto it
	fnames_q = IterableQueue()
	fnames_producer = fnames_q.get_producer()
	archive = tarfile.open(archive_path)
	for member in archive:

		# Extract the contents of the corenlp files, putting the text
		# for each file directly onto the queue
		if member.name.endswith('xml'):
			fnames_producer.put((
				member.name,
				archive.extractfile(member).read()
			))

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
			args=(fnames_consumer, features_producer, 'content')
		)
		process.start()

	# Close the iterable queues
	fnames_q.close()
	features_q.close()

	# Accumulate the results.  This blocks until workers are finished
	dep_tree_features = defaultdict(Counter)
	baseline_features = defaultdict(Counter)
	hand_picked_features = defaultdict(Counter)
	dictionary = UnigramDictionary()

	for extract in features_consumer:
		dictionary.add_dictionary(extract['dictionary'])
		for key in extract['dep_tree_features']:
			dep_tree_features[key] += extract['dep_tree_features'][key]
		for key in extract['baseline_features']:
			baseline_features[key] += extract['baseline_features'][key]
		for key in extract['hand_picked_features']:
			hand_picked_features[key] += (
				extract['hand_picked_features'][key])

	elapsed = time.time() - start
	print 'elapsed', elapsed

	return {
		'dep_tree_features':dep_tree_features, 
		'baseline_features': baseline_features, 
		'hand_picked_features': hand_picked_features,
		'dictionary': dictionary
	}


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
	dep_tree_features = defaultdict(Counter)
	baseline_features = defaultdict(Counter)
	hand_picked_features = defaultdict(Counter)
	dictionary = UnigramDictionary()

	for extract in features_consumer:
		dictionary.add_dictionary(extract['dictionary'])
		for key in extract['dep_tree_features']:
			dep_tree_features[key] += extract['dep_tree_features'][key]
		for key in extract['baseline_features']:
			baseline_features[key] += extract['baseline_features'][key]
		for key in extract['hand_picked_features']:
			hand_picked_features[key] += (
				extract['hand_picked_features'][key])

	elapsed = time.time() - start
	print 'elapsed', elapsed

	return {
		'dep_tree_features':dep_tree_features, 
		'baseline_features': baseline_features, 
		'hand_picked_features': hand_picked_features,
		'dictionary': dictionary
	}


def extract_features_from_articles(
	files_consumer, features_producer, has_content='True'
):
	'''
	Extracts features from articles on `files_consumer`, and puts 
	featres onto `features_producer`.  If `has_content` is 'true', then
	each item is a tuple containing path and a string representing the
	file contents.  Otherwise, only the path is provided, and the file
	will be opened and read here.
	'''

	dep_tree_features = defaultdict(Counter)
	baseline_features =  defaultdict(Counter)
	hand_picked_features = defaultdict(Counter)
	dictionary = UnigramDictionary()

	# Read articles named in files_consumer, accumulate features from them
	for item in files_consumer:

		if has_content:
			fname, content = item
		else:
			fname = item
			content = open(fname).read()

		print 'processing', fname, '...'
		# Get features from this article
		article = AnnotatedText(content)
		extract = extract_features_from_article(article)

		# Accumulate the features
		dictionary.add_dictionary(extract['dictionary'])
		for key in extract['dep_tree_features']:
			dep_tree_features[key] += extract['dep_tree_features'][key]
		for key in extract['baseline_features']:
			baseline_features[key] += extract['baseline_features'][key]
		for key in extract['hand_picked_features']:
			hand_picked_features[key] += (
				extract['hand_picked_features'][key])

	# Put the accumulated features onto the producer queue then close it
	features_producer.put({
		'dep_tree_features': dep_tree_features,
		'baseline_features': baseline_features,
		'hand_picked_features': hand_picked_features,
		'dictionary': dictionary
	})
	features_producer.close()


#TODO: Add a call to produce baseline features in this function
def extract_features_from_article(article):

	dep_tree_features = defaultdict(Counter)
	dictionary = UnigramDictionary()
	baseline_features = defaultdict(Counter)
	hand_picked_features = defaultdict(Counter)

	for sentence in article.sentences:
		for token in sentence['tokens']:

			pos = token['pos']
			lemma = token['lemma']

			# We're only interested in common nouns
			if pos != 'NN' and pos != 'NNS':
				continue

			# Add features seen for this instance.  We use the lemma as
			# the key around which to aggregate features for the same word
			dep_tree_features[lemma] += get_dep_tree_features(token)
			baseline_features[lemma] += get_baseline_features(token)
			hand_picked_features[lemma] += get_hand_picked_features(token)
			dictionary.add(lemma)

	return {
		'dep_tree_features':dep_tree_features, 
		'baseline_features': baseline_features, 
		'hand_picked_features': hand_picked_features,
		'dictionary': dictionary
	}


def get_baseline_features(token):
	'''
	Looks for specific syntactic relationships that are indicative of
	relational nouns.  Keeps track of number of occurrences of such
	relationships and total number of occurrences of the token.
	'''
	baseline_features = Counter({'count':1})

	# Record all parent signatures
	for relation, child_token in token['children']:
		if relation == 'nmod:of' and child_token['pos'].startswith('NN'):
			baseline_features['nmod:of:NNX'] += 1
		if relation == 'nmod:poss':
			baseline_features['nmod:poss'] += 1

	return baseline_features


def get_or_none(dictionary, key):
	try:
		return dictionary[key]
	except KeyError:
		return None


# IDEA: look at mention property of tokens to see if it in a coref chain
# 	with named entity, demonym, placename, etc.
#
# IDEA: ability to infer in cases where conj:and e.g. '9fd7d0a5189bb351 : 2'
def get_hand_picked_features(token):
	'''
	Looks for a specific set of syntactic relationships that are
	indicative of relational nouns.  It's a lot more thourough than 
	the baseline features.
	'''
	features = Counter({'count':1})

	# Get features that are in the same noun phrase as the token
	NP_tokens = get_constituent_tokens(token['c_parent'], recursive=False)
	focal_idx = NP_tokens.index(token)
	for i, sibling_token in enumerate(NP_tokens):

		# Don't consider the token itself
		if sibling_token is token:
			continue

		# Get the position of this token relative to the focal token
		rel_idx = i - focal_idx

		# Note the sibling's POS
		key = 'sibling(%d):pos(%s)' % (rel_idx, sibling_token['pos'])
		features[key] += 1

		# Note if the sibling is a noun of some kind
		if sibling_token['pos'].startswith('NN'):
			features['sibling(%d):pos(NNX)' % rel_idx] += 1

		# Note the sibling's named entity type
		key = 'sibling(%d):ner(%s)' % (rel_idx, sibling_token['ner'])
		features[key] += 1

		# Note if the sibling is a named entity of any type
		if sibling_token['ner'] is not None:
			features['sibling(%d):ner(x)' % rel_idx] += 1

		# Note if the sibling is a demonym
		if sibling_token['word'] in GAZETTEERS['demonyms']:
			features['sibling(%d):demonym' % rel_idx] += 1

		# Note if the sibling is a place name
		if sibling_token['word'] in GAZETTEERS['names']:
			features['sibling(%d):place-name' % rel_idx] += 1

	# Note if the noun is plural
	if token['pos'] == 'NNS':
		features['plural'] += 1

	# Detect construction "is a <noun> of"
	children = {
		relation:child for relation, child 
		in reversed(token['children'])
	}
	cop = children['cop']['lemma'] if 'cop' in children else None
	det = children['det']['lemma'] if 'det' in children else None
	nmod = (
		'of' if 'nmod:of' in children 
		else 'to' if 'nmod:to' in children 
		else None
	)
	poss = 'nmod:poss' in children 

	# In this section we accumulate various combinations of having
	# a copula, a prepositional phrase, a posessive, and a determiner.
	if nmod:
		features['<noun>-prp'] += 1
		features['<noun>-%s' % nmod] += 1

	if poss:
		features['poss-<noun>'] += 1

	if cop and nmod:
		features['is-<noun>-prp'] += 1
		features['is-<noun>-%s' % nmod] += 1
		if det:
			features['is-%s-<noun>-prp' % det] += 1

	if det and nmod:
		features['%s-<noun>-prp' % det] += 1
		features['%s-<noun>-%s' % (det, nmod)] += 1

	if cop and poss:
		features['is-poss-<noun>'] += 1

	if det and poss:
		features['%s-poss-<noun>' % det] += 1

	if det and not nmod and not poss:
		features['%s-<noun>' % det] += 1
	
	if cop and det and poss:
		features['is-det-poss-<noun>'] += 1

	if cop and det and nmod:
		features['is-det-<noun>-prp'] += 1

	# Next we consider whether the propositional phrase has a named
	# entity, demonym, or place name in it
	if nmod:

		for prep_type in ['of', 'to', 'for']:

			# See if there is a prepositional noun phrase of this type, and
			# get it's head.  If not, continue to the next type
			NP_head = get_first_matching_child(token, 'nmod:%s' % prep_type)
			if NP_head is None:
				continue

			# Get all the tokens that are part of the noun phrase
			NP_constituent = NP_head['c_parent']
			NP_tokens = get_constituent_tokens(NP_constituent)

			# Add feature counts for ner types in the NP tokens
			ner_types = set([t['ner'] for t in NP_tokens])
			for ner_type in ner_types:
				features['prp(%s)-ner(%s)' % (prep_type, ner_type)] += 1

			# Add feature counts for demonyms 
			lemmas = [t['lemma'] for t in NP_tokens]
			if any([l in GAZETTEERS['demonyms'] for l in lemmas]):
				features['prp(%s)-demonyms' % prep_type] += 1

			# Add feature counts for place names 
			if any([l in GAZETTEERS['names'] for l in lemmas]):
				features['prp(%s)-place' % prep_type] += 1 
	
	# Next we consider whether the posessor noun phrase has a named
	# entity, demonym, or place name in it
	if poss:
		NP_head = get_first_matching_child(token, 'nmod:poss')
		NP_constituent = NP_head['c_parent']
		NP_tokens = get_constituent_tokens(NP_constituent)

		# Add feature counts for ner types in the NP tokens
		ner_types = set([t['ner'] for t in NP_tokens])
		for ner_type in ner_types:
			features['poss-ner(%s)' % ner_type] += 1

		# Add feature counts for demonyms 
		lemmas = [t['lemma'] for t in NP_tokens]
		if any([l in GAZETTEERS['demonyms'] for l in lemmas]):
			features['poss-demonyms'] += 1

		# Add feature counts for place names 
		if any([l in GAZETTEERS['names'] for l in lemmas]):
			features['poss-place'] += 1 

	return features


def get_first_matching_child(token, relation):
	'''
	Finds the first child of `token` in the dependency tree related by
	`relation`.
	'''
	try:
		return [
			child for rel, child in token['children'] if rel == relation
		][0]

	except IndexError:
		return None


def get_constituent_tokens(constituent, recursive=True):

	tokens = []
	for child in constituent['c_children']:
		if isinstance(child, Token):
			tokens.append(child)
		elif recursive:
			tokens.extend(get_constituent_tokens(child, recursive))

	return tokens
	

def get_dep_tree_features(token):
	'''
	Extracts a set of features based on the dependency tree relations
	for the given token.  Each feature describes one dependency tree 
	relation in terms of a "signature".  The signature of a dependency
	tree relation consists of whether it is a parent or child, what the 
	relation type is (e.g. nsubj, prep:for, etc), and what the pos of the 
	target is.
	'''
	dep_tree_features = Counter({'count':1})

	# Record all parent signatures
	for relation, token in token['parents']:
		signature = '%s:%s:%s' % ('parent', relation, token['pos'])
		#signature = '%s:%s:%s' % ('parent', relation, token['ner'])
		dep_tree_features[signature] += 1

	# Record all child signatures
	for relation, token in token['children']:
		signature = '%s:%s:%s' % ('child', relation, token['pos'])
		#signature = '%s:%s:%s' % ('child', relation, token['ner'])
		dep_tree_features[signature] += 1

	return dep_tree_features

	
def get_fnames():

	# Get the absolute path to all the corenlp files under batch '9fd'
	path = os.path.join(GIGAWORD_DIR, 'data', '9fd', 'CoreNLP')
	fnames = check_output(['ls %s' % path], shell=True).split()
	fnames = [os.path.join(path, fname) for fname in fnames]

	return fnames


if __name__ == '__main__':

	# Accept the number of articles to process for feature extraction
	limit = int(sys.argv[1])

	# Extract and save the features
	extract_and_save_features(limit)

