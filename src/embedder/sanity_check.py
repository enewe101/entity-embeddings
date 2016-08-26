#!/usr/bin/env python

import os
import sys
sys.path.append('..')

import json
import numpy as np
from relation2vec_embedder import Relation2VecEmbedder
from dataset_reader import Relation2VecDatasetReader as DatasetReader
from SETTINGS import DATA_DIR

RELATION_FILES = ['senators-states.txt', 'countries-capitals.txt']
LOAD_DICTIONARY_DIR = os.path.join(DATA_DIR, 'dictionaries')
MIN_QUERY_FREQUENCY = 30
MIN_CONTEXT_FREQUENCY = 30


def get_types():

	senators_states = read_pairs('senators-states.txt')
	senators = [s[0] for s in senators_states]
	states = [s[1] for s in senators_states]

	countries_capitals = read_pairs('countries-capitals.txt')
	countries = [c[0] for c in countries_capitals]
	capitals = [c[1] for c in countries_capitals]

	companies_countries = read_pairs('companies-countries.txt')
	companies = [c[0] for c in companies_countries]

	return [
		('senators', senators),
		('states', states),
		('countries', countries), 
		('capitals', capitals), 
		('companies', companies)
	]


def read_pairs(fname):
	pairs = []
	for line in open(fname):

		# Skip blank lines
		if line.strip() == '':
			continue
		
		source, target = line.strip().split()
		pairs.append((source, target))

	return pairs


def test_type_pairings(named_types, reader, embedder):
	'''
	Compares the entity embeddings of different entity types.  For example
	compares the embeddings of senators to embeddings of states.  Also
	compares embeddings among the same entity types (so compares embeddings
	of senators to senators).
	
	Embeddings of the same type should be more similar if the entity
	embeddings are encoding information that characterizes the entity types
	'''

	# Get the entity embeddings out of the embedder
	embeddings = embedder.get_param_values()[0]

	# Iterate over entity types
	for i in range(len(named_types)):

		# Iterate over entity types
		for j in range(i, len(named_types)):

			# unpack the type name and list of examples for each type.
			# Examples are listed as plain tokens
			types1, examples1 = named_types[i]
			types2, examples2 = named_types[j]

			# Convert examples to ids, which serve as indices into the 
			# embeddings parameters
			example_ids1 = reader.entity_dictionary.get_ids(examples1)
			example_ids2 = reader.entity_dictionary.get_ids(examples2)

			example_emb1 = [embeddings[e] for e in example_ids1]
			example_emb2 = [embeddings[e] for e in example_ids2]
			match_score = average_pairwise_match(example_emb1, example_emb2)

			print match_score, types1, types2


def average_pairwise_match(examples1, examples2):
	cosines = []
	for i in examples1:
		for j in examples2:
			cosines.append(cosine(i,j))

	return np.mean(cosines)



def load(
	load_dictionary_dir,
	min_query_frequency,
	min_context_frequency,
	embeddings_dir
):

	print 'loading embeddings from %s' % embeddings_dir
	print 'loading dictionaries from %s' % load_dictionary_dir
	# load the minibatch generator.  Prune very rare tokens.
	print 'Loading and pruning dictionaries'
	reader = DatasetReader(
		load_dictionary_dir=load_dictionary_dir
	)

	if min_query_frequency > 0 or min_context_frequency > 0:
		reader.prune(
			min_query_frequency=min_query_frequency,
			min_context_frequency=min_context_frequency
		)

	# Make an embedder with the correct sizes
	print 'Making the embedder'
	embedder = Relation2VecEmbedder(
		entity_vocab_size=reader.entity_vocab_size(),
		context_vocab_size=reader.context_vocab_size(),
	)
	print 'Loading previously trained embeddings'
	embedder.load(embeddings_dir)

	return embedder, reader


def read_entity_pairs(reader):
	'''
	Read in a series of entity-pairs from a few files.  Each file has a
	different type of pairing, reflecting a specific kind of relationship,
	e.g. senators and the states they represent, or countries and their
	capitals.  Along with the list of entity-pairs, retain a string 
	representing the type of relationship, e.g. "senator-state" for 
	human readability.
	'''

	pair_lists = []
	for fname in RELATION_FILES:

		# Get the pair type from the filename (remove it's extension)
		relation_type = '.'.join(fname.split('.')[:-1])

		# Get ready to accumulate all the pairs of that type
		pair_list = []
		pair_lists.append((relation_type, pair_list))

		# Iterate over the lines in the file, accumulating the entity pairs
		for line in open(fname):

			# Skip blank lines
			if line.strip() == '':
				continue

			e1, e2 = line.strip().split()
			e1_id, e2_id = reader.entity_dictionary.get_ids([e1, e2])
			pair_list.append([e1_id, e2_id])

	return pair_lists


# TODO: add a comparison between correct types and incorrect types.
def compare_relationship_embeddings(embedder, pair_lists):
	'''
	Compare the embeddings of "correct" relationships to one another.  This
	includes comparring "like" relationships (e.g. many senator-state 
	pairings amongst one another) as well as "unlike" relationships
	(e.g. senator-state pairings to country-capital pairings).  We expect
	that relationships of the same type should be more similar than 
	relationships of a different type.
	'''

	relation_embedding_lists = []
	for relation_type, pair_list in pair_lists:
		relation_embedding_list = embedder.embed_relationship(pair_list)
		relation_embedding_lists.append((
			relation_type, relation_embedding_list
		))

	cosines = []
	for relation_type_1, embedding_set_1 in relation_embedding_lists:
		for relation_type_2, embedding_set_2 in relation_embedding_lists:
			embeddings_set_similarity = average_pairwise_match(
				embedding_set_1, embedding_set_2
			)
			cosines.append((
				relation_type_1 + ':' + relation_type_2, 
				embeddings_set_similarity
			))

	return cosines


def get_noise_pair_list(pair_list, num_entities):
	noise_pair_list = []
	for e1, e2 in pair_list:
		if np.random.uniform() > 0.5:
			e1 = np.random.randint(num_entities)
		else:
			e2 = np.random.randint(num_entities)

		noise_pair_list.append((e1, e2))

	return noise_pair_list


# TODO: add a comparison between correct types and incorrect types.
def compare_correct_incorrect(embedder, pair_lists):
	'''
	Compare the embeddings of "correct" relationships to "incorrect"
	relationships, for various types of relationships.  We first read
	correct relationships from disk, and then generate incorrect ones
	by sampling entities randomly from the entity_dictionary.
	'''

	relation_embedding_lists = []
	for relation_type, pair_list in pair_lists:

		# Generate a set of "noise" relationships by replacing one of the
		# entities in each pair with a randomly selected one
		noise_pair_list = get_noise_pair_list(
			pair_list, embedder.entity_vocab_size
		)

		# Get the relationship embeddings for both the "correct" and "noise"
		# relationships
		relation_embedding_list = embedder.embed_relationship(pair_list)
		noise_relation_embedding_list = embedder.embed_relationship(
			noise_pair_list)

		# Append the "correct" and "noise" relaitonships along with the
		# type of relationship they are based on.
		relation_embedding_lists.append((
			relation_type,
			relation_embedding_list, 
			noise_relation_embedding_list
		))

	cosines = []
	for item in relation_embedding_lists:
		relation_type, correct_relations, noise_relations = item

		# Calculate cosine simliarity among correct relationships
		# and between correct and incorrect relationships
		correct_cosine = average_pairwise_match(
			correct_relations, correct_relations)
		noise_cosine = average_pairwise_match(
			noise_relations, noise_relations)

		cosines.append((relation_type, correct_cosine, noise_cosine))

	return cosines


def cosine(embedding1, embedding2):
	dot = np.dot(embedding1, embedding2)
	n1 = np.linalg.norm(embedding1)
	n2 = np.linalg.norm(embedding1)

	cos = dot / (n1 * n2)
	return cos



def run_tests(props):

	# Unpack arguments
	load_dictionary_dir = props['load_dictionary_dir']
	min_query_frequency = props['min_query_frequency']
	min_context_frequency = props['min_context_frequency']
	embeddings_dir = props['embeddings_dir']

	# Load the embedder and dataset reader
	embedder, reader = load(
		load_dictionary_dir, 
		min_query_frequency,
		min_context_frequency, 
		embeddings_dir
	)

	# Get a set of test entities from a curated set
	entity_types = get_types()

	# Look for regularity in the entity embeddings
	print 'Compare embeddings for various types of entities.'
	test_type_pairings(entity_types, reader, embedder)

	# Get sets of entity-pairs from curated set.  We will look at the
	# embeddings of their *relationships* next.
	pair_lists = read_entity_pairs(reader)
	
	# TODO: make a prettier way to print this
	# Compare embeddings of relationships
	print
	print 'Compare embeddings for various types of relationships.'
	cosines = compare_relationship_embeddings(embedder, pair_lists)
	for relation_type_pairing, cosine in cosines:
		print '%s\t%f' % (relation_type_pairing, cosine)

	print
	print 'Compare "correct" embeddings to "incorrect" embeddings.'
	cosines = compare_correct_incorrect(embedder, pair_lists)
	for relation_type, correct_cosine, incorrect_cosine in cosines:
		print (
			'%s\tcorrect: %f\tincorrect: %f' 
			% (relation_type, correct_cosine, incorrect_cosine)
		)


def commandline2dict():
	properties = {}
	for arg in sys.argv[1:]:
		key, val = arg.split('=')

		# Interpret numeric, list, and dictionary values properly, as
		# well as strings enquoted in properly escaped quotes
		try:
			properties[key] = json.loads(val)

		# It's cumbersome to always have to escape quotes around strings.
		# This caught exception interprets unenquoted tokens as strings
		except ValueError:
			properties[key] = val

	return properties


if __name__ == '__main__':

	# Merge default properties with properties specified on command line
	props = {
		'load_dictionary_dir': LOAD_DICTIONARY_DIR,
		'min_query_frequency': MIN_QUERY_FREQUENCY,
		'min_context_frequency': MIN_CONTEXT_FREQUENCY,
	}
	command_line_props = commandline2dict()
	props.update(command_line_props)

	# Verify that the embeddings dir is set
	if 'embeddings_dir' not in command_line_props:
		raise ValueError(
			'You must specify a directory containing the embeddings to be '
			'analyzed'
		)

	# Interpret the embeddings dir relative to DATA_DIR
	props['embeddings_dir'] = os.path.join(
		DATA_DIR, props['embeddings_dir'])

	# Run the tests on the embeddings
	run_tests(props)

