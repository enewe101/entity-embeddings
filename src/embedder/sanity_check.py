import os
import numpy as np
from relation2vec_embedder import Relation2VecEmbedder
from dataset_reader import Relation2VecDatasetReader as DatasetReader
from SETTINGS import DATA_DIR, CORPUS_DIR

RELATION_FILES = ['senators-states.txt', 'countries-capitals.txt']
LOAD_DICTIONARY_DIR = os.path.join(DATA_DIR, 'dictionaries')
MIN_FREQUENCY = 20


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


def read_pairs(fname):
	pairs = []
	for line in open(fname):

		# Skip blank lines
		if line.strip() == '':
			continue
		
		source, target = line.strip().split()
		pairs.append((source, target))

	return pairs


def cosine(embedding1, embedding2):
	dot = np.dot(embedding1, embedding2)
	n1 = np.linalg.norm(embedding1)
	n2 = np.linalg.norm(embedding1)

	cos = dot / (n1 * n2)
	return cos


def load(load_dictionary_dir, min_frequency, embeddings_dir):

	# load the minibatch generator.  Prune very rare tokens.
	print 'Loading and pruning dictionaries'
	reader = DatasetReader(
		load_dictionary_dir=load_dictionary_dir
	)

	if min_frequency is not None:
		reader.prune(min_frequency)

	# Make an embedder with the correct sizes
	print 'Making the embedder'
	embedder = Relation2VecEmbedder(
		entity_vocab_size=reader.get_entity_vocab(),
		context_vocab_size=reader.get_contexgt_vocab(),
	)
	print 'Loading previously trained embeddings'
	embedder.load(embeddings_dir)

	return embedder, reader


def read_entities(minibatcher):
	entities = []
	lengths = []
	for fname in RELATION_FILES:
		length = 0
		for line in open(fname):

			# Skip blank lines
			if line.strip() == '':
				continue

			length += 1
			e1, e2 = line.strip().split()
			e1_id, e2_id = minibatcher.entity_dictionary.get_ids([e1, e2])
			entities.append([e1_id, e2_id])

		lengths.append(length)
	return entities, lengths


def compare_embeddings(embedder, entities, lengths):
	embeddings = embedder.embed_relationship(entities)
	cosines = []
	for i in range(len(embeddings)):

		# Insert a horizontal divider
		if i == lengths[0]:
			cosines.append(['-']* len(embeddings))

		cosines_row = []
		cosines.append(cosines_row)
		for j in range(len(embeddings)):

			# Insert a vertical divider
			if j == lengths[0]:
				cosines_row.append('||')

			norm_product = (
				np.linalg.norm(embeddings[i]) 
				* np.linalg.norm(embeddings[j])
			)
			cosine = np.dot(embeddings[i], embeddings[j]) / norm_product
			cosines_row.append(cosine)

	return cosines


def run_tests(props):

	# Unpack arguments
	load_dictionary_dir = props['load_dictionary_dir']
	min_frequency = props['min_frequency']
	embeddings_dir = props['embeddings_dir']

	# Load the embedder and dataset reader
	embedder, reader = load(
		load_dictionary_dir, min_frequency, embeddings_dir
	)

	# Get a set of test entities from a curated set
	entity_types = get_types()

	# Look for regularity in the entity embeddings
	test_type_pairings(entity_types, reader, embedder)


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
	properties = {
		'load_dictionary_dir': LOAD_DICTIONARY_DIR,
		'min_frequency': MIN_FREQUENCY
	}
	command_line_properties = commandline2dict()
	properties.update(command_line_properties)

	# Verify that the embeddings dir is set
	if 'embeddings_dir' not in command_line_properties:
		raise ValueError(
			'You must specify a directory containing the embeddings to be '
			'analyzed'
		)

	# Run the tests on the embeddings
	run_tests(properties)

