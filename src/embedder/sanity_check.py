import os
import numpy as np
from relation2vec_embedder import Relation2VecEmbedder
from minibatcher import Relation2VecMinibatcher
from SETTINGS import DATA_DIR, CORPUS_DIR
SAVEDIR = os.path.join(DATA_DIR, 'relation2vec')
MIN_FREQUENCY = 20
RELATION_FILES = ['senators-states.txt', 'countries-capitals.txt']


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


def test_type_pairings(named_types, minibatcher, embedder):

	embeddings = embedder.get_param_values()[0]

	for i in range(len(named_types)):
		for j in range(i, len(named_types)):
			types1, examples1 = named_types[i]
			types2, examples2 = named_types[j]

			example_ids1 = minibatcher.entity_dictionary.get_ids(examples1)
			example_ids2 = minibatcher.entity_dictionary.get_ids(examples2)

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


def load():
	# load the minibatch generator.  Prune very rare tokens.
	print 'Loading and pruning dictionaries'
	minibatcher = Relation2VecMinibatcher()
	minibatcher.load(SAVEDIR)
	minibatcher.prune(min_frequency=MIN_FREQUENCY)

	# Make an embedder with the correct sizes
	print 'Making the embedder'
	embedder = Relation2VecEmbedder(
		entity_vocab_size=len(minibatcher.entity_dictionary),
		context_vocab_size=len(minibatcher.context_dictionary),
	)
	print 'Loading previously trained embeddings'
	embeddings_filename = os.path.join(SAVEDIR, 'embeddings.npz')
	embedder.load(embeddings_filename)

	return embedder, minibatcher


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


