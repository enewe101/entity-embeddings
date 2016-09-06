import sys
import numpy as np
from pair_dictionary import PairDictionary
from word2vec import Word2VecEmbedder
import theano

EMBEDDINGS_PATH = '/gs/project/eeg-641-aa/enewel3/entity-embeddings/data/ep-between-min1000-batch1000-learn0.01'
MIN_ENTITY_PAIR_FREQUENCY = 1000
PAIR_DICTIONARY_PATH = '/gs/project/eeg-641-aa/enewel3/entity-embeddings/data/ep-between-min1000-batch1000-learn0.01/query-dictionary'

NUM_BEST = 5
NUM_WORST = 10


def read_pair_set(fname):
	pairs = [
		['YAGO:%s' % e for e in p.split('\t')]
		for p in open(fname).read().strip().split('\n')
	]
	print pairs
	return pairs


def binary_test(
	embeddings_path,
	pair_dictionary_path,
	pairs_fname1,
	pairs_fname2,
	min_entity_pair_frequency=None
):

	# Open a word2vec embedder.  initializations don't matter because we'll
	# load it from file.  Supplying small numbers prevents delay of 
	# allocating large amounts of memory that we won't use
	small_number = 10
	embedder = Word2VecEmbedder(
		theano.tensor.imatrix(),
		batch_size=small_number, 
		query_vocabulary_size=small_number, 
		context_vocabulary_size=small_number, 
		num_embedding_dimensions=small_number
	)

	# Load the embeddings, and pull them out of the embedder so that
	# we can start to work with them
	embedder.load(embeddings_path)

	# Read in the entity-pair dictionary, and prune it to the correct
	# minimum frequency
	dictionary = PairDictionary()
	dictionary.load(pair_dictionary_path)

	vectors1, entities1 = get_vectors(pairs_fname1, embedder, dictionary)
	vectors2, entities2 = get_vectors(pairs_fname2, embedder, dictionary)

	results1 = test_membership(vectors1, vectors2)
	print '\naccuracy = %2.1f' % (sum(results1) / float(len(results1)))
	print '-'*32
	for result, entity_pair in zip(results1, entities1):
		print ('T\t' if result else '\tF') + '\t' + entity_pair

	results2 = test_membership(vectors2, vectors1)
	print '\naccuracy = %2.1f' % (sum(results2) / float(len(results2)))
	print '-'*32
	for result, entity_pair in zip(results2, entities2):
		print ('T\t' if result else '\tF') + '\t' + entity_pair
	


def get_vectors(fname, embedder, dictionary):

	entities = read_pair_set(fname)

	# Convert to ids, filter out UNKs
	ids = [
		id for id in dictionary.get_ids(entities) if id != 0
	]
	filtered_entities = dictionary.get_tokens(ids)

	print '-----\n' + '\n'.join(dictionary.get_tokens(ids)) + '\n\n'

	# Use masked arrays to make excluding rows easier
	vectors = embedder.embed(ids)
	vectors = np.ma.array(vectors, mask=False)

	return vectors, filtered_entities



def test_membership(vectors1, vectors2):
	# See if the model predicts membership in group1
	prototype2 = np.mean(vectors2, axis=0)
	results = []
	for i in range(len(vectors1)):

		# compute the average of group1, excluding the vector to be 
		# classified
		vectors1.mask = False
		vectors1.mask[i] = True
		prototype1 = np.mean(vectors1, axis=0)

		# Test fitness for this vector in both groups
		group1_fitness = cosine(vectors1[i], prototype1)
		group2_fitness = cosine(vectors1[i], prototype2)

		# Print the result
		results.append(group1_fitness > group2_fitness)

	return results



def cosine(vec1, vec2):

	dot = np.dot(vec1, vec2)
	norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
	return dot / norm


if __name__ == '__main__':
	embeddings_path = sys.argv[1]
	min_entity_pair_requency = sys.argv[2]
	pair_dictionary_path = sys.argv[3]

	test_similarity(
		embeddings_path,
		min_entity_pair_requency,
		pair_dictionary_path
	)

