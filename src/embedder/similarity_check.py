import sys
import numpy as np
from dataset_reader import PairDictionary
from word2vec import Word2VecEmbedder
import theano

EMBEDDINGS_PATH = '/gs/project/eeg-641-aa/enewel3/entity-embeddings/data/ep-between-min1000-batch1000-learn0.01'
MIN_ENTITY_PAIR_FREQUENCY = 1000
PAIR_DICTIONARY_PATH = '/gs/project/eeg-641-aa/enewel3/entity-embeddings/data/ep-between-min1000-batch1000-learn0.01/query-dictionary'

NUM_BEST = 5
NUM_WORST = 10

def test_similarity(
	embeddings_path,
	min_entity_pair_frequency,
	pair_dictionary_path
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
	W, C = embedder.get_param_values()

	# Read in the entity-pair dictionary, and prune it to the correct
	# minimum frequency
	dictionary = PairDictionary()
	dictionary.load(pair_dictionary_path)


	for i, w in enumerate(W):
		match_scores = [
			(np.dot(w, W[j])/(np.linalg.norm(w)*np.linalg.norm(W[j])), j)
			for j in range(len(W)) if i != j
		]
		match_scores.sort(reverse=True)
		# print [s for s in match_scores]
		best_match_idxs = [m[1] for m in match_scores[:NUM_BEST]]
		worst_match_idxs = [m[1] for m in match_scores[-NUM_WORST:]]

		print dictionary.get_token(i).upper(), '\n'
		print 'best matches:'
		print '\t%s\n' * NUM_BEST % tuple(
			dictionary.get_tokens(best_match_idxs))

		print 'worst matches:'
		print '\t%s\n' * NUM_WORST % tuple(
			dictionary.get_tokens(worst_match_idxs))

		print '\n'

if __name__ == '__main__':
	embeddings_path = sys.argv[1]
	min_entity_pair_requency = sys.argv[2]
	pair_dictionary_path = sys.argv[3]

	test_similarity(
		embeddings_path,
		min_entity_pair_requency,
		pair_dictionary_path
	)

