import sys
sys.path.join('../lib')
from SETTINGS import DATA_DIR
import classifier
import utils

BEST_CLASSIFIER_THRESHOLD = -0.331195
BEST_CLASSIFIER_CONFIG = {
	'kind': 'svm',
	'on_unk': False,
	'C': 0.01,
	'syntax_feature_types': ['baseline', 'dependency', 'hand_picked'],
	'semantic_similarity': 'res',
	'include_suffix' :  True,
	'syntactic_multiplier': 10.0,
	'semantic_multiplier': 2.0,
	'suffix_multiplier': 0.2
}
BEST_WORDNET_ONLY_FEATURES_PATH = os.path.join(
	DATA_DIR, 'relational-noun-features-wordnet-only', 'accumulated-pruned-5000'
)

CANDIDATES_DIR = os.path.join(DATA_DIR, 'relational_nouns', 'candidates')

def do_generate_candidates():

	# Where will we write the new candidates?
	t4k.ensure_exists(CANDIDATES_DIR)
	out_path = os.path.join(CANDIDATES_DIR, 'candidates1.txt')

	# Read in the seed set, which we'll need for a couple things
	pos, neg, neut = utils.get_full_seed_set()

	# Make the best performing classifier.  This is what we'll use to score the
	# "relationalness" of new words.
	clf = classifier.make_classifier(
		features=BEST_WORDNET_ONLY_FEATURES_PATH,
		positives=pos,
		negatives=neg,
		**BEST_CLASSIFIER_CONFIG
	)

	print clf.threshold


	## Now generate the candidates
	#for token in dictionary.get_token_list():
	#	clf.score(token)

	#generate_candidates(1000, clf.score, pos|neg|neut, out_path)


def generate_candidates(num_to_generate, sorted_dictionary, score_func, exclude):

	work_queue = iterable_queue.IterableQueue()
	results_queue = iterable_queue.IterableQueue()

	# Add all of the words to the work producer
	all_words = utils.read_wordnet_index()
	for word in all_words:
		work_producer.put(word)


	work_producer = work_queue.get_producer()
	for token in sorted_dictionary.get_token_list():
		work_producer.put(candidates)


if __name__ == '__main__':
	do_generate_candidates()
