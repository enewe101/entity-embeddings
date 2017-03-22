import multiprocessing
import iterable_queue as iq
import itertools
from functools import partial
import json
import os
import sys
sys.path.append('..')
from SETTINGS import DATA_DIR
from t4k import UnigramDictionary, SILENT
import classifier as c
DICTIONARY_DIR = os.path.join(DATA_DIR, 'dictionary')
import utils
from utils import (
	read_seed_file, get_train_sets, filter_seeds, ensure_unicode,
	get_test_sets
)
from nltk.stem import WordNetLemmatizer

UNRECOGNIZED_TOKENS_PATH = os.path.join(DATA_DIR, 'unrecognized-tokens.txt')


def get_MAP(n, ranks):
	'''
	Get the Mean Average Precision for the first n relevant documents.
	`ranks` should give the rank for each of the first n documents, 
	zero-indexed.
	'''

	# Define the MAP for zero results to be 1
	if n == 0:
		return 1.0

	# Look at the rank for the first n relevant documents (or as many are
	# available, if less than n) and accumulate the contributions each 
	# makes to the MAP
	numerator = 0
	precision_sum = 0
	average_precision_sum = 0
	for k in range(n):

		# Calculate the contribution of the kth relevant document 
		# to the MAP numerator.  If the k'th relevant doc didn't exist,
		# then it contributes nothing
		try:
			precision_sum += (k+1) / float(ranks[k]+1)
		except IndexError:
			pass

		average_precision_sum += precision_sum / float(k+1)

	mean_average_precision = average_precision_sum / float(n)

	return mean_average_precision


def get_top(
	n=50,
	kind='svm',
	on_unk=False,

	# SVM options
	syntax_feature_types=['baseline', 'dependency', 'hand_picked'],
	semantic_similarity=None,
	syntactic_multiplier=1.0,
	semantic_multiplier=1.0,

	# KNN options
	k=3,
):
	'''
	Build a classifier based on the given settings, and then return the
	n highest-scoring words
	'''
	evaluator = get_map_evaluator(
		kind=kind,
		on_unk=on_unk,
		syntax_feature_types=syntax_feature_types,
		semantic_similarity=semantic_similarity,
		syntactic_multiplier=syntactic_multiplier,
		semantic_multiplier=semantic_multiplier,
		k=k,
	)
	evaluator.get_scores()
	print '\n'.join([s[1] for s in evaluator.scores[:n]])


def generate_classifier_definitions(
    parameter_ranges,
    constants={}
):
    tagged_param_values = []
    for param in parameter_ranges:
        this_param_tagged_values = [
            (param, val) for val in parameter_ranges[param]]
        tagged_param_values.append(this_param_tagged_values)

    definitions = []
    for combo in itertools.product(*tagged_param_values):
        new_def = dict(constants)
        new_def.update(dict(combo))
        definitions.append(new_def)

    return definitions


def optimize_classifier(
    classifier_definitions,
    features,
    train_pos, train_neg,
    test_pos, test_neg,
    out_path,
    num_procs=12
):

    # Open the file where we'll write the results
    out_f = open(out_path, 'w')
    
    # Open queues to spread work and collect results
    work_queue = iq.IterableQueue()
    results_queue = iq.IterableQueue()

    # Load all of the classifier definitions onto the work queue, then close it
    work_producer = work_queue.get_producer()
    for clf_def in classifier_definitions:
        work_producer.put(clf_def)
    work_producer.close()

    # Start a bunch of workers, give them iterable queue endpoints.
    for proc in range(num_procs):
        p = multiprocessing.Process(
            target=evaluate_classifiers,
            args=(
                work_queue.get_consumer(),
                results_queue.get_producer(), 
                features, 
                train_pos, train_neg,
                test_pos, test_neg
            )
        )
        p.start()

    # Get an endpoint for collecting the results
    results_consumer = results_queue.get_consumer()

    # We're done making queue endpoints
    work_queue.close()
    results_queue.close()

    # Collect the results, and write them to disc
    best_score, best_threshold, best_clf_def = None, None, None
    for score, threshold, clf_def in results_consumer:

        # Write the result to stdout and to disc
        performance_record = '%f\t%f\t%s\n' % (score, threshold, str(clf_def))
        print performance_record
        out_f.write(performance_record)

        # Keep track of the best classifier definition and its performance
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_clf_def = clf_def

    # Write the best performance out to sdout and disc
    best_performance_record = '%f\t%f\t%s\n' % (
        best_score, best_threshold, str(best_clf_def))
    print best_performance_record
    out_f.write('\nBest:\n')
    out_f.write(best_performance_record)


def evaluate_classifiers(
    classifier_definitions, results_queue, 
    features, 
    train_pos, train_neg,
    test_pos, test_neg
):

    # Evaluate performance of classifier for each classifier definition, and
    # put the results onto the result queue.
    for clf_def in classifier_definitions:
        best_f1, threshold = evaluate_classifier(
            clf_def, features, 
            train_pos, train_neg,
            test_pos, test_neg,
        )
        results_queue.put((best_f1, threshold, clf_def))

    # Close the results queue when no more work will be added
    results_queue.close()
    

def evaluate_classifier(
    classifier_definition,
    features,
    train_pos, train_neg,
    test_pos, test_neg
):
    """
    Assesses the performance of the classifier defined by the dictionary
    ``classifier_definitions``.  That dictionary should provide the arguments
    needed to construct the classifier when provided to the function
    classifier.make_classifier.
    """
    print 'evaluating:', str(classifier_definition)
    cls = c.make_classifier(
        features=features,
        positives=train_pos,
        negatives=train_neg,
        **classifier_definition
    )

    scored_typed = [
        (cls.score(lemma)[0], 'pos') for lemma in test_pos
    ] + [
        (cls.score(lemma)[0], 'neg') for lemma in test_neg
    ]

    best_f1, threshold = calculate_best_score(scored_typed, metric='f1')

    return best_f1, threshold


def calculate_best_score(scored_typed, metric='f1'):
    """
    The best threshold score, above which items should be labelled positive and
    below which items should be labelled negative, is found by maximizing the
    f1-score or accuracy that would result.  The value of the metric is then 
    returned, along with the threshold score.

    INPUTS
        ``scored_typed`` should be a list of tuples of scored items, where the
        first element of the tuple is the score, and the second element is the
        true class of the item, which should be 'pos' or 'neg'

        ``metric`` can be 'f1' or 'accuracy'.

    OUTPUTS 
        ``(best_metric, threshold)`` where best_metric is the best value for
        the chosen metric, achieved when threshold is used to label items
        according to their assigned scores.
    """

    # sort all the scores, keeping their clasification bound to the score
    sorted_scored_typed = sorted(scored_typed, reverse=True)

    # We begin with the threshold score set at the max score, which means
    # putting all items into the 'neg' class.  The number of correct
    # classifications according to that threshold is the number of items that
    # are actually in the neg class
    true_pos = 0
    labelled_pos = 0
    num_pos = sum([st[1]=='pos' for st in sorted_scored_typed])
    num_correct = sum([st[1]=='neg' for st in sorted_scored_typed])

    best_f1 = 0
    best_count = num_correct
    best_pointer = -1

    # Move down through the scored items, shifting each one up to the 'pos'
    # class, and note the effect on the number of correct classifications
    # keep track of the point at which we get the largest correct count.
    for pointer in range(len(sorted_scored_typed)):
        labelled_pos += 1
        if sorted_scored_typed[pointer][1] == 'pos':
            num_correct += 1
            true_pos += 1
        elif sorted_scored_typed[pointer][1] == 'neg':
            num_correct -= 1

        if metric == 'f1':
            precision = true_pos / float(labelled_pos)
            recall = true_pos / float(num_pos)
            f1 = (
                0 if precision * recall == 0 
                else 2*precision*recall / (precision + recall)
            )
            if f1 > best_f1:
                best_f1 = f1
                best_pointer = pointer

        elif metric == 'accuracy':
            if num_correct > best_count:
                best_count = num_correct
                best_pointer = pointer

        else:
            raise ValueError(
                'Unrecognized value for `metric`: %s. ' % metric
                + "Expected 'f1' or 'accuracy'."
            )

    # Place the threshold below the last item shifted into the positive class
    if best_pointer > -1 and best_pointer < len(sorted_scored_typed) - 1:
        threshold = 0.5 * (
            sorted_scored_typed[best_pointer][0] 
            + sorted_scored_typed[best_pointer+1][0]
        )
    elif best_pointer == -1:
        threshold = sorted_scored_typed[best_pointer][0] + 0.1
    elif best_pointer == len(sorted_scored_typed) - 1:
        threshold = sorted_scored_typed[best_pointer][0] - 0.1
    else:
        RuntimeError('Impossible state reached')

    if metric == 'f1':
        return best_f1, threshold

    elif metric == 'accuracy':
        return best_count / float(len(scored_typed)), threshold


def diagnose_map_evaluators(
	classifier_definitions=[],
	map_evaluators=None,
	out_path=UNRECOGNIZED_TOKENS_PATH,
	n=100
):
	'''
	Given a set of classifier definitions (which should be a list of
	dictionaries containing keyword arguments for the function 
	get_map_evaluator), create the classifier for each, and find the
	unrecognized tokens (i.e. not in the test set) for each of the 
	classifier's n top-scoring tokens
	'''
	out_file = open(UNRECOGNIZED_TOKENS_PATH, 'w')

	# Create the map evaluators, if they weren't supplied
	if map_evaluators is None:
		map_evaluators = [
			(cdef, get_map_evaluator(**cdef)) 
			for cdef in classifier_definitions
		]

	# For each classifier, find the unrecognized tokens among the top n
	# scoring tokens, and write them to file
	for cdef, map_evaluator in map_evaluators:
		unrecognized_tokens = map_evaluator.diagnose_MAP(n)
		out_file.write(str(cdef)+'\n')
		out_file.write('\n'.join(unrecognized_tokens) + '\n\n')

	return map_evaluators


def get_map_evaluator(
	kind='svm',
	on_unk=False,
	syntax_feature_types=['baseline', 'dependency', 'hand_picked'],
	semantic_similarity=None,
	syntactic_multiplier=1.0,
	semantic_multiplier=1.0,
	k=3,
):
	classifier = c.make_classifier(
		kind=kind,
		on_unk=on_unk,
		syntax_feature_types=syntax_feature_types,
		semantic_similarity=semantic_similarity,
		syntactic_multiplier=syntactic_multiplier,
		semantic_multiplier=semantic_multiplier,
		k=k,
	)
	train_positives, train_negatives, train_neutrals = get_train_sets()
	test_positives, test_negatives, test_neutrals = get_test_sets()
	evaluator = RelationalNounMapEvaluator(
		classifier, train_positives, train_negatives, test_positives,
		test_negatives
	)

	return evaluator


def get_descriminant_func(test_positives=None, test_negatives=None):

	if test_positives is None or test_negatives is None:
		test_positives, test_negatives = get_test_sets()
	
	def descriminant_func(result):
		try:
			if result in test_positives:
				return True
			elif result in test_negatives:
				return False
			else:
				raise ValueError(
					'Cannot decide if %s is relevant' % str(result)
				)
		except TypeError:
			print type(result), repr(result)

	return descriminant_func



class RelationalNounMapEvaluator(object):

	def __init__(
		self, 
		classifier,
		train_positives, 
		train_negatives,
		test_positives,
		test_negatives,
	):

		# Register parameters
		self.classifier = classifier
		self.training_data = train_positives | train_negatives
		self.testing_data = test_positives | test_negatives
		self.test_positives = test_positives
		self.test_negatives = test_negatives
		self.vocabulary = utils.read_wordnet_index()

		# Make a MAP evaluator
		descriminant_func = get_descriminant_func(
			test_positives, test_negatives
		)
		self.evaluator = MapEvaluator(descriminant_func)

		# This will hold the "relational-nounishness-scores" to rank the
		# nouns that seem most relational
		self.scores = None


	def diagnose_MAP(self, n):
		# Get classifier's scores for all tokens (if not already done)
		if self.scores is None:
			self.get_scores()

		# Go through the top n tokens and collect all unrecognized ones
		print 'checking...'
		num_seen = 0
		unrecognized_tokens = []
		top_tokens = (s[1] for s in self.scores)
		for token in top_tokens:

			# Check if this token is recognized
			if token not in self.testing_data:
				unrecognized_tokens.append(token)

			# Only look at the top n tokens
			num_seen += 1
			if num_seen >= n:
				break

		return unrecognized_tokens


	def get_MAP(self, n):
		if self.scores is None:
			self.get_scores()

		top_tokens = (s[1] for s in self.scores)
		return self.evaluator.get_MAP(n, top_tokens)


	def get_scores(self):
		# Get the scores for each token in the vocabulary
		# Skip any tokens that were in the training_data!
		print 'scoring...'
		self.scores = [
			(self.classifier.score(token), token)
			for token in self.vocabulary
			if token not in self.training_data
		]
		print 'sorting...'
		self.scores.sort(reverse=True)



class MapEvaluator(object):
	'''
	Class for evaluating Mean Average Precision for some result-iterator.
	It requires a `descriminant_funct` -- a function that can distinguish 
	which results are relevant and which are irrelevant.
	'''

	def __init__(self, descriminant_func):
		self.descriminant_func = descriminant_func


	def get_MAP(self, n, results_iterator):
		'''
		Given some iterator that yields results (`results_iterator`), 
		calculate it's Mean Average Precision. 
		'''
		ranks = self.get_ranks(n, results_iterator)
		return get_MAP(n, ranks)


	def get_ranks(self, n, results_iterator):
		'''
		Given some iterator that yields results (`results_iterator`), 
		find the ranks for the first n relevant results.  
		'''

		# If n is 0, we don't have to report any ranks at all
		if n == 0:
			return []

		i = 0		# indexes relevant results
		ranks = []	# stores ranks

		# Get the ranks for the first `n` relevant
		for rank, result in enumerate(results_iterator):

			# If a result is positive, save it's rank
			if self.descriminant_func(result):
				ranks.append(rank)
				i += 1

			# Stop if we've found `n` relevant results
			if i >= n:
				break

		return ranks


def cross_val_positives(classifier='svm', clf_kwargs={}, use_wordnet=False):
	positive_seeds, negative_seeds = get_train_sets()
	features = load_features()
	dictionary = get_dictionary(features)
	positive_seeds = filter_seeds(positive_seeds, dictionary)
	negative_seeds = filter_seeds(negative_seeds, dictionary)

	num_correct = 0
	num_tested = 0
	for test_item in positive_seeds:
		positive_seeds_filtered = [
			p for p in positive_seeds if p is not test_item
		]

		args = (features,dictionary,positive_seeds_filtered,negative_seeds)
		if classifier == 'svm':
			clf = make_svm_classifier(
				*args, clf_kwargs=clf_kwargs, use_wordnet=use_wordnet)
		elif classifier == 'knn':
			clf = make_knn_classifier(*args, clf_kwargs=clf_kwargs)
		else:
			raise ValueError('Unexpected classifier type: %s.' % classifier)

		num_tested += 1
		prediction = clf.predict(test_item)[0]
		if prediction:
			num_correct += 1

		padding = 40 - len(test_item)
		print test_item, ' ' * padding, 'correct' if prediction else '-'

	print (
		'\n' + '-'*70 + '\n\n' +
		'true positives / positives = %f' 
		% (num_correct / float(num_tested))
	)




def cross_val_negatives(classifier='svm', clf_kwargs={}, use_wordnet=False):
	positive_seeds, negative_seeds = get_train_sets()
	features = load_features()
	dictionary = get_dictionary(features)
	positive_seeds = filter_seeds(positive_seeds, dictionary)
	negative_seeds = filter_seeds(negative_seeds, dictionary)

	num_correct = 0
	num_tested = 0
	for test_item in negative_seeds:
		negative_seeds_filtered = [
			n for n in negative_seeds if n is not test_item
		]

		args = (features,dictionary,positive_seeds,negative_seeds_filtered)
		if classifier == 'svm':
			clf = make_svm_classifier(
				*args, clf_kwargs=clf_kwargs, use_wordnet=use_wordnet)
		elif classifier == 'knn':
			clf = make_knn_classifier(*args, clf_kwargs=clf_kwargs)
		else:
			raise ValueError('Unexpected classifier type: %s.' % classifier)

		num_tested += 1
		prediction = clf.predict(test_item)[0]
		if not prediction:
			num_correct += 1

		padding = 40 - len(test_item)
		print test_item, ' ' * padding, '-' if prediction else 'correct'

	print (
		'\n' + '-'*70 + '\n\n' +
		'true negatives / negatives = %f' 
		% (num_correct / float(num_tested))
	)

if __name__ == '__main__':
	diagnose_map_evaluators([{}])
