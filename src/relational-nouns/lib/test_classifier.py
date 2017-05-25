import t4k
import numpy as np
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
from utils import read_seed_file, filter_seeds, ensure_unicode
from nltk.stem import WordNetLemmatizer
import annotations
import extract_features
import kernels

UNRECOGNIZED_TOKENS_PATH = os.path.join(DATA_DIR, 'unrecognized-tokens.txt')


all_sets = set(['rand', 'guess', 'top'])
def evaluate_simple_classifier(
    annotations, features, kernel, test_sets, num_folds=3
):
    """
    Assesses the performance of the classifier defined by the dictionary
    ``classifier_definitions`` using the provided data.
    """
    # For the first test, we'll take a random split of the top words
    test_sets = set(test_sets)
    training_sets = all_sets - test_sets
    results = []
    for fold in range(num_folds):

        print '\nstarting fold %d\n' % fold
        # Get the test set folds.
        X_test_folds, Y_test_folds = annotations.get(test_sets)
        X_train_fold, X_test = t4k.get_fold(X_test_folds, num_folds, fold)
        Y_train_fold, Y_test = t4k.get_fold(Y_test_folds, num_folds, fold)

        # Initially the training set consists of data not in the test set, but 
        # we also add in test_set data not being tested in this fold.
        X_train, Y_train = annotations.get(training_sets, test_sets)
        X_train = np.concatenate([X_train, X_train_fold])
        Y_train = np.concatenate([Y_train, Y_train_fold])

        # Train the classifier
        options = {'pre-bound-kernel': kernel.eval}
        clf = c.SimplerSvmClassifier(X_train, Y_train, features, options)

        # Run classifier on test set
        prediction = clf.predict(X_test)

        # Adjust the predictions / labels to be in the correct range
        prediction = prediction / 2.0
        Y_test = np.array(Y_test) / 2.0

        precision, recall, f1 = get_ordinal_f1(prediction, Y_test)
        result = {
            'test': ','.join(test_sets),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        print result
        results.append(result)

    return results




all_sets = set(['rand', 'guess', 'top'])
def evaluate_ordinal_classifier_(
    annotations, features, kernel, test_sets, num_folds=3
):
    """
    Assesses the performance of the classifier defined by the dictionary
    ``classifier_definitions`` using the provided data.
    """
    # For the first test, we'll take a random split of the top words
    test_sets = set(test_sets)
    training_sets = all_sets - test_sets
    results = []
    for fold in range(num_folds):

        print '\nstarting fold %d\n' % fold
        # Get the test set folds.
        test_pos, test_neut, test_neg = annotations.get_as_tokens(test_sets)
        pos_train_fold, pos_test_fold = t4k.get_fold(
            list(test_pos), num_folds, fold)
        neut_train_fold,neut_test_fold = t4k.get_fold(
            list(test_neut), num_folds, fold)
        neg_train_fold, neg_test_fold = t4k.get_fold(
            list(test_neg), num_folds, fold)

        train_pos, train_neut, train_neg = annotations.get_as_tokens(
            training_sets, test_sets)

        train_pos = list(train_pos) + list(pos_train_fold)
        train_neut = list(train_neut) + list(neut_train_fold)
        train_neg = list(train_neg) + list(neg_train_fold)

        # Train the classifier
        options = {'pre-bound-kernel': kernel.eval}
        # Train the classifier
        clf = c.OrdinalSvmNounClassifier(
            list(train_pos), list(train_neut), list(train_neg), kernel, features
        )

        # Run classifier on test set
        test_X = list(pos_test_fold) + list(neut_test_fold) + list(neg_test_fold)
        test_Y = (
            [1]*len(pos_test_fold) 
            + [0.5]*len(neut_test_fold) 
            + [0]*len(neg_test_fold)
        )

        prediction = np.array(clf.predict(test_X))
        prediction = (prediction + 1) / 2.0

        precision, recall, f1 = get_ordinal_f1(prediction, Y_test)
        result = {
            'test': ','.join(test_sets),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        print result
        results.append(result)

    return results











    ## For the first test, we'll take a random split of all the annotated data.
    #print 'training on all, testing on holdout'
    ## Get the test set
    #X, Y = annotations.get(['top', 'rand', 'guess'])
    #X_train, X_test = t4k.get_fold(X, 5, 0)
    #Y_train, Y_test = t4k.get_fold(Y, 5, 0)

    ## Train the classifier
    #options = {'pre-bound-kernel': kernel.eval}
    #clf = c.SimplerSvmClassifier(X_train, Y_train, features, options)
    #
    ## Run classifier on test set
    #prediction = clf.predict(X_test)

    ## Adjust the predictions / labels to be in the correct range
    #prediction = prediction / 2.0
    #Y_test = np.array(Y_test) / 2.0

    #precision, recall, f1 = get_ordinal_f1(prediction, Y_test)
    #result = {
    #    'test': 'top,rand,guess (holdout)', 
    #    'train': 'top,rand,guess',
    #    'precision': precision,
    #    'recall': recall,
    #    'f1': f1
    #}
    #print result
    #results.append(result)


    tests = [
        #('top', ['rand', 'guess']),
        #('rand', ['top', 'guess']),
        #('guess', ['rand', 'top']),
        ('top', ['guess']),
        ('guess', ['top']),
        #('top', ['rand']),
    ]
    for test, train in tests:

        print 'training on %s; testing on %s.' % (str(train), str(test))
        # Get the test set
        X_train, Y_train = annotations.get(train)
        X_test, Y_test = annotations.get(test, train)

        # Train the classifier
        options = {'pre-bound-kernel': kernel.eval}
        clf = c.SimplerSvmClassifier(X_train, Y_train, features, options)
        
        # Run classifier on test set
        prediction = clf.predict(X_test)

        # Adjust the predictions / labels to be in the correct range
        prediction = prediction / 2.0
        Y_test = np.array(Y_test) / 2.0

        precision, recall, f1 = get_ordinal_f1(prediction, Y_test)
        result = {
            'test': test, 
            'train': ','.join(train), 
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        print result
        results.append(result)

    return results



data_sources = ['guess', 'rand', 'top']
def evaluate_ordinal_classifier(annots=None, features=None, kernel=None):

    # Get the labels, features, and kernel
    if annots is None:
        annots = annotations.Annotations()
    if features is None:
        features = extract_features.get_accumulated_features()
    if kernel is None:
        kernel = kernels.PrecalculatedKernel(features)
        # Precalculate the kernel for all the data in parallel
        kernel.precompute_parallel(annots.examples.keys())

    # Do tests with an ordinal svm
    results = {}
    results['ordinal'] = []
    for i in range(len(data_sources)):

        # Use one of the sets as the training set
        train = data_sources[:i] + data_sources[i+1:]
        test = data_sources[i]

        print 'training on %s; testing on %s.' % (str(train), str(test))
        # Get the test set
        train_pos, train_neut, train_neg = annots.get_as_tokens(train)
        test_pos, test_neut, test_neg = annots.get_as_tokens(test, train)

        # Train the classifier
        clf = c.OrdinalSvmNounClassifier(
            list(train_pos), list(train_neut), list(train_neg), kernel, features
        )
        
        # Set up the training set and evaluate the classifier
        x_test = list(test_pos) + list(test_neut) + list(test_neg)
        y_test = np.array(
            [1.0]*len(test_pos) + [0.5]*len(test_neut) + [0.0]*len(test_neg))
        prediction = np.array(clf.predict(x_test))

        # Adjust the predictions to be in the correct range / scale
        prediction = (prediction + 1) / 2.0
        precision, recall, f1 = get_ordinal_f1(prediction, y_test)
        result = {
            'test': test[0], 
            'train': ','.join(train), 
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        print result
        results['ordinal'].append(result)

    return results


        
def get_ordinal_f1(predictions, actual):

    # Rescale the class coding.  Now a relational noun is 1.0
    # (for fully relational) and an occasionally relational is 0.5 (for 
    # halfway relational).
    positives = 0
    true_positives = 0
    false_positives = 0
    for predicted, actual in zip(predictions, actual):

        positives += actual
        if predicted == 0:
                pass
        elif predicted == 0.5:
            if actual == 0:
                false_positives += 0.5
            elif actual == 0.5:
                true_positives += 0.5
            elif actual == 1.0:
                true_positives += 0.5

        elif predicted == 1.0:
            if actual == 0:
                false_positives += 1.0
            elif actual == 0.5:
                true_positives += 0.5
                false_positives += 0.5
            elif actual == 1.0:
                true_positives += 1.0

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if positives == 0:
        recall = 0
    else:
        recall = true_positives / positives

    if precision+recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision , recall, f1


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

    best_f1, threshold = utils.calculate_best_score(scored_typed, metric='f1')

    return best_f1, threshold



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

        i = 0        # indexes relevant results
        ranks = []    # stores ranks

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
