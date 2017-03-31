import random
import os
import sys
sys.path.append('..')
import cjson
import t4k
from SETTINGS import (
	WORDNET_INDEX_PATH, DATA_DIR,
    #TRAIN_PATH, TEST_PATH, SEED_PATH,
)


def calculate_best_score(scored_typed, metric='f1'):
    """
	This function helps to convert from a scoring function to a classification
	function.  Given some function which provides scores to items that are
	either "positive" or "negative", find the best threshold score that gives
	the greatest performance, when used to label any items whose score is
	higher as "positive" and lower as "negative"

	"Best performance" can either mean highest f1-score or highest accuracy,
	determined by passing either 'f1' or 'accuracy' as the argument for 
	``metric``.

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

def read_wordnet_index():
    return set(open(WORDNET_INDEX_PATH).read().split('\n'))


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


def get_seed_set(path):
    '''
    Get a set of positive (relational) words and a set of negative 
    (non-relational) words, to be used as a training set
    '''
    positives, negatives, neutrals = set(), set(), set()
    for line in open(path):
        token, classification = line.strip().split('\t')[:2]
        if classification == '+':
            positives.add(token)
        elif classification == '0':
            neutrals.add(token)
        elif classification == '-':
            negatives.add(token)
        elif classification == '?':
            continue
        else:
            raise ValueError(
                'Unexpected classification for token "%s": %s' 
                % (token, classification)
            )

    return positives, negatives, neutrals


def read_all_labels(path):
	"""
	Read all the label files within the directory given by ``path``.
	"""
	positives = set()
	negatives = set()
	neutrals = set()
	for file_path in t4k.ls(path, dirs=False):
		pos, neg, neut = get_seed_set(file_path)
		positives.update(pos)
		negatives.update(neg)
		neutrals.update(neut)

	return positives, negatives, neutrals



def get_full_seed_set():
	seed_path = os.path.join(DATA_DIR, 'relational-nouns', 'categorized.tsv')
	pos, neg, neut = get_seed_set(seed_path)
	return pos, neg, neut


def get_train_test_split():
    random.seed(0)
    split_ratio = 0.33
    pos, neg, neut = get_seed_set(SEED_PATH)

    train = {}
    test = {}

    train['pos'] = set(random.sample(pos, int(len(pos)*split_ratio)))
    test['pos'] = pos - train['pos']

    train['neg'] = set(random.sample(neg, int(len(neg)*split_ratio)))
    test['neg'] = neg - train['neg']

    train['neut'] = set(random.sample(neut, int(len(neut)*split_ratio)))
    test['neut'] = neut - train['neut']

    return train, test


def get_dictionary(path):
    dictionary = t4k.UnigramDictionary()
    dictionary.load(path)
    return dictionary


#def get_train_sets():
#    '''
#    Get a set of positive (relational) words and a set of negative 
#    (non-relational) words, to be used as a training set
#    '''
#    positives, negatives, neutrals = get_seed_set(TRAIN_PATH)
#    return positives, negatives, neutrals


#def get_test_sets():
#    '''
#    Get a set of positive (relational) words and a set of negative 
#    (non-relational) words, to be used as a test set
#    '''
#    positives, negatives, neutrals = get_seed_set(TEST_PATH)
#    return positives, negatives, neutrals


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
    return [w for w in words if w in dictionary]


def ensure_unicode(s):
    try:
        return s.decode('utf8')
    except UnicodeEncodeError:
        return s

def normalize_token(s):
    """Ensures the token is unicode and lower-cased."""
    return ensure_unicode(s).lower()
