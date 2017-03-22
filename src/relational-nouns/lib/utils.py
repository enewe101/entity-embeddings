import random
import os
import sys
sys.path.append('..')
import cjson
from t4k import UNK, UnigramDictionary
from SETTINGS import TRAIN_PATH, TEST_PATH, WORDNET_INDEX_PATH, SEED_PATH


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
    dictionary = UnigramDictionary()
    dictionary.load(path)
    return dictionary


def get_train_sets():
    '''
    Get a set of positive (relational) words and a set of negative 
    (non-relational) words, to be used as a training set
    '''
    positives, negatives, neutrals = get_seed_set(TRAIN_PATH)
    return positives, negatives, neutrals


def get_test_sets():
    '''
    Get a set of positive (relational) words and a set of negative 
    (non-relational) words, to be used as a test set
    '''
    positives, negatives, neutrals = get_seed_set(TEST_PATH)
    return positives, negatives, neutrals


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
