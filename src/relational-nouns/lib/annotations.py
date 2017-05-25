import t4k
import os
import sys
from SETTINGS import DATA_DIR

ANNOTATIONS_PATH = os.path.join(
    DATA_DIR, 'relational-nouns', 'all-annotations.tsv')
DICTIONARY_DIR = os.path.join(
    DATA_DIR, 'relational-noun-features-wordnet-only', 
    'accumulated-pruned-5000', 'dictionary'
)

class Annotations(object):

    def __init__(self):

        # Read a dictionary which maps tokens to ids (so they can be used in 
        # classifiers)
        self.dictionary = t4k.UnigramDictionary()
        self.dictionary.load(DICTIONARY_DIR)

        # All tokens and their classification are stored here, according to 
        # their source
        self.examples_by_source = {
            'rand':set(),
            'top':set(),
            'guess':set()
        }
        self.guessed_examples_by_source = {
            'pos': set(),
            'neut': set(),
            'neg': set()
        }
        self.example_source_lookup = {}

        self.examples = {}

        for line in open(ANNOTATIONS_PATH):

            # Clean trailing whitespace and skip blank lines
            line = line.strip()
            if line == '':
                continue

            # Parse and store this example
            token, sources, annotator, label = line.split('\t')
            self.examples[token] = get_label(label)
            self.example_source_lookup[token] = (
                get_sources(sources) + list(get_guessed_sources(sources)))
            for source in get_sources(sources):
                self.examples_by_source[source].add(token)
            for source in get_guessed_sources(sources):
                self.guessed_examples_by_source[source].add(token)


    def get_source_tokens(self, sources, exclude_sources=[]):

        # Sources can be a single source or list of sources. Normalize to list.
        if isinstance(sources, basestring):
            sources = [sources]

        # Same for exclude sources
        if isinstance(exclude_sources, basestring):
            exclude_sources = [exclude_sources]

        # Get all the tokens for the requested sources
        tokens = set()
        for source in sources:
            if source in self.guessed_examples_by_source:
                tokens |= self.guessed_examples_by_source[source]
            else:
                tokens |= self.examples_by_source[source]

        # Get all the tokens for the excluded sources
        exclude_tokens = set()
        for source in exclude_sources:
            if source in self.guessed_examples_by_source:
                exclude_tokens |= self.guessed_examples_by_source[source]
            else:
                exclude_tokens |= self.examples_by_source[source]

        tokens = tokens - exclude_tokens

        return tokens


    def get_as_tokens(self, sources, exclude_sources=[]):
        """
        Return token sets corresponding to the positives, negatives, and
        neutrals found within the given sources but not found within exclude
        sources.
        """
        tokens = self.get_source_tokens(sources, exclude_sources)

        # convert tokens to an array of token-ids and provide labels as 
        # separate array
        positive, negative, neutral = set(), set(), set()
        for token in tokens:
            if self.examples[token] == 2:
                positive.add(token)
            elif self.examples[token] == 1:
                neutral.add(token)
            elif self.examples[token] == 0:
                negative.add(token)

        return positive, neutral, negative


    def get(self, sources, exclude_sources=[]):
        """
        Get the feature and label arrays, in a format suitable for scikit 
        classifiers.
        """

        tokens = self.get_source_tokens(sources, exclude_sources)

        # convert tokens to an array of token-ids and provide labels as 
        # separate array
        X, Y = [], []
        X = [[self.dictionary.get_id(token)] for token in tokens]
        Y = [self.examples[token] for token in tokens]

        return X, Y

                


# This converts the sources from which an example was drawn into a 
# smaller set of standardized sources.
SOURCE_MAP = {
    'rand': 'rand',
    'rand2': 'rand',
    'top': 'top',
    'pos': 'guess',
    'neg': 'guess',
    'neut': 'guess',
    'pos2': 'guess',
    'neg2': 'guess',
    'neut2': 'guess'
}
def get_sources(sources):
    return [SOURCE_MAP[s] for s in  sources.split(':')]

GUESSED_SOURCE_MAP = {
    'pos': 'pos',
    'neg': 'neg',
    'neut': 'neut',
    'pos2': 'pos',
    'neg2': 'neg',
    'neut2': 'neut'
}

def get_guessed_sources(sources):
    for source in sources.split(':'):
        if source in GUESSED_SOURCE_MAP:
            yield GUESSED_SOURCE_MAP[source]


# This converts labels to integers
LABEL_MAP = {
    '-': 0, # never relational
    '0': 1, # occasionally relational
    '+': 2  # usually relational
}
def get_label(label):
    return LABEL_MAP[label]
