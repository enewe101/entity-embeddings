import os
GIGAWORD_DIR = '/home/ndg/dataset/gigaword-corenlp/'
DATA_DIR = '/home/ndg/users/enewel3/relational-nouns/data'
SCRATCH_DIR = '/gs/scratch/enewel3/relational-nouns'
COOCCURRENCE_DIR = '/home/ndg/users/enewel3/relational-nouns/data/cooccurrence'
RELATIONAL_WORDS_PATH = os.path.join(DATA_DIR, 'relational-nouns')
TRAIN_PATH = os.path.join(RELATIONAL_WORDS_PATH, 'train', 'all.tsv')
TEST_PATH = os.path.join(RELATIONAL_WORDS_PATH, 'test', 'all.tsv')
SEED_PATH = os.path.join(RELATIONAL_WORDS_PATH, 'categorized.tsv')
NOMBANK_PATH = os.path.join(DATA_DIR, 'nombank.txt')

TRAIN_POSITIVE_PATH = os.path.join(
	RELATIONAL_WORDS_PATH, 'train-positive.txt'
)
TRAIN_NEGATIVE_PATH = os.path.join(
	RELATIONAL_WORDS_PATH, 'train-negative.txt'
)
TEST_POSITIVE_PATH = os.path.join(
	RELATIONAL_WORDS_PATH, 'test-positive.txt'
)
TEST_NEGATIVE_PATH = os.path.join(
	RELATIONAL_WORDS_PATH, 'test-negative.txt'
)

#DEPENDENCY_FEATURES_PATH = os.path.join(
#    DATA_DIR, 'relational-noun-dependency-features.json')
#BASELINE_FEATURES_PATH = os.path.join(
#    DATA_DIR, 'relational-noun-baseline-features.json')
#HAND_PICKED_FEATURES_PATH = os.path.join(
#    DATA_DIR, 'relational-noun-hand-picked-features.json')
#DICTIONARY_DIR = os.path.join(DATA_DIR, 'lemmatized-noun-dictionary')

WORDNET_INDEX_PATH = os.path.join(DATA_DIR, 'wordnet_index.txt')
RELATIONAL_NOUN_FEATURES_DIR = os.path.join(
	DATA_DIR, 'relational-noun-features-lexical-wordnet')

ACCUMULATED_FEATURES_PATH = os.path.join(RELATIONAL_NOUN_FEATURES_DIR, '000')

SUFFIX_PATH = os.path.join(DATA_DIR, 'suffixes.txt')
GOOGLE_VECTORS_PATH = os.path.join(DATA_DIR, 'google-vectors-negative-300.txt')
