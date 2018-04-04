import os
GIGAWORD_DIR = '/home/ndg/dataset/gigaword-corenlp/'
DATA_DIR = '/home/ndg/users/enewel3/entity-embeddings/data'
SCRATCH_DIR = '/gs/scratch/enewel3/entity-embeddings'
COOCCURRENCE_DIR = '/home/ndg/users/enewel3/entity-embeddings/data/cooccurrence'
RELATIONAL_WORDS_PATH = os.path.join(DATA_DIR, 'relational-nouns')
TRAIN_DIR = os.path.join(RELATIONAL_WORDS_PATH, 'train')
TEST_DIR = os.path.join(RELATIONAL_WORDS_PATH, 'test')

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
DEPENDENCY_FEATURES_PATH = os.path.join(
    DATA_DIR, 'relational-noun-dependency-features.json')
BASELINE_FEATURES_PATH = os.path.join(
    DATA_DIR, 'relational-noun-baseline-features.json')
HAND_PICKED_FEATURES_PATH = os.path.join(
    DATA_DIR, 'relational-noun-hand-picked-features.json')
DICTIONARY_DIR = os.path.join(DATA_DIR, 'lemmatized-noun-dictionary')

RELATIONAL_NOUN_FEATURES_DIR = os.path.join(
	DATA_DIR, 'relational-noun-features'
)
