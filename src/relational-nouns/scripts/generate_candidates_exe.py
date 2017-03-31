import sys
sys.path.append('../lib')
import os
import t4k
from SETTINGS import DATA_DIR
import generate_candidates
import utils
import random

CANDIDATES_DIR = os.path.join(DATA_DIR, 'relational-nouns', 'candidates')
BEST_FEATURES_DIR = os.path.join(
    DATA_DIR, 'relational-noun-features-wordnet-only', 
    'accumulated-pruned-5000')

def do_generate_candidates1():

    # Decide the output path and the number of positive candidates to find
    t4k.ensure_exists(CANDIDATES_DIR)
    out_path = os.path.join(CANDIDATES_DIR, 'candidates1.txt')
    num_to_generate = 1000

    # Read in the seed set, which is the basis for the model that selects new 
    # candidates
    pos, neg, neut = utils.get_full_seed_set()

    # Don't keep any candidates that were already in the seed set
    exclude = pos | neg | neut

    generate_candidates.generate_candidates(
        num_to_generate, out_path, pos, neg, exclude)


def generate_uniform_random_candidates1():

    # Open a path that we want to write to 
    out_path = os.path.join(CANDIDATES_DIR, 'random_candidates1.txt')
    out_f = open(out_path, 'w')

    # Open the dictionary of words seen in the corpus
    dictionary_path = os.path.join(BEST_FEATURES_DIR, 'dictionary')
    dictionary = t4k.UnigramDictionary()
    dictionary.load(dictionary_path)

    # Don't keep any candidates that were already in the seed set
    pos, neg, neut = utils.get_full_seed_set()
    exclude = pos | neg | neut

    # Uniformly randomly sample from it
    samples = set()
    num_to_sample = 500
    while len(samples) < num_to_sample:
        token = random.choice(dictionary.token_map.tokens)
        if token != 'UNK' and token not in exclude and token not in samples:
            samples.add(token)

    out_f.write('\n'.join(samples))




if __name__ == '__main__':
    # do_generate_candidates1()
    generate_uniform_random_candidates1()
