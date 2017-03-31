import sys
sys.path.append('../lib')
import utils
from SETTINGS import DATA_DIR
import os
import random
import csv
import t4k

CANDIDATES_DIR = os.path.join(DATA_DIR, 'relational-nouns', 'candidates')
CROWDFLOWER_DIR = os.path.join(DATA_DIR, 'crowdflower')

def make_crowdflower_csv():

    # Seed randomness for reproducibility
    random.seed(0)

    # Open a file at which to write the csv file
    t4k.ensure_exists(CROWDFLOWER_DIR)
    csv_path = os.path.join(CROWDFLOWER_DIR, 'task1.csv')
    csv_f = open(csv_path, 'w')
    
    # First read the scored candidates
    pos_common_candidates = []
    neg_common_candidates = []
    for line in open(os.path.join(CANDIDATES_DIR, 'candidates1.txt')):
        token, class_ = line.split('\t')[:2]
        if class_ == '+':
            pos_common_candidates.append(token)
        elif class_ == '-':
            neg_common_candidates.append(token)
        else:
            raise ValueError('Unexpected classification character: %s' % class_)

    # We'll only keep the first 500 negatives.
    neg_common_candidates = neg_common_candidates[:500]

    # Next read the random candidates
    random_candidates_path = os.path.join(
        CANDIDATES_DIR, 'random_candidates1.txt')
    random_candidates = open(random_candidates_path).read().strip().split('\n')

    # Collect all the candidate words together and elminate dupes
    all_candidates = set(
        pos_common_candidates + neg_common_candidates + random_candidates
    )

    # Now keep track of why each word was included (i.e. was it a word labelled
    # by the classifier-to-date as positive? negative? or was it randomly 
    # sampled?  Note that a word could be both randomly drawn and labelled.
    pos_common_candidates = set(pos_common_candidates)
    neg_common_candidates = set(neg_common_candidates)
    random_candidates = set(random_candidates)
    sourced_candidates = []
    for candidate in all_candidates:
        sources = []
        if candidate in pos_common_candidates:
            sources.append('pos')
        if candidate in neg_common_candidates:
            sources.append('neg')
        if candidate in random_candidates:
            sources.append('rand')
        sourced_candidates.append((candidate, ':'.join(sources)))

    # randomize the ordering
    random.shuffle(sourced_candidates)

    # Write a csv file with the candidate words in it
    writer = csv.writer(csv_f)
    writer.writerow(['token', 'source'])
    writer.writerows(sourced_candidates)



if __name__ == '__main__':
    make_crowdflower_csv()
