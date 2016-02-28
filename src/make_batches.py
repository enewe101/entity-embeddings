#!/usr/bin/env python
'''
This script outputs a "batch file", which is used to separate the work
of processing all the data through the worldviews pipeline into batches.
Each batch is run on a separate machine in the cluster.  This batch file
not only makes it easy to identify mutually exclusive subsets of the data
for processing, it also splits the pipeline into two separate steps:
one performs entity linking using AIDA, and the other does the remaining
desired steps (but these remaining steps exclude 
'extract-intervening-texts','patty-edgelist', and 'aggregate-edgelists',
which are of no use for the entity-embedding project).
'''

import os
import itertools
import json
from SETTINGS import DATA_DIR, SCRATCH_DIR

BATCHES_PATH = os.path.join(DATA_DIR, 'batches.txt')

def make_batches():

	batches_f = open(BATCHES_PATH, 'w')
	dirnames = [
		''.join(c) 
		for c in itertools.product('0123456789abcdef', repeat=2)
	]

	# Make the batches for the aida linking step 
	for dirname in dirnames:
		in_dir = os.path.join(SCRATCH_DIR, 'scraped-text', dirname)
		out_dir = os.path.join(SCRATCH_DIR, 'worldviews', dirname)
		aida_batch = {
			'in_dirs': in_dir,
			'out_dir': out_dir,
			'until': 'aida-link'
		}
		batches_f.write(json.dumps(aida_batch) + '\n')

	# Make the batches for all steps other than aida linking step
	for dirname in dirnames:
		in_dir = os.path.join(SCRATCH_DIR, 'scraped-text', dirname)
		out_dir = os.path.join(SCRATCH_DIR, 'worldviews', dirname)
		all_but_aida_batch = {
			'in_dirs': in_dir,
			'out_dir': out_dir,
			'skip': [
				'extract-intervening-texts','patty-edgelist',
				'aggregate-edgelists'
			]
		}
		batches_f.write(json.dumps(all_but_aida_batch) + '\n')


if __name__ == '__main__':
	make_batches()

	
