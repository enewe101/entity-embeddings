#!/usr/bin/env python


# Import external libraries
import json
import t4k
import time
import sys
import re
import os
import numpy as np
from theano import tensor as T
from r2v import relation2vec


# Import internal libraries
sys.path.append('..')
from theano_minibatcher import NoiseContrastiveTheanoMinibatcher
from relation2vec_embedder import Relation2VecEmbedder
from dataset_reader import Relation2VecDatasetReader as DatasetReader
from SETTINGS import DATA_DIR, COOCCURRENCE_DIR, SRC_DIR


# Seed randomness for reproducibility
np.random.seed(0)


# Set defaults and constants
USAGE = (
	'Usage: run-r2v \'command="command"\' \'save_dir="save/dir"\''
	' [optional_key=val ...]'
)
DIRECTORIES = [COOCCURRENCE_DIR]
FILES = [
	#os.path.join(COOCCURRENCE_DIR, '%s.tsv' % hex(i)[2:].zfill(3))
	#for i in range(2)
]
SKIP = [re.compile('README.txt')]
BATCH_SIZE=int(1e4)
MACROBATCH_SIZE=int(1e6)
NOISE_RATIO = 15
MIN_FREQUENCY = 10
NUM_EMBEDDING_DIMENSIONS = 500
NUM_EPOCHS = 1
LEARNING_RATE = 0.002
NUM_PROCESSES = 1
MOMENTUM = 0.9
MAX_QUEUE_SIZE = 2
VERBOSE = True
LOAD_DICT_DIR = os.path.join(DATA_DIR, 'dictionaries')
READ_DATA_ASYNC = True

def prepare_dataset(params):
	save_dir = params.pop('save_dir')
	reader = DatasetReader(**params)
	reader.prepare(save_dir=save_dir)


def train(params):
	relation2vec(**params)


def commandline2dict():
	properties = {}
	for arg in sys.argv[1:]:
		key, val = arg.split('=')

		# Interpret numeric, list, and dictionary values properly, as
		# well as strings enquoted in properly escaped quotes
		try:
			properties[key] = json.loads(val)

		# It's cumbersome to always have to escape quotes around strings.
		# This caught exception interprets unenquoted tokens as strings
		except ValueError:
			properties[key] = val

	return properties


def print_params(params):

	# Print to stdout the set of parameters defining this run in a 
	# json-like format, but with keys sorted lexicographically
	params_to_print = dict(params)
	params_to_print['skip'] = [r.pattern for r in params['skip']]
	for key in sorted(params_to_print.keys()):
		print key, '=', params_to_print[key]


if __name__ == '__main__':

	commandline_params = commandline2dict()
	try:
		command = commandline_params.pop('command')
		save_dir = commandline_params.pop('save_dir')
	except KeyError:
		raise ValueError(USAGE)

	# This is just included to test argument parsing
	if command == 'args':
		print json.dumps(commandline_params)

	# Run over the entire dataset, prepare and save the entity and context 
	# dictionaries.  No training is done.  This only has to be done once,
	# subsequent calls to the training subcommand will use the dictionaries
	# saved by this method
	elif command == 'prepare':

		# Setup default params for this command
		params = {
			# Note we override the global default here
			'load_dictionary_dir': None,

			# Take these relevant settings from global defaults
			'files': FILES,
			'directories': DIRECTORIES,
			'skip': SKIP,
			'save_dir': save_dir,
			'noise_ratio': NOISE_RATIO,
			'macrobatch_size': MACROBATCH_SIZE,
			'max_queue_size': MAX_QUEUE_SIZE,
			'verbose': VERBOSE,
			'num_processes': NUM_PROCESSES,
		}

		# Merge in command line params (which override the defaults) 
		params.update(commandline_params)

		# Record params to stdout
		print
		print 'command:', command
		print_params(params)
		print

		# Run the dictionary preparation, recording total elapsed time
		start = time.time()
		prepare_dataset(params)
		elapsed = time.time() - start
		print '\nelapsed:', elapsed


	elif command == 'train':

		params = {
			'files': FILES,
			'directories': DIRECTORIES,
			'skip': SKIP,
			'save_dir': save_dir,
			'num_epochs': NUM_EPOCHS,
			'load_dictionary_dir': LOAD_DICT_DIR,
			'min_frequency': MIN_FREQUENCY,
			'noise_ratio': NOISE_RATIO,
			'batch_size': BATCH_SIZE,
			'macrobatch_size': MACROBATCH_SIZE,
			'max_queue_size': MAX_QUEUE_SIZE,
			'num_embedding_dimensions': NUM_EMBEDDING_DIMENSIONS,
			'learning_rate': LEARNING_RATE,
			'momentum': MOMENTUM,
			'verbose': VERBOSE,
			'num_processes': NUM_PROCESSES,
			'read_data_async': READ_DATA_ASYNC
		}

		# get command-line overrides of property values
		params.update(commandline_params)

		# Record params to stdout
		print
		print 'command:', command
		print_params(params)
		print

		# Run the dictionary preparation, recording total elapsed time
		start = time.time()
		train(params)
		elapsed = time.time() - start
		print '\nelapsed:', elapsed


	else:
		raise ValueError(
			'got unexpected subcommand: %s\n' % command 
			+ USAGE
		)


