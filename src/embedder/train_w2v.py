#!/usr/bin/env python

# Import external libraries
import json
import time
import sys
sys.path.append('..') # This means that settings is importable
import re
import os
import numpy as np

# Import internal libraries
from dataset_reader import word2vec_parse
from SETTINGS import DATA_DIR, COOCCURRENCE_DIR
from word2vec import word2vec, DatasetReader
from SETTINGS import DATA_DIR, COOCCURRENCE_DIR 


# Seed randomness for reproducibility
np.random.seed(0)


# Define training defaults
USAGE = (
	'Usage: ./train_r2v.py \'command="command"\' \'save_dir="save/dir"\''
	' [optional_key=val ...]'
)
FILES = [
	os.path.join(COOCCURRENCE_DIR, '%s.tsv' % file_num)
	for file_num in [
		'002', '003', '006', '007', '009', '00d'
		'00e', '010', '017', '018', '01b', '01d'
	]
]
DIRECTORIES = [
	#COOCCURRENCE_DIR
]
SKIP = [
	re.compile(r'README\.txt'), re.compile(r'test')
]
THRESHOLD = 1 # This means there will be no discarding
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
LOAD_DICT_DIR = os.path.join(DATA_DIR, 'word2vec-dictionaries')
READ_DATA_ASYNC = True
FREEZE_CONTEXT = False
_T = 1e-5
KERNEL=[1,2,3,4,5,5,4,3,2,1]


def prepare_dataset(params):
	save_dir = params.pop('save_dir')

	# Use a parsing function specific to the gigaword cooccurrence files
	params['parse'] = word2vec_parse

	reader = DatasetReader(**params)
	reader.prepare(save_dir=save_dir)


def train(params):
	word2vec(**params)


legal_params = {
	
	# Command option (either prepare or train)
	'command', 

	# Input / output options
	'files',
	'directories',
	'skip',
	'save_dir', 
	'read_data_async',
	'num_processes',
	'max_queue_size',

	# Batching options
	'num_epochs',
	'batch_size',
	'macrobatch_size',

	# Dictionary options
	'load_dictionary_dir',
	'min_frequency',

	# Sampling options
	'noise_ratio',
	'kernel'
	't',

	# Embedding options
	'num_embedding_dimensions',

	# Learning rate options
	'learning_rate',
	'momentum',

	# verbosity
	'verbose'
}


def commandline2dict():
	properties = {}
	for arg in sys.argv[1:]:
		key, val = arg.split('=')

		if key not in legal_params:
			raise ValueError('Unrecognized argument: %s' % key)

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
	if 'skip' in params_to_print:
		params_to_print['skip'] = [r.pattern for r in params['skip']]
	for key in sorted(params_to_print.keys()):
		print key, '=', params_to_print[key]


if __name__ == '__main__':

	commandline_params = commandline2dict()
	try:
		command = commandline_params.pop('command')
		assert('save_dir' in commandline_params)
	except (KeyError, AssertionError):
		raise ValueError(USAGE)

	# This is just included to test argument parsing
	if command == 'args':
		print
		print 'command:', command
		print_params(commandline_params)
		print

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
			'noise_ratio': NOISE_RATIO,
			't': _T,
			'num_processes': NUM_PROCESSES,
			'kernel': KERNEL,
			'max_queue_size': MAX_QUEUE_SIZE,
			'macrobatch_size': MACROBATCH_SIZE,
			'verbose': VERBOSE,
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
			'read_data_async': READ_DATA_ASYNC,
			'kernel': KERNEL,
			't': _T,
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


