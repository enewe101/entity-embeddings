#!/usr/bin/env python

# Import external libraries
import json
import time
import sys
sys.path.append('..')
import re
import os
import numpy as np


# Import internal libraries
from r2v import relation2vec
from theano_minibatcher import NoiseContrastiveTheanoMinibatcher
from dataset_reader import (
	Relation2VecDatasetReader as DatasetReader,
	FULL_CONTEXT, RANDOM_SINGLE_CHOICE
)
from SETTINGS import DATA_DIR, COOCCURRENCE_DIR 


# Seed randomness for reproducibility
np.random.seed(0)


# Set defaults and constants
USAGE = (
	'Usage: ./train_r2v.py \'command="command"\' \'save_dir="save/dir"\''
	' [optional_key=val ...]'
)
DIRECTORIES = [COOCCURRENCE_DIR]
FILES = [
	#os.path.join(COOCCURRENCE_DIR, '%s.tsv' % hex(i)[2:].zfill(3))
	#for i in range(2)
]
SKIP = [r'README\.txt']
BATCH_SIZE=int(1e4)
MACROBATCH_SIZE=int(1e6)
NOISE_RATIO = 15
MIN_QUERY_FREQUENCY = 10
MIN_CONTEXT_FREQUENCY = 10
NUM_EMBEDDING_DIMENSIONS = 500
NUM_EPOCHS = 1
LEARNING_RATE = 0.002
NUM_PROCESSES = 1
MOMENTUM = 0.9
MAX_QUEUE_SIZE = 2
VERBOSE = True
LOAD_DICT_DIR = os.path.join(DATA_DIR, 'dictionaries')
READ_DATA_ASYNC = True
CONTEXT_EMBEDDINGS_FNAME = os.path.join(
	DATA_DIR, 'google-vectors-negative-300.txt')
ENTITY_NOISE_RATIO = 0.0
SIGNAL_SAMPLE_MODE = RANDOM_SINGLE_CHOICE
LEN_CONTEXT = 1
FREEZE_CONTEXT = False


def prepare_dataset(params):
	save_dir = params.pop('save_dir')
	reader = DatasetReader(**params)
	reader.prepare(save_dir=save_dir)


def train(params):
	relation2vec(**params)


legal_params = {
	'command', 'files', 'directories', 'skip', 'save_dir', 'num_epochs',
	'min_query_frequency', 'min_context_frequency', 'noise_ratio', 
	'batch_size', 'macrobatch_size',
	'max_queue_size', 'num_embedding_dimensions', 'learning_rate',
	'momentum', 'verbose', 'num_processes', 'read_data_async',
	'context_embeddings_fname', 'load_dictionary_dir', 'signal_sample_mode',
	'entity_noise_ratio', 'len_context', 'freeze_context'
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
	for key in sorted(params.keys()):
		print key, '=', repr(params[key])



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
			'num_epochs': NUM_EPOCHS,
			'load_dictionary_dir': LOAD_DICT_DIR,
			'min_query_frequency': MIN_QUERY_FREQUENCY,
			'min_context_frequency': MIN_CONTEXT_FREQUENCY,
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
			'context_embeddings_fname': CONTEXT_EMBEDDINGS_FNAME,
			'signal_sample_mode': SIGNAL_SAMPLE_MODE,
			'entity_noise_ratio': ENTITY_NOISE_RATIO,
			'len_context': LEN_CONTEXT,
			'freeze_context': FREEZE_CONTEXT,
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


