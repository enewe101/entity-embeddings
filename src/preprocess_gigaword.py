#!/usr/bin/env python
import subprocess
from SETTINGS import WORLDVIEWS_DIR
import os
import sys
from getopt import getopt, GetoptError
from lib import utils

DEFAULT_NUM_CORES = 12
WORLDVIEWS_EXECUTABLE = os.path.join(WORLDVIEWS_DIR, 'run.py')

def get_options(args):

	usage = (
		'Usage:\n\n\t'
		'./preprocess_gigaword.py '
		'[-n number-of-processes] '
		'-b batch-number'
	)

	# Parse the command line arguments
	try:
		options, extraneous_args = getopt(sys.argv[1:], 'n:b:')

	# If arg parsing fails, emit usage
	except GetoptError:
		print 'Could not parse arguments\n'
		print usage
		exit(1)

	# If unrecognized args given, emit usage
	if len(extraneous_args) > 0:
		print 'Unrecognized arguments\n'
		print usage
		exit(1)

	options = dict(options)

	# Get the intput dir.  Fail with usage message if none provided.
	try:
		batch_num = options['-b']
	except KeyError:
		print 'The batch number must be specified.\n'
		print usage
		exit(1)

	# Get the number of proceses, or use the default
	num_processes = int(options.get('-n', DEFAULT_NUM_CORES))

	return batch_num, num_processes


def preprocess_gigaword():
	batch_num, num_processes = get_options(sys.argv[1:])
	batch_num = int(batch_num)
	in_dir, out_dir, until, skip, only = utils.read_batch(batch_num)
	print 'batch-num', batch_num
	print 'num_processes', num_processes
	print 'in_dir', in_dir
	print 'out_dir', out_dir
	print 'until', until
	print 'skip', skip
	print 'only', only

	# Invoke the worldviews pipeline
	worldviews_command = [
		WORLDVIEWS_EXECUTABLE,
		'-i', in_dir,
		'-o', out_dir,
		'-n', str(num_processes),
		'-l', '0'*(3-len(hex(batch_num-1)[2:])) + hex(batch_num-1)[2:]
	]
	if until is not None:
		worldviews_command.extend(['-u', until])
	if skip is not None:
		worldviews_command.extend(['-s', skip])
	if only is not None:
		only = ','.join(only)
		worldviews_command.extend(['-j', only])
	
	subprocess.check_call(worldviews_command)


if __name__ == '__main__':
	preprocess_gigaword()
