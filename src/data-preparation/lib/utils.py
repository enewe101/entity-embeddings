import sys
sys.path.append('..')
import math
import json
import os
import re
import time
from multiprocessing import Process
import t4k
from SETTINGS import DATA_DIR


def format_elapsed_string(seconds_elapsed):
	'''
	Converts the given number of seconds to a string showing hours, minutes,
	and seconds.
	'''

	hours = int(seconds_elapsed / 3600)
	minutes = int((seconds_elapsed - 3600 * hours) / 60)
	seconds = int((seconds_elapsed - 3600 * hours - 60 * minutes))

	elapsed_string = '%d : %d : %d' % (hours, minutes, seconds)

	return elapsed_string


def sluggify(string):
	sluggify_pattern = re.compile('\W+')
	return sluggify_pattern.sub('-', string)




def relative_to_data(path):

	# ensure that path is in DATA_DIR
	if DATA_DIR not in path:
		raise ValueError('relative_to_data: supplied path is not in DATA_DIR')

	# remove the portion of path up to data_dir
	path = path.replace(DATA_DIR, '')

	# remove the leftover leading slash
	return path[1:]


def lsfiles(path, whitelist='^.*$', blacklist='^$'):
	whitelist = re.compile(whitelist)
	blacklist = re.compile(blacklist)

	items = os.listdir(path)
	files = [
		f for f in items
		if os.path.isfile(os.path.join(path, f))
		and whitelist.match(f) and not blacklist.match(f)
	]
	return files


class MyPool(object):

	CHECK_PROCESSES_DELAY = 0.5

	def __init__(self, num_processes):
		self.num_processes = num_processes
		self.processes = []


	def start_next_worker(self, target, args):
		self.wait_if_max_processes()
		p = Process(target=target, args=args)
		self.processes.append(p)
		p.start()


	def wait_if_max_processes(self):

		# If necessary, wait for a process to finish
		while len(self.processes) >= self.num_processes:
			for process in [p for p in self.processes]:
				if not process.is_alive():
					self.processes.remove(process)
			time.sleep(self.CHECK_PROCESSES_DELAY)


	def join(self):
		for process in self.processes:
			process.join()


def read_batch(batch_num):
	'''
	Parses line <batch_num> from the batches.txt file, and returns the
	configurations for that batch. 
	'''

	# Get info for <batch_num>
	batch_path = os.path.join(DATA_DIR, 'batches.txt')
	batch_jsons = open(batch_path).readlines()
	batch = json.loads(batch_jsons[batch_num-1])

	print 'batch-def:',batch_jsons[batch_num-1]

	# Absolutize paths (they are assumed relative to DATA_DIR)
	in_dir = os.path.join(DATA_DIR, batch['in_dir'])
	out_dir = os.path.join(DATA_DIR, batch['out_dir'])
	until = batch.get('until', None)
	only = batch.get('only', None)
	skip = batch.get('skip', None)

	return in_dir, out_dir, until, skip, only


class ReverseUserLookup(object):

	def __init__(self, path):
		self.path = path
		self.lookup = self.make_dict(path)

	def make_dict(self, path):
		lookup = {}
		tracker = t4k.ProgressTracker(path)
		for url, url_spec in tracker:
			try:
				user_id = url_spec['user_id']
				doc_hash = os.path.basename(url_spec['web-text-path'])
			except KeyError:
				continue
			else:
				lookup[doc_hash] = user_id

		return lookup

	def __getitem__(self, doc_hash):
		return self.lookup[doc_hash]



