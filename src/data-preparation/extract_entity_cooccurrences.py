from SETTINGS import DATA_DIR
import t4k
from Queue import Empty
from multiprocessing import Queue, Process, Pipe
import re
import itertools as itools
import os
import lib.utils as utils
import time
from linguini import SimpleTask
from lib.annotated_article import AnnotatedArticle

DONE = 0
COOCCURRENCE_DIR = os.path.join(DATA_DIR, 'cooccurrence')

def extract_entity_cooccurrences(num_processes, in_dir, out_dir):
	'''
		For the set of files defined by the inputs, look at every sentence,
		and, for every pair of entities that cooccurr, extract the 
		intervening text using the approach used in the PATTY system.  In 
		other words, take all tokens that form the the shortest path in the 
		dependency tree, then add in any adverbial or adjectival modifiers.
	'''

	start = time.time()
	print '\n\t*** starting ExtractInterveningText ***\n'

	# Work out the relevant input/output paths
	aida_dir = in_dir
	batch_dir_name = os.path.basename(in_dir)

	# Make sure the cooccurrence_dir exists
	if not os.path.exists(COOCCURRENCE_DIR):
		os.makedirs(COOCCURRENCE_DIR)

	# Pull together all the articles, put them in batches
	parsed_dir = os.path.join(out_dir, 'stanford-parsed')
	stanford_paths = list(t4k.ls(parsed_dir, absolute=True))
	num_batches = min(len(stanford_paths), num_processes * 2)
	stanford_path_batches = t4k.group(stanford_paths, num_batches)


	# Make a couple pipes for communicating with processes
	write_queue = Queue()
	status_receiver, status_sender = Pipe()

	# Start a process for writing results
	writer = Process(
		target=write, args=(write_queue, status_receiver, batch_dir_name)
	)
	writer.start()

	# Assemble arguments lists to dispatch the work to a pool
	args_lists = [
		(stanford_path_batch, out_dir, aida_dir, write_queue)
		for stanford_path_batch in stanford_path_batches
	]

	# Dispatch the work to a pool
	pool = utils.MyPool(num_batches)
	for args in args_lists:
		pool.start_next_worker(target=process_articles, args=args)

	pool.join()
	status_sender.send(DONE)
	writer.join()

	# Record the amount of time spent on this task
	elapsed = time.time() - start
	elapsed_string = utils.format_elapsed_string(elapsed)
	open(os.path.join(out_dir, 'timer.txt'), 'a').write(
		'ExtractInterveningText: %s\n' % elapsed_string)


def write(write_receiver, status_receiver, batch_dir_name):

	out_f = open(os.path.join(
		COOCCURRENCE_DIR, '%s.tsv' % batch_dir_name
	), 'w')

	# Keep looking for work to do until a signal is received to quit.
	while True:
		print 'waiting for work'

		# As long as there's text in the queue, write it to file
		while not write_receiver.empty():
			try:
				text_to_write = write_receiver.get(1)
			except Empty:
				continue

			out_f.write(text_to_write)

		# If the queue is empty, check to see if processing is all done
		# This is the only way the writer will return
		if status_receiver.poll(1):
			state = status_receiver.recv()
			if state == DONE:
				return

		# Prevent rapid in case the queue is empty for awhile
		time.sleep(1)



def process_articles(
	stanford_path_batch,
	out_dir,
	aida_dir,
	write_queue
):
	print '\tstarting a worker...'
	for stanford_path in stanford_path_batch:
		process_article(stanford_path, out_dir, aida_dir, write_queue)

	print '\ta worker finished...'


def process_article(
	stanford_path,
	out_dir,
	aida_dir,
	write_queue
):

	# Work out paths to input files
	original_fname = os.path.basename(stanford_path)[:-len('.xml')]
	aida_fname = '%s.json' % original_fname
	aida_path = os.path.join(aida_dir, aida_fname)
  
	# Read parse xml and linkage json; open as annotated article
	stanford_xml = open(stanford_path).read()
	try:
		aida_json = open(aida_path).read()
	except IOError:
		print (
			"AIDA output for this article (%s) doesn't exsist, skipping"
			% aida_path
		)
		return

	article = AnnotatedArticle(
		stanford_xml, aida_json, dependencies='basic')

	# Process all sentences to extract intervening texts
	# Note the first null sentence is skipped
	lines = ''
	for sentence in article.sentences[1:]:

		# Get the linked entity identifiers and spans
		kbIdentifier_spans = [
			((m['kbIdentifier'], str(m['start']), str(m['end'])))
			for m in sentence['mentions'] if 'kbIdentifier' in m
		]

		# Sort the entity identifiers based on start point of their spans
		kbIdentifier_spans.sort(key=lambda x: x[1])

		# Turn each span tuple into a string
		kbIdentifier_spans = [','.join(s) for s in kbIdentifier_spans]

		num_unique_kbids = len(set([
			m['kbIdentifier'] for m in sentence['mentions'] 
			if 'kbIdentifier' in m
		]))

		# Note: this skips the initial null root token.
		tokens = [t['word'] for t in sentence['tokens'][1:]]

		token_string = ' '.join(tokens).encode('utf8')
		kbIdentifier_string = '\t'.join(kbIdentifier_spans).encode('utf8')
		fname_to_write = original_fname.encode('utf8')
		lines += (
			str(num_unique_kbids) + '\t' + token_string + '\t' 
			+ kbIdentifier_string + '\t' + fname_to_write + '\n'
		)

	# Send lines to the writing process
	write_queue.put(lines)



