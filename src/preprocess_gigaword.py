#!/usr/bin/env python
'''
Preprocess gigaword, a large corpus of news stories.
1) Only take articles of type "story"
2) Detect and eliminate directives to editors, which appear as a series of
	all-caps words.
3) Split the data into separate files for each article.
	a) Only put the plain text into those files
	b) Extract metadata for articles and Use a progress tracker to 
		associate metadata to the article files.
4) Convert SGML entities &amp; &gt; &lt;
5) Eliminate spanish articles and articles misclassified as being of type
	story using the corrections provided as metadata (spanish.file-doc.map
	and other.file-doc.map)
6) Keep the headline?
'''

import hashlib
import t4k
import sys
import re
import gzip
import os
from SETTINGS import GIGAWORD_DIR, DATA_DIR


GIGAWORD_PLAINTEXT_DIR = os.path.join(DATA_DIR, 'gigaword-plaintext')
DOCS_TO_SKIP_FNAMES = ['other.file-doc.map', 'spanish.file-doc.map']
DOC_ID_EXTRACTOR = re.compile('id="([^"]*)"')
DOC_TYPE_EXTRACTOR = re.compile('type="([^"]*)"')
AMPERSAND = re.compile('&amp;')
LEFT_ANGLE_BRACKET = re.compile('&lt;')
RIGHT_ANGLE_BRACKET = re.compile('&gt;')


def read_docs_to_skip():
	'''
	Some of the documents in gigaword were misclassified as stories
	(as opposed to other newswire document types) or as English (when
	they were in fact spanish).  These misclassifications are listed in a 
	metadata files which this function reads
	'''
	docs_to_skip = set()
	for fname in DOCS_TO_SKIP_FNAMES:
		path = os.path.join(GIGAWORD_DIR, 'docs', fname)
		docs_to_skip.update([l.strip().split()[1] for l in open(path)])

	return docs_to_skip


def escape_sgml_entities(text):
	'''
	The characters "&", "<" and ">" are escaped in the gigaword dataset.
	Unescape them.
	'''
	text = AMPERSAND.sub('&', text)
	text = LEFT_ANGLE_BRACKET.sub('<', text)
	text = RIGHT_ANGLE_BRACKET.sub('>', text)

	return text


def gigaword_data_file_paths():
	gigaword_data_dir = os.path.join(GIGAWORD_DIR, 'data')
	for dirname in os.listdir(gigaword_data_dir):
		news_agency_dir = os.path.join(gigaword_data_dir, dirname)
		if not os.path.isdir(news_agency_dir):
			continue
		for fname in os.listdir(news_agency_dir):
			if fname.startswith('.'):
				continue
			fpath = os.path.join(news_agency_dir, fname)
			yield fpath


def gigaword_raw_documents(fpath):
	'''
	Read a gigaword file, and yield portions of the file (as lists of lines)
	that correspond to single raw documents.
	'''
	doc_lines = []
	for line in gzip.open(fpath):
		doc_lines.append(line)
		if line.startswith('</DOC>'):
			yield doc_lines
			doc_lines = []


def parse_gigaword_doc(doc_lines):
	doc_id = DOC_ID_EXTRACTOR.search(doc_lines[0]).group(1)
	doc_type = DOC_TYPE_EXTRACTOR.search(doc_lines[0]).group(1)
	headlines = []
	datelines = []
	textlines = []
	currently_reading = None
	for line in doc_lines:

		line = line.strip()

		# Keep track of what part of document we are in based on opening
		# and closing tags (all tags appear on their own line)
		if line in ('</TEXT>', '</DATELINE>', '</HEADLINE>'):
			currently_reading  = None
			continue
		if line == '<TEXT>':
			currently_reading  = 'text' 
			continue
		if line == '<DATELINE>':
			currently_reading  = 'dateline' 
			continue
		if line == '<HEADLINE>':
			currently_reading  = 'headline' 
			continue

		# Associate the lines to the correct part of the document
		if currently_reading == 'headline':
			headlines.append(line)
		elif currently_reading == 'dateline':
			datelines.append(line)
		elif currently_reading == 'text':

			# Don't include tags in text, but do separate paragraphs by
			# two newlines
			if line == '<P>':
				continue
			if line == '</P>':
				textlines.append('')

			# Skip directives to editors, which appear in all caps
			if line.upper() == line:
				continue

			textlines.append(line)

	return {
		'id': doc_id,
		'type': doc_type,
		'headline': escape_sgml_entities('\n'.join(headlines)),
		'dateline': escape_sgml_entities('\n'.join(datelines)),
		'text': escape_sgml_entities('\n'.join(textlines))
	}


def preprocess_gigaword():

	# Make the output directory if it doesn't exist
	if not os.path.exists(GIGAWORD_PLAINTEXT_DIR):
		os.makedirs(GIGAWORD_PLAINTEXT_DIR)

	# We'll skip these misclassified documents
	docs_to_skip = read_docs_to_skip()

	# Use trackers to keep track of which files have been processed
	file_tracker = t4k.ProgressTracker(
		os.path.join(DATA_DIR, 'gigaword_files'))

	# Use a tracker to keep track of the documents extracted
	doc_tracker = t4k.ProgressTracker(
		os.path.join(DATA_DIR, 'gigaword_docs'))
	doc_tracker.hold()

	# Process each file, extracting plain text for all the news articles
	i = 0
	for fpath in gigaword_data_file_paths():

		# use a tracker to keep track of whether the file has been processed
		if file_tracker.check_or_add(fpath):
			print 'skipping %s' % fpath
		else:
			print 'processing %s' % fpath

		# Get the plain text for each document in the file
		for raw_doc in gigaword_raw_documents(fpath):

			doc = parse_gigaword_doc(raw_doc)

			# We only use news stories
			if doc['type'] != 'story':
				continue

			# If the document is among the misclassified documents, skip it
			if doc['id'] in docs_to_skip:
				print '\tskipping misclassified: %s' % doc['id']

			# Show progress.  Periodically sync the dock_tracker
			if i % 1000 == 0:
				print '\t.'
				doc_tracker.unhold()
				doc_tracker.hold()

			# The output document's file name is made from hash of its id
			doc_hash = hashlib.sha1(doc['id']).hexdigest()[:16]
			out_fname = doc_hash + '.txt'
			out_path = os.path.join(GIGAWORD_PLAINTEXT_DIR, out_fname)

			# Save the document in its own file
			open(out_path, 'w').write(doc['text'])

			# Keep the document's metadata in the tracker
			doc_tracker.check_or_add(out_fname)
			doc_tracker.set(out_fname, 'from-file', os.path.basename(fpath))
			doc_tracker.set(out_fname, 'id', doc['id'])
			doc_tracker.set(out_fname, 'headline', doc['headline'])
			doc_tracker.set(out_fname, 'dateline', doc['dateline'])

		# Mark this file as done
		file_tracker.markdone(fpath)

	doc_tracker.unhold()


if __name__ == '__main__':
	preprocess_gigaword()

