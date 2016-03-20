import t4k
import re
import itertools as itools
import os
import lib.utils as utils
import time
from linguini import SimpleTask
from lib.annotated_article import AnnotatedArticle
from lib.shared_settings import YAGO_DIR


MODIFIER_PATTERN = re.compile('.*mod|det|neg|case|aux.*')


def extract_parsed_entity_text(num_rocesses, in_dir, out_dir):
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
	intervening_text_dir = os.path.join(
		out_dir, 'patty-intervening-text')

	# Make sure the intervening_text_dir exists
	if not os.path.exists(intervening_text_dir):
		os.makedirs(intervening_text_dir)

	# Pull together all the articles, put them in batches
	parsed_dir = os.path.join(out_dir, 'stanford-parsed')
	stanford_paths = list(t4k.ls(parsed_dir, absolute=True))
	num_batches = min(len(stanford_paths), num_processes * 2)
	stanford_path_batches = t4k.group(stanford_paths, num_batches)

	# Assemble arguments lists to dispatch the work to a pool
	args_lists = [
		(stanford_path_batch, out_dir, aida_dir)
		for stanford_path_batch in stanford_path_batches
	]

	for arglist in args_lists:
		print arg_list
	## Dispatch the work to a pool
	#pool = utils.MyPool(num_batches)
	#for args in args_lists:
	#	pool.start_next_worker(target=process_articles, args=args)
	#pool.join()

	## Record the amount of time spent on this task
	#elapsed = time.time() - start
	#elapsed_string = utils.format_elapsed_string(elapsed)
	#open(os.path.join(out_dir, 'timer.txt'), 'a').write(
	#	'ExtractInterveningText: %s\n' % elapsed_string)


def process_articles(
	stanford_path_batch,
	out_dir,
	aida_dir
):
	for stanford_path in stanford_path_batch:
		process_article(stanford_path, out_dir, aida_dir)


def process_article(
	stanford_path,
	out_dir,
	aida_dir
):

	print '\tstarting a worker...'

	# Work out paths to input files
	original_fname = os.path.basename(stanford_path)[:-len('.xml')]
	intervening_text_dir = os.path.join(out_dir, 'patty-intervening-text')
	aida_fname = '%s.json' % original_fname
	aida_path = os.path.join(aida_dir, aida_fname)
	out_fname = '%s.tsv' % original_fname
	out_path = os.path.join(intervening_text_dir, out_fname)

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

	# Open a writer for co-occurence output
	f = open(out_path, 'w')
	writer = t4k.UnicodeTsvWriter(f)

	# Process all sentences to extract intervening texts
	for sentence in article.sentences:

		# Each sentence can yield intervening text 
		# for multiple entity pairs
		intervening_texts, errors = process_sentence(
			sentence, stanford_path)

		writer.writerows(intervening_texts)

	print '\ta worker finished...'


def process_sentence(sentence, stanford_path):

	# at this stage we should select only aida-linked entities
	intervening_texts = []
	errors = []

	# get mentions that were linked by AIDA, and that have an entry in YAGO
	aida_linked_mentions = [
		m for m in sentence['mentions']
		if 'kbIdentifier' in m 
	]

	for m1, m2 in itools.combinations(aida_linked_mentions, 2):

		# don't process mentions that refer to the same entity
		if m1['kbIdentifier'] == m2['kbIdentifier']:
			continue

		source = m1['head']
		target = m2['head']

		if source is None or target is None:
			continue

		# if the source and target are improperly ordered,
		# swap'em
		if source['id'] > target['id']:
			source, target = target, source
			m1, m2 = m2, m1

		# This is the old way of getting the intervening text.  It tries
		# to emulate how patty gets intervening text, based on the shortest
		# path in the dependency tree.  Based on conversations with the 
		# creator of patty, this is not actually necessary.
		#
		#tokens = sentence.shortest_path(source, target)

		## in rare cases,
		## it is impossible to resolve shortest path
		#if tokens is None:

		#	# log the event and continue
		#	errors.append(('unable to resolve shortest path',
		#		stanford_path, str(sentence['id']), get_mention_label(m1),
		#		get_mention_label(m2), sentence.as_string()
		#	))
		#	continue

		## include modifiers even if not on shortest path
		#tokens = augment_with_modifiers(tokens)

		## ensure all tokens of the mention are included
		## (this is actually to ensure clean removal of the 
		## mentions in the next step. Removal is easier if *all*
		## mentions' tokens are guaranteed to be there)
		#tokens = m1['tokens'] + tokens + m2['tokens']

		## sort the tokens according order of occurence in sentence
		## remove duplicate tokens
		## remove tokens that are not between the mentions
		## as well tokens that are part of the mentions themselves
		#start = m1['tokens'][-1]
		#end = m2['tokens'][0]
		#intervening_tokens = sort_and_trim(tokens, start, end)

		# Get all of the tokens between m1 and m2
		intervening_tokens = sentence['tokens'][m1['end']+1:m2['start']]

		# don't include cooccurrences that have no 
		# intervening text
		if len(intervening_tokens) < 1:
			continue

		# include the actual text of the mention, and
		# the that of the representative mention for the entity
		#m1_label = get_mention_label(m1)
		#m2_label = get_mention_label(m2)
		m1_label = ' '.join([t['word'] for t in m1['tokens']])
		m2_label = ' '.join([t['word'] for t in m2['tokens']])

		# If m1 ends with a posessive inflection, move that into the 
		# intervening text (patty patterns contain the posessive "s" from
		# the end of the earlier mention when present.
		# This step was normally accomplished in `sort_and_trim()`
		if m1_label.endswith("'s"):
			intervening_tokens = [{'word':'s', 'pos':'POS'}] + intervening_tokens

		# Get the corresponding wordnet types for the entities (from YAGO)
		wordnet_types_1 = m1['types']
		wordnet_types_2 = m2['types']

		# keep surface form and POS tags for the intervening 
		# text
		intervening_text = ' '.join([
			t['word'].lower() + '||' + t['pos'] for t in intervening_tokens
		])

		# return the row instead of writing
		intervening_texts.append((
			intervening_text, 
			m1_label,
			m2_label,
			m1['kbIdentifier'],
			m2['kbIdentifier'],
			'|'.join(wordnet_types_1),
			'|'.join(wordnet_types_2),
			sentence.as_string(),
			str(sentence['id'])
		))

	return intervening_texts, errors



def get_mention_label(mention_obj):
	'''
		returns a string that represents the mention text by including 
		the actual surface form of the mention, followed by the surface
		form of the representative mention e.g.:

			`the president [Barack Obama]`
	'''
	label = ' '.join([t['word'] for t in mention_obj['tokens']])
	label += ' [' + ' '.join(
		[t['word'] for t in 
			mention_obj['reference']['representative']['tokens']
		]) + ']'

	return label


def sort_and_trim(tokens, source, target):

	# sort tokens by order in sentence
	tokens.sort(key=lambda x: x['id'])

	# eliminate duplicates
	tokens_no_dupes = []
	last_token = None
	for token in tokens:
		if token is not last_token:
			tokens_no_dupes.append(token)
		last_token = token

	tokens = tokens_no_dupes

	# eliminate tokens that don't lie between source and target tokens
	token_ids = [t['id'] for t in tokens]
	start = token_ids.index(source['id'])
	end = token_ids.index(target['id'])

	# however, if first mention ends with possessive `'s`, keep an `s`
	if source['word'] == "'s":
		start -= 1
		source['word'] = "s"

	# this actually removes the source and target too
	tokens = tokens[start+1:end]

	return tokens


def augment_with_modifiers(tokens):

	return_tokens = []
	ptr = 0 

	while ptr < len(tokens):

		token = tokens[ptr]
		ptr += 1

		return_tokens.append(token)
		for relation, child in token.get_children():
			if MODIFIER_PATTERN.match(relation):
				return_tokens.append(child)

	return return_tokens


