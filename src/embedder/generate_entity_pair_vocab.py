'''
Calculate the top-pmi scoring tokens seen between each entity pair, relative
to the unigram frequency, and relative to the frequency of tokens seen 
between all pairs.
'''

import numpy as np
from dataset_reader import Relation2VecDatasetReader
import json
import re
import sys
import os
sys.path.append('..')
from SETTINGS import DATA_DIR, COOCCURRENCE_DIR, DICTIONARY_DIR
import itertools as itools

NUM_TO_KEEP = 100
FNAMES = [hex(i)[2:].zfill(3) + '.tsv' for i in range(256)]
FPATHS = [os.path.join(COOCCURRENCE_DIR, fname) for fname in FNAMES]

TEST_DICTIONARY_DIR = os.path.join(DATA_DIR, 'test-dictionaries')

def generate_entity_pair_vocab(
		pair_pmi_fname,
		unigram_pmi_fname,
		load_dictionary_dir,
		files=[], 
		directories=[], 
		skip=[],
	):
	'''
	Reads through the entire corpus, tracking the occurrence of 
	words in the presence of entity pairs, and then determines the
	PMI of the word given the entity pair, relative to the generic
	entity pair context, or the general unigram context
	'''

	pair_f = open(pair_pmi_fname, 'w')
	unigram_f = open(unigram_pmi_fname, 'w')

	reader = Relation2VecDatasetReader(
		#files=FPATHS,	# DEBUG!
		files=files,
		directories=directories,
		skip=skip,
		load_dictionary_dir=DICTIONARY_DIR,	
		verbose=True,
	)

	entity_pair_vocabs = {'all-entities':{'all-tokens':0}}

	for fname in reader.generate_filenames():
		for tokens, entity_spans in reader.parse(fname):
			
			# Sentences with less than two entities can't be used for
			# learning
			if len(entity_spans) < 2:
				continue

			token_ids = reader.context_dictionary.get_ids(tokens)

			# loop over all pairs of entities
			for e1, e2 in itools.combinations(entity_spans, 2):

				# convert entities into ids
				e1_id, e2_id = reader.entity_dictionary.get_ids([e1, e2])

				# Generate a unique string key for this pair of entities.
				# Mind the fact that pairs can occur in different orders.
				if e1_id > e2_id:
					e2_id, e1_id = e1_id, e2_id
				entity_key = str((e1_id, e2_id))

				# Get (or make) the sub-vocabulary for this particular
				# entity pair
				try:
					entity_pair_vocab = entity_pair_vocabs[entity_key]
				except KeyError:
					entity_pair_vocab = {'all-tokens':0}
					entity_pair_vocabs[entity_key] = entity_pair_vocab

				# Get the tokens between the entities as context
				context_indices = reader.find_tokens_between_closest_pair(
					entity_spans[e1], entity_spans[e2]
				)
				context = [token_ids[i] for i in context_indices]
				for token_id in context:

					# Increment the total counts
					entity_pair_vocab['all-tokens'] += 1
					entity_pair_vocabs['all-entities']['all-tokens'] += 1

					# Increment the count for the context word in the
					# vocabulary of this particular entity pair
					try:
						entity_pair_vocab[token_id] += 1
					except KeyError:
						entity_pair_vocab[token_id] = 1

					# Increment the collective count maintained across all
					# entity pairs
					try:
						entity_pair_vocabs['all-entities'][token_id] += 1
					except KeyError:
						entity_pair_vocabs['all-entities'][token_id] = 1

	# write these vocabs to disc
	#f.write(json.dumps(entity_pair_vocabs, indent=2))

	# for each entity pair, we calculate the pointwise mutual information
	# for each token that appears between the entities.  We're interested
	# in seeing those tokens with high PMI scores, that is, those tokens 
	# that are much more likely to arise between the entities.  Therefore,
	# we need not consider tokens that were less common between the entities
	dictionary = reader.context_dictionary
	e_dictionary = reader.entity_dictionary
	i = 0
	for entity_key in entity_pair_vocabs: 

		if i % 1000 == 0:
			print i * 100 / float(len(entity_pair_vocabs))
		i += 1

		pair_pmis = []
		unigram_pmis = []

		# Skip the special entry used to capture the frequency of tokens
		# among all entity pairs
		#if entity_key == 'all-entities':
		#	continue

		unigram_total = float(entity_pair_vocabs[entity_key]['all-tokens'])
		counts = entity_pair_vocabs[entity_key]
		all_pair_total = float(
			entity_pair_vocabs['all-entities']['all-tokens'])
		all_pair_counts = entity_pair_vocabs['all-entities']

		# Consider each token seen between this pair of entities
		for token_id in counts:

			# Skip the special entry used to capture the frequency of 
			# all tokens for this entity pair.
			if token_id == 'all-tokens':
				continue

			# Get the frequency of this token among the entities, and its
			# frequency at large (unigram frequency), and its frequency
			# among all entity pairs
			pair_frequency = counts[token_id] / unigram_total
			unigram_frequency = dictionary.get_probability(token_id)

			all_pair_frequency = all_pair_counts[token_id] / all_pair_total

			# If the token is more common between the pair of entities than
			# the unigram frequency, compute and store it's pmi
			if pair_frequency > unigram_frequency:
				unigram_pmi = np.log(pair_frequency / unigram_frequency)
				unigram_pmis.append((token_id, unigram_pmi))

			# Do similarly, but with respect to the frequency among all
			# entity pairs.
			if pair_frequency > all_pair_frequency:
				pair_pmi = np.log(pair_frequency / all_pair_frequency)
				pair_pmis.append((token_id, pair_pmi))

		# Keep (at most) the 50 highest-pmi tokens (in descending order)
		pair_pmis.sort(key=lambda x: x[1], reverse=True)
		pair_pmis = pair_pmis[:NUM_TO_KEEP]
		unigram_pmis.sort(key=lambda x: x[1], reverse=True)
		unigram_pmis = unigram_pmis[:NUM_TO_KEEP]

		# Translate from token_ids to the actual tokens (words), and
		# put each entry into a printable string format
		pair_pmis = [
			'%s:::%f' % (dictionary.get_token(t), p) 
			for t, p in pair_pmis
		]

		unigram_pmis = [
			'%s:::%f' % (dictionary.get_token(t), p) 
			for t, p in unigram_pmis
		]

		# Convert the entity_pair into a string (with each entity separated
		# by a tab.  However, we are also doing the calculation for the
		# *all-entities* entry (which accumulates counts accross all 
		# entities, so handle that case specially
		if entity_key == 'all-entities':
			entity_pair_str = 'all-entities\tall-entities'
		else:
			e1_id, e2_id = [int(e) for e in entity_key[1:-1].split(',')]
			e1 = e_dictionary.get_token(e1_id)
			e2 = e_dictionary.get_token(e2_id)
			entity_pair_str = e1 + '\t' + e2

		# Write the top tokens and their pmis for this entity pair, 
		# both for the calculation relative to unigram frequency and 
		# relative to all-entity-pair frequency.
		unigram_f.write(
			entity_pair_str + '\t' + '\t'.join(unigram_pmis) + '\n')
		pair_f.write(
			entity_pair_str + '\t' + '\t'.join(pair_pmis) + '\n')


if __name__ == '__main__':
	skip = [re.compile('README'), re.compile('test')]
	generate_entity_pair_vocab(
		unigram_pmi_fname=os.path.join(
			DATA_DIR, 'entity-pair-unigram-pmi.tsv'),
		pair_pmi_fname=os.path.join(
			DATA_DIR, 'entity-pair-pair-pmi.tsv'),
		directories=[COOCCURRENCE_DIR],
		skip=skip,
		load_dictionary_dir=DICTIONARY_DIR,
	)

