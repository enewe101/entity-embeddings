import sys
sys.path.append('..')
import os
from collections import defaultdict
import t4k
from SETTINGS import DATA_DIR

USUALLY_RELATIONAL = '+'
OCCASIONALLY_RELATIONAL = '0'
NEVER_RELATIONAL = '-'


def interpret_annotations(crowdflower_results):
	"""
	Given a CrowdflowerResult object (obtained from passing a crowdflower 
	json report into t4k.CrowdFlowerResult), interpret the label for given 
	words (either "Usually Relational", "Occasionally Relational", or "Almost
	Never Relational".
	"""

	word_labels = defaultdict(dict)

	for result in crowdflower_results:

		# Work out the word, its sampling source(s), and its label
		word = result['data']['token']
		sources =  result['data']['source'].split(':')
		label = get_label(result)

		# Store accordingly
		for source in sources:
			word_labels[source][word] = label

	return word_labels


def transcribe_labels(results_fname):
	"""
	Read the results stored under the crowdflower subdir of the data dir
	named by results_fname, and interpret the annotations into labels.  Then,
	write those labels out into the relational-nouns subdir of the data dir
	within a subsubdir that has the same name as the results_fname.  Record teh
	results in three separate files -- based on what source the word was drawn
	from.
	"""

	# Work out paths
	results_path = os.path.join(DATA_DIR, 'crowdflower', results_fname)
	result_fname_no_ext = results_fname.rsplit('.', 1)[0]
	labels_dir = os.path.join(DATA_DIR, 'relational-nouns', result_fname_no_ext)
	t4k.ensure_exists(labels_dir)

	# Read in the results, and interpret labels
	crowdflower_results = t4k.CrowdflowerResults(
		results_path, lambda x:x['data']['token'])
	word_labels = interpret_annotations(crowdflower_results)

	# Write labels to disk, with words coming from different sources put into
	# different files.
	for source in word_labels:
		source_label_file = open(os.path.join(labels_dir, source + '.tsv'), 'w')
		for word, label in word_labels[source].iteritems():
			source_label_file.write(word + '\t' + label + '\n')


def get_label(result):
	"""
	Gets the correct label based on the annotations of a single word.  If a
	strict majority of annotations indicate that the word is either "usually
	relational" or "almost never relational", then the corresponding label is
	taken.  Otherwise, for ambiguous cases, "occasionally relational" is taken.
	"""
	# If there is a clear majority for "usually relational" or "almost
	# never relational" then take the correponding label, otherwise we
	# default to "occasionally relational".
	label = OCCASIONALLY_RELATIONAL
	if len(result['mode']['response']) < 2:
		if result['mode']['response'][0] == 'usually relational':
			label = USUALLY_RELATIONAL
		elif result['mode']['response'][0] == 'almost never relational':
			label = NEVER_RELATIONAL

	return label


