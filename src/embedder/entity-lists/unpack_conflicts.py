import re
import itertools as itools
import sys


SPLIT_COALITIONS = re.compile(' |,')


def unpack_conflicts(
	input_conflicts_fname,
	output_coalitions_fname,
	output_antagonists_fname
):

	conflicts_f = open(input_conflicts_fname)
	coalitions_f = open(output_coalitions_fname, 'w')
	antagonists_f = open(output_antagonists_fname, 'w')

	coalition_pairs = set()
	antagonist_pairs = set()

	for i, line in enumerate(conflicts_f):

		# skip blank lines
		if line.strip() == '':
			continue

		# Get out all the countries involved in this conflict, and split 
		# them into their respective coalitions
		side1, side2 = line.strip().split(';')
		side1 = [
			'YAGO:%s' % c for c in SPLIT_COALITIONS.split(side1.strip())
		]
		side2 = [
			'YAGO:%s' % c for c in SPLIT_COALITIONS.split(side2.strip())
		]

		# Generate coalition pairs and antagonist pairs
		add_coalition_pairs = (
			list(itools.combinations(side1, 2))
			+ list(itools.combinations(side2, 2))
		)
		add_antagonist_pairs = list(itools.product(side1, side2))

		coalition_pairs.update(add_coalition_pairs)
		antagonist_pairs.update(add_antagonist_pairs)

	# Remove pairs that appeared both as a coalition and as antagonists
	ambiguous_pairs = coalition_pairs & antagonist_pairs
	print ambiguous_pairs
	coalition_pairs -= ambiguous_pairs
	antagonist_pairs -= ambiguous_pairs

	# Add them to the output files
	for coalition_pair in coalition_pairs:
		coalitions_f.write('%s\t%s' % coalition_pair + '\n')

	for antagonist_pair in antagonist_pairs:
		antagonists_f.write('%s\t%s' % antagonist_pair + '\n')


if __name__ == '__main__':
	input_conflicts_fname = sys.argv[1]
	output_coalitions_fname = sys.argv[2]
	output_antagonists_fname = sys.argv[3]

	unpack_conflicts(
		input_conflicts_fname,
		output_coalitions_fname,
		output_antagonists_fname
	)



		
