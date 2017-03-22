import os
import t4k
import sys
sys.path.append('../lib')
from SETTINGS import GIGAWORD_DIR
import extract_features as e
import utils


def do_extract_all_features(vocabulary):

	gigaword_archive_paths = t4k.ls(
		os.path.join(GIGAWORD_DIR, 'data'),
		match=r'\.tgz'
	)
	for path in gigaword_archive_paths:
		try:
			e.extract_all_features(
                path, untar=True, limit=None, vocabulary=vocabulary
            )
		except KeyboardInterrupt:
			raise
		except Exception as exc:
			print 'problem with %s: %s' % (os.path.basename(path), str(exc))


def extract_all_featurea_for_wordet_nouns():
    do_extract_all_features(vocabulary=utils.read_wordnet_index())


def extract_all_features_for_all_nouns():
    do_extract_all_features(vocabulary=None)


if __name__ == '__main__':
	#extract_all_featurea_for_wordet_nouns()
    extract_all_features_for_all_nouns()

