import os
import t4k
import sys
sys.path.append('../lib')

from SETTINGS import GIGAWORD_DIR, WORDNET_INDEX_PATH

import extract_features as e

def do_extract_features():

	gigaword_archive_paths = t4k.ls(
		os.path.join(GIGAWORD_DIR, 'data'),
		match=r'\.tgz'
	)
	for path in gigaword_archive_paths:
		try:
			e.extract_and_save_features(
				path, WORDNET_INDEX_PATH, untar=True, limit=None
			)
		except KeyboardInterrupt:
			raise
		except Exception as exc:
			print 'problem with %s: %s' % (os.path.basename(path), str(exc))


if __name__ == '__main__':
	do_extract_features()

