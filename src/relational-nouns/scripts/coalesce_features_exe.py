import sys
sys.path.append('../lib')
import extract_features as ef
from SETTINGS import DATA_DIR
import os

def coalesce_wordnet_only():
	ef.coalesce_features(
        'accumulated',
        os.path.join(DATA_DIR, 'relational-noun-features-wordnet-only')
    )

def coalesce():
	ef.coalesce_features(
        'accumulated',
        os.path.join(DATA_DIR, 'relational-noun-features')
    )

if __name__ == '__main__':
    #coalesce_wordnet_only()
    coalesce()


