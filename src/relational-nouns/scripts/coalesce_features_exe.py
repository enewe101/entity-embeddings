import sys
import t4k
sys.path.append('../lib')
import extract_features as ef
from SETTINGS import DATA_DIR
import os
import utils


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


def coalesce_batch(batch_num):
    """
    Coalesce some of the feature extracts (a batch of 50 of them).  
    batch_num determines which 50 extracts will be coalesced.
    Do some light pruning.
    """

    in_dir = os.path.join(DATA_DIR, 'relational-noun-features-lexical-wordnet')
    start = 50*batch_num
    stop = 50*(batch_num+1)
    feature_dirs = t4k.ls(
        in_dir, absolute=True, match='/[0-9a-f]{3,3}$', files=False
    )[start:stop]
    out_dir = os.path.join(
        in_dir, 'accumulated50-min_token_5-min_feat100-%d'%batch_num)

    ef.coalesce_features(
        out_dir=out_dir,
        min_token_occurrences=2,
        min_feature_occurrences=100,
        vocabulary=utils.read_wordnet_index(),
        feature_dirs=feature_dirs
    )

if __name__ == '__main__':
    #coalesce_wordnet_only()
    #coalesce()
    coalesce_batch(5)

