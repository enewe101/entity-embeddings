import numpy as np
import sys
sys.path.append('..')
from t4k import UnigramDictionary
from nltk.corpus import wordnet, wordnet_ic
import utils as u

INFORMATION_CONTENT_FILE = 'ic-treebank-resnik-add1.dat'
LEGAL_SIMILARITIES = [
    'jcn', 'wup', 'res', 'path', 'lin', 'lch' 
]
LEGAL_SYNTACTIC_SIMILARITIES = ['hand_picked', 'dependency', 'baseline']


def bind_dist(features, dictionary):
    def dist(X1, X2):
        token1, token2 = dictionary.get_tokens([X1[0], X2[0]])
        features1 = features[token1]
        features2 = features[token2]
        dot = dict_dot(features1, features2)
        if dot == 0:
            return np.inf
        return 1 / float(dot)
    return dist


def bind_kernel(
    features=None, # Must be provided if syntax_feature_types is True
    syntax_feature_types=['baseline', 'dependency', 'hand_picked'],
    semantic_similarity='res',
    include_suffix=True,
    syntactic_multiplier=0.33,
    semantic_multiplier=0.33,
    suffix_multiplier=0.33
):
    '''
    Returns a kernel function that has a given dictionary and features
    lookup bound to its scope.
    '''

    # Validate that a sensible value for semantic similarity was provided
    semantic_similarity_is_valid = (
        semantic_similarity in LEGAL_SIMILARITIES 
        or semantic_similarity is None
    )
    if not semantic_similarity_is_valid:
        raise ValueError(
            'semantic_similarity must be one of the following: '
            + ', '.join(LEGAL_SIMILARITIES) 
            + '.  Got %s.' % repr(semantic_similarity)
        )

    # Validate that a sensible value for syntactic similarity was provided
    syntactic_similarity_is_valid = syntax_feature_types is None or all(
        feature_type in LEGAL_SYNTACTIC_SIMILARITIES 
        for feature_type in syntax_feature_types
    )
    if not syntactic_similarity_is_valid:
        raise ValueError(
            'syntax_feature_types must be a list with any of the following: '
            + ', '.join(LEGAL_SYNTACTIC_SIMILARITIES) 
            + '.  Got %s.' % repr(syntax_feature_types)
        )

    # Semantic similarity functions need an "information content" file 
    # to calculate similarity values.
    if semantic_similarity is not None:
        information_content = wordnet_ic.ic(INFORMATION_CONTENT_FILE)
        
    def kernel(A,B):
        '''
        Custom kernel function.  This counts how often the links incident on
        two different words within their respective dependency trees are the 
        same, up to the dependency relation and the POS of the neighbour.

        Note that A references a set of words' dependency trees, and B
        references another set.  So that this function end up making
        len(A) * len(B) of such comparisons, and return the result as a 
        len(A) by len(B) matrix.
        '''

        result = []
        for a in A:

            token_a = u.ensure_unicode(features.get_token(int(a[0])))

            # Get token_a's dependency tree features
            if syntax_feature_types is not None:
                syntax_features_a = features.get_features_idx(
                    int(a[0]), syntax_feature_types
                )

            # Get the token_a's synset if semantic similarity is being used
            if semantic_similarity is not None:
                semantic_features_a = nouns_only(wordnet.synsets(token_a))

            if include_suffix:
                suffix_a = features.get_suffix(token_a)

            result_row = []
            result.append(result_row)
            for b in B:

                kernel_score = 0
                token_b = u.ensure_unicode(features.get_token(int(b[0])))

                # Calculate the dependency tree kernel
                if syntax_feature_types is not None:
                    syntax_features_b = features.get_features_idx(
                        int(b[0]), syntax_feature_types
                    )
                    kernel_score += syntactic_multiplier * dict_dot(
                        syntax_features_a, syntax_features_b)

                # Calculate semantic similarity is being used
                if semantic_similarity is not None:
                    semantic_features_b = nouns_only(wordnet.synsets(token_b))
                    kernel_score += semantic_multiplier * max_similarity(
                        semantic_similarity, semantic_features_a, 
                        semantic_features_b, information_content
                    )

                # Determine if suffixes match
                if include_suffix:
                    suffix_b = features.get_suffix(token_b)
                    if suffix_a is not None and suffix_a == suffix_b:
                        kernel_score += suffix_multiplier

                result_row.append(kernel_score)

        return result

    return kernel


def nouns_only(synsets):
    '''
    Filters provided synsets keeping only nouns.
    '''
    return [s for s in synsets if s.pos() == 'n']


def max_similarity(
    similarity_type,
    synsets_a,
    synsets_b,
    information_content
):

    similarity_type += '_similarity'
    max_similarity = 0
    for synset_a in synsets_a:
        for synset_b in synsets_b:
            similarity = getattr(synset_a, similarity_type)(
                synset_b, information_content)
            if similarity > max_similarity:
                max_similarity = similarity

    return max_similarity


def dict_dot(a,b):
    result = 0
    for key in a:
        if key in b:
            result += a[key] * b[key]
    return result


class WordnetFeatures(object):

    def __init__(self, dictionary_path=None):

        if dictionary_path is None:
            self.dictionary = None
        else:
            self.dictionary = UnigramDictionary()
            self.dictionary.load(dictionary_path)

    def get_concept_weight(self, name):
        # Given a concept name, get it's weight.  This takes into account
        # the frequency of occurrence of all lemmas that can disambiguate
        # to that concept.  No attempt is made to figure out how often
        # a term in fact did disambiguate to the lemma
        pass


    def get_wordnet_features(self, lemma):
        synsets = [s for s in wordnet.synsets(lemma) if s.pos() == 'n']
        concepts = set()
        for s in synsets:
            concepts.update(self.get_wordnet_features_recurse(s))
        return concepts

    def get_wordnet_features_recurse(self, synset):
        concepts = set([synset.name()])
        parents = synset.hypernyms()
        parents = [p for p in parents if p.pos() == 'n']
        for p in parents:
            concepts.update(self.get_wordnet_features_recurse(p))
        return concepts

    def wordnet_kernel(self, lemma0, lemma1):
        concepts0 = self.get_wordnet_features(lemma0)
        concepts1 = self.get_wordnet_features(lemma1)
        concepts_in_common = concepts0 & concepts1

        kernel_score = 0
        for c in concepts_in_common:
            kernel_score += self.concept_weighting[c]

        return kernel_score
