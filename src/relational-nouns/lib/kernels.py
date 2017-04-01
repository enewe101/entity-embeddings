from nltk.corpus.reader.wordnet import WordNetError
import multiprocessing
import iterable_queue as iq
import itertools
import numpy as np
import sys
sys.path.append('..')
import t4k
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


class PrecalculatedKernel(object):

    def __init__(
        self, 
        features=None, # Must be provided if syntax_feature_types is True
        syntax_feature_types=['baseline', 'dependency', 'hand_picked'],
        semantic_similarity='res',
        include_suffix=True,
        syntactic_multiplier=0.33,
        semantic_multiplier=0.33,
        suffix_multiplier=0.33,
    ):
        self.features = features
        self.syntax_feature_types = syntax_feature_types
        self.semantic_similarity = semantic_similarity
        self.include_suffix = include_suffix
        self.syntactic_multiplier = syntactic_multiplier
        self.semantic_multiplier = semantic_multiplier
        self.suffix_multiplier = suffix_multiplier

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
            self.information_content = wordnet_ic.ic(INFORMATION_CONTENT_FILE)
            
        self.cache = {}


    def precompute(self, examples):
        """
        Precompute the kernel evaluation of all pairs in examples.
        """
        # Add all the example pairs to the work queue
        for ex1, ex2 in itertools.combinations(examples, 2):
            dot = self.eval_pair(ex1, ex2)


    def precompute_parallel(self, examples, num_processes=12):
        """
        Use multiprocessing to precompute the kernel evaluation of all pairs in 
        examples.
        """
        work_queue = iq.IterableQueue()
        result_queue = iq.IterableQueue()

        # Add all the example pairs to the work queue
        work_producer = work_queue.get_producer()
        for ex1, ex2 in itertools.combinations(examples, 2):
            work_producer.put((ex1, ex2))
        work_producer.close()

        # Start a bunch of workers
        for proc in range(num_processes):
            p = multiprocessing.Process(
                target=self.precompute_worker,
                args=(work_queue.get_consumer(), result_queue.get_producer())
            )
            p.start()

        # Get a result consumer, which is the last endpoint.  No more endpoints 
        # will be made from either queue, so close them
        result_consumer = result_queue.get_consumer()
        result_queue.close()
        work_queue.close()

        # Get all the results and cache them
        for ex1, ex2, dot in result_consumer:
            self.cache[frozenset((ex1, ex2))] = dot


    def precompute_worker(self, work_consumer, result_producer):
        for ex1, ex2 in work_consumer:
            dot = self.eval_pair(ex1, ex2)
            result_producer.put((ex1, ex2, dot))
        result_producer.close()


    def eval_pair(self, a, b):
        '''
        Custom kernel function.  This counts how often the links incident on
        two different words within their respective dependency trees are the 
        same, up to the dependency relation and the POS of the neighbour.

        Note that A references a set of words' dependency trees, and B
        references another set.  So that this function end up making
        len(A) * len(B) of such comparisons, and return the result as a 
        len(A) by len(B) matrix.
        '''

        # Check the cache
        if frozenset((a,b)) in self.cache:
            #t4k.out('+')
            return self.cache[frozenset((a,b))]
        #t4k.out('.')

        # Get a's dependency tree features
        if self.syntax_feature_types is not None:
            syntax_features_a = self.features.get_features(
                a, self.syntax_feature_types
            )

        # Get the a's synset if semantic similarity is being used
        if self.semantic_similarity is not None:
            try:
                semantic_features_a = nouns_only(wordnet.synsets(a))
            except WordNetError:
                print a
                return 0

        if self.include_suffix:
            suffix_a = self.features.get_suffix(a)

        kernel_score = 0

        # Calculate the dependency tree kernel
        if self.syntax_feature_types is not None:
            syntax_features_b = self.features.get_features(
                b, self.syntax_feature_types
            )
            kernel_score += self.syntactic_multiplier * dict_dot(
                syntax_features_a, syntax_features_b)

        # Calculate semantic similarity is being used
        if self.semantic_similarity is not None:
            try:
                semantic_features_b = nouns_only(wordnet.synsets(b))
            except WordNetError:
                print b
                return 0

            try:
                kernel_score += self.semantic_multiplier * max_similarity(
                    self.semantic_similarity, semantic_features_a, 
                    semantic_features_b, self.information_content
                )
            except WordNetError:
                print a, b

        # Determine if suffixes match
        if self.include_suffix:
            suffix_b = self.features.get_suffix(b)
            if suffix_a is not None and suffix_a == suffix_b:
                kernel_score += self.suffix_multiplier

        self.cache[frozenset((a,b))] = kernel_score

        return kernel_score


    def eval(self, A, B):

        """
        Custom kernel function.  This counts how often the links incident on
        two different words within their respective dependency trees are the 
        same, up to the dependency relation and the POS of the neighbour.

        Note that A references a set of words' dependency trees, and B
        references another set.  So that this function end up making
        len(A) * len(B) of such comparisons, and return the result as a 
        len(A) by len(B) matrix.
        """

        result = []
        for a in A:
            result_row = []
            result.append(result_row)
            for b in B:
                token_a = u.ensure_unicode(self.features.get_token(int(a[0])))
                token_b = u.ensure_unicode(self.features.get_token(int(b[0])))
                result_row.append(self.eval_pair(token_a,token_b))

        return result



def bind_cached_kernel(
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
        
    cache = {}
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

                # Check the cache
                if frozenset((a,b)) in cache:
                    result_row.append(cache[frozenset((a,b))])
                    continue

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

                cache[frozenset((a,b))] = kernel_score
                result_row.append(kernel_score)

        return result

    return kernel



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
            self.dictionary = t4k.UnigramDictionary()
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
