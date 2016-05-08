from relation2vec_embedder import Relation2VecEmbedder
from minibatcher import Relation2VecMinibatcher
from SETTINGS import DATA_DIR, CORPUS_DIR
SAVEDIR = os.path.join(DATA_DIR, 'relation2vec')


# load the minibatch generator.  Prune very rare tokens.
print 'Loading and pruning dictionaries'
minibatcher = Relation2VecMinibatcher()
minibatcher.load(SAVEDIR)
minibatcher.prune(min_frequency=MIN_FREQUENCY)

# Make an embedder with the correct sizes
embedder = Relation2VecEmbedder(
	entity_vocab_size=len(minibatcher.entity_dictionary),
	context_vocab_size=len(minibatcher.context_dictionary),
)
