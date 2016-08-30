from word2vec import Word2VecEmbedder as E
import numpy as np
from theano import tensor as T
from test import sigma

embedder = E(T.imatrix(), batch_size=200, query_vocabulary_size=5, context_vocabulary_size=5)
embedder.load('/gs/project/eeg-641-aa/enewel3/entity-embeddings/data/entity-pair-pretrain-freeze-between-test')
W, C = embedder.get_param_values()
print np.round(sigma(np.dot(W, C.T)), 2)
