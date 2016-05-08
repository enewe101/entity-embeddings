In this project we experiment with learning embeddings for entities
using a variation of the skip-gram idea.  Given two entities, we learn
the embedding for each that enables prediction of the nearby words.

Put this in front of executables to make sure theano uses the GPU:
THEANO_FLAGS='floatX=float32,device=gpu'
