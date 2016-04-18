#!/usr/bin/env python
import time
import sys
import re
from word2vec import Word2Vec as W, MinibatchGenerator as MG

def parse(filename):

	tokenized_sentences = []
	for line in open(filename):
		tokenized_sentences.append(line.split('\t')[1].split())

	return tokenized_sentences


def prepare_mg():
	directories=[
		'/home/rldata/gigaword-corenlp/cooccurrence',
		'/home/rldata/gigaword-corenlp/cooccurrence/0'
	]
	skip = [
		re.compile('README.txt')
	]
	mg = MG(
		directories=directories,
		skip=skip,
		batch_size=10000,
		t=1e-4,
		parse=parse
	)
	mg.prepare(savedir='/home/2012/enewel3/entity-embeddings/data/word2vec')

if __name__ == '__main__':
	if sys.argv[1] == 'prepare':

		start = time.time()
		prepare_mg()
		elapsed = time.time() - start
		print 'Elapsed:', elapsed
