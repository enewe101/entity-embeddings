import os 
from multiprocessing import Process, Pipe, Queue
from Queue import Empty
import sys


def parse(filename):
	tokenized_sentences = []
	for line in open(filename):
		tokenized_sentences.append(line.strip().split())
	return tokenized_sentences




