from word2vec import UnigramDictionary, UNK, SILENT, ERROR
from collections import defaultdict

DELIMITER = ':::'

class PairDictionary(UnigramDictionary):

	# TODO: handle the case where init token_map and counter sampler aren't
	# None
	def __init__(
		self,
		on_unk=SILENT,
		token_map=None,
		counter_sampler=None,
		delimiter=DELIMITER
	):

		# Delegate to super's constructor
		super(PairDictionary, self).__init__(
			on_unk, token_map, counter_sampler)

		# Pairs are the countable item that we will be tracking.  However,
		# we also want to know about singles, and be able to look up all of
		# the pairs that a given single token is involved in
		self.singles_map = {}

		self.delimiter = delimiter


	def remove(self, token_pair):
		token1, token2, pair_token = self.glom(token_pair)
		idx = self.get_id(token_pair)
		self.token_map.remove(pair_token)
		self.counter_sampler.remove(idx)


	def compact(self):
		super(PairDictionary, self).compact()
		self.build_singles_map()


	def prune(self, min_frequency=5):
		super(PairDictionary, self).prune(min_frequency)
		self.build_singles_map()


	def build_singles_map(self):
		'''
		Completely rebuilds the singles map from the existing pair_tokens
		in the token_map.  
		'''
		self.singles_map = {}
		for pair_token in self.token_map.tokens:

			# Skip the special UNK token
			if pair_token == 'UNK':
				continue

			pair1, pair2 = pair_token.split(self.delimiter)
			try:
				self.singles_map[pair1].add(pair_token)
			except KeyError:
				self.singles_map[pair1] = {pair_token}

			try:
				self.singles_map[pair2].add(pair_token)
			except KeyError:
				self.singles_map[pair2] = {pair_token}


	def glom(self, token_pair):
		'''
		Ensures tokens are in lexicographic order.  Generates a merged
		token by concatenating the two tokens (in lexo order) separated
		by the delimiter.  Returns the individual tokens in lexo order
		followed by their concatenation.
		'''
		# Unpack the token pair
		token1, token2 = token_pair

		# Put tokens in lexicographic order
		if token1 > token2:
			token1, token2 = token2, token1

		pair_token = '%s%s%s' % (token1, self.delimiter, token2)

		return token1, token2, pair_token


	def add(self, token_pair):

		# Get lexicographically ordered tokens, and their merged version.
		token1, token2, pair_token = self.glom(token_pair)

		# The pair token is formed by concatenating the two tokens
		# We insert this combined token in the usual way, so that pairs
		# get counted.
		super(PairDictionary, self).add(pair_token)

		# We also make a record in the singles_map so that we can keep
		# track of what pairs a given single token is involved in
		try:
			self.singles_map[token1].add(pair_token)
		except KeyError:
			self.singles_map[token1] = {pair_token}
		try:
			self.singles_map[token2].add(pair_token)
		except KeyError:
			self.singles_map[token2] = {pair_token}


	def get_id(self, token_pair):
		token1, token2, pair_token = self.glom(token_pair)
		return super(PairDictionary, self).get_id(pair_token)


	def get_ids(self, token_pair_iterator):
		return [self.get_id(pair) for pair in token_pair_iterator]
		

	def load(self, loaddir):
		super(PairDictionary, self).load(loaddir)
		self.build_singles_map()



