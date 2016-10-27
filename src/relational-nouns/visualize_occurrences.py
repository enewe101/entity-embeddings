import sys
sys.path.append('..')
from SETTINGS import DATA_DIR, GIGAWORD_DIR
from subprocess import check_output
from corenlp_xml_reader import AnnotatedText 
import os
import utils

GREP_RESULT_PATH = os.path.join(DATA_DIR, 'grep-result.html')

def grep_giga_for_relational_nouns():
	positives, negatives = utils.get_training_sets()
	grep_giga(positives, limit=None)


def gigaword(limit=100):
	giga_path = os.path.join(GIGAWORD_DIR, 'data', '9fd', 'CoreNLP')
	fnames = check_output(['ls', giga_path]).split()
	fnames = [os.path.join(giga_path, f) for f in fnames][:limit]

	return [(fname, AnnotatedText(open(fname).read())) for fname in fnames]

def get_article_id(path):
	fname = path.split('/')[-1]
	article_id = fname.split('.')[0]
	return article_id


def grep_giga(target_lemmas, limit=100, fname=GREP_RESULT_PATH):
	'''
	Search through `limit` number of gigaword articles, finding sentences
	that match the lemmas listed in `target_lemmas` (a set of strings), 
	and create an html page that displays the matched senteces with
	matched text highlighted
	'''

	if fname is not None:
		out_file = open(fname, 'w')

	markup = ''
	for fname, article in gigaword(limit=limit):

		# Get the markup for matched sentences
		match_markups = grep_article(target_lemmas, article)

		# Wrap it in additional markup, and accumulate the markup
		for sentence_id, match_markup in match_markups:
			markup += '<div class="sentence">'
			markup += '<div class="sentence_id">'
			markup += '%s : %d' % (get_article_id(fname), sentence_id)
			markup += '</div>'
			markup += match_markup
			markup += '</div>'

	# Wrap the markup in an html page with styling
	markup = '<html>%s<body>%s</body></html>' % (get_html_head(), markup)

	# Write markup to file (if given)
	if fname is not None:
		out_file.write(markup)

	# Return the markup
	return markup


def grep_article(target_lemmas, annotated_text):
	'''
	Find sentences in `annotated_text` that have lemmas that match
	elements in `target_lemmas`, then output html markup for such sentences
	so as to highlight occurrences of elements in `target_lemmas` and 
	indicate parts of speech.
	- `target_lemmas` should be a set of strings representing lemmas
	- `annotated_text` a corenlp_xml_reader.AnnotatedText instance
	'''

	for sentence_id, sentence in enumerate(annotated_text.sentences):

		# If this sentence matches any of the lemmas, add markup for it
		marked_up_sentences = []
		lemmas = set([t['lemma'] for t in sentence['tokens']])
		if lemmas & target_lemmas:

			markup = ''
			for token in sentence['tokens']:
				token_markup = '<span class="token">'

				# Add the word, highlight if its lemma was a match
				if token['lemma'] in target_lemmas:
					token_markup += (
						'<span class="match">%s</span>' % token['word'])
				else:
					token_markup += token['word']

				# Add the pos tag then close the token tag
				token_markup += '<span class="pos">'
				token_markup += '<span class="pos-inner">'
				token_markup += token['pos'] + '</span></span></span> '

				# Accumulate the markup for each token
				markup += token_markup

			marked_up_sentences.append((sentence_id, markup))

	return marked_up_sentences


def get_html_head():
	return ' '.join([
		'<head><style>',
		'.pos {position: absolute; font-size: 0.6em;',
			'top: 6px; left: 50%; font-weight: normal;',
			'font-style: normal}',
		'.pos-inner {position: relative; left:-50%}',
		'body {line-height: 40px;}',
		'p {margin-bottom: 30px; margin-top: 0}',
		'.match {color: blue; font-weight: bold;}',
		'.token {position: relative}',
		'.attribution-id {display: block; font-size:0.6em;',
			'margin-bottom:-20px;}',
		'</style></head>'
	])


if __name__ == '__main__':
	grep_giga_for_relational_nouns()
