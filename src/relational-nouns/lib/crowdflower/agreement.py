"""
Calculate the agreement between annotators
"""

from collections import defaultdict
import krippendorff_alpha as k


def as_number_string(response):
	return {
		'usually relational': '2',
		'occasionally relational': '1',
		'almost never relational': '0',
		'+':'2',
		'0':'1',
		'-':'0'
	}[response]


def convert_interpeted_annotations(expert_annotations, participant_annotations):
	all_words = set()
	expert_converted = []
	participant_converted = []
	for word in expert_annotations:
		expert_converted.append(as_number_string(expert_annotations[word]))
		participant_converted.append(as_number_string(
			participant_annotations[word]))

	return [expert_converted, participant_converted]


def convert_data(results):
	"""
	Convert the data from the format given from analyze_results.get_results
	to the format expected by the Krippendorph alpha calculation package.
	This involves aggregating the responses on a per-worker basis.
	"""

	# Each worker is a key, and the values are a question:response dictionary
	# summarizing that worker's responses
	question_ids = set()
	workers_responses = defaultdict(dict)
	for result in results:
		question_id = result['data']['token']
		question_ids.add(question_id)
		judgments = result['results']['judgments']
		for judgment in judgments:
			worker = judgment['worker_id']

			# Get the response from one of two locations
			try:
				response = judgment['data']['response']
			except KeyError:
				response = judgment['data']['is_relational']

			# Convert it into a number
			response = as_number_string(response)

			workers_responses[worker][question_id] = response

	worker_responses_as_list = []
	ordered_questions = list(question_ids)
	for response_set in workers_responses.values():
		ordered_responses = [
			response_set[x] if x in response_set else '*'
			for x in ordered_questions
		]
		worker_responses_as_list.append(ordered_responses)

	# Now convert the dictionary of worker responses to a list, one row per
	# worker.  (Using a dict just helped to accumulate responses on a
	# per-worker basis, whereas the agreement calculator just expects workers
	# as "rows")

	return worker_responses_as_list


def calculate_krippendorf(data, metric='nominal'):

	# Calculate agreement
	if metric == 'nominal':
		print k.krippendorff_alpha(
			data,
			k.nominal_metric,
			missing_items='*'
		)
	elif metric == 'interval':
		print k.krippendorff_alpha(
			data,
			k.interval_metric,
			missing_items='*'
		)

if __name__ == '__main__': 
	print("Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha")

	data = ( 
		"*	*	*	*	*	3	4	1	2	1	1	3	3	*	3", # coder A
		"1	*	2	1	3	3	4	3	*	*	*	*	*	*	*", # coder B
		"*	*	2	1	3	4	4	*	2	1	1	3	3	*	4", # coder C
	)   

	missing = '*' # indicator for missing values
	array = [d.split() for d in data]  # convert to 2D list of string items


	print k.krippendorff_alpha(
		array,
		k.nominal_metric,
		missing_items='*'
	)
	
	#print("nominal metric: %.3f" % k.krippendorff_alpha(array, k.nominal_metric, missing_items=missing))
	#print("interval metric: %.3f" % k.krippendorff_alpha(array, k.interval_metric, missing_items=missing))
