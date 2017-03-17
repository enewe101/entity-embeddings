from xml.dom import minidom
import random

question_groupings = {
	'basic_examples': [
		'mother', 'aunt', 'husband', 'nephew', 'relative', 'friend', 'mentor', 
		'coworker', 'entourage', 'confidante', 'sibling', #'ally', 

		'bread', 'fever', 'book', 'box', 'flower', 'bus', 'mannequin', 
		'direction', 'friendship', 'conflict', 'connection', #'alliance', 
		'agreement',
	],

	'functional_relationships': [
		'passenger', 'author', 'subsidiary', 'replacement', 'successor', 'hole',
		'solution', 'province', 'heir', 'legacy', 
		
		'distribution', 'organization', 'delivery', 'girder', 'roof', 
		'building', 'adherence',
	],

	'relative_parts': [
		'corner', 'middle', 'part', 'edge', 'outside', 'stern',
		'front', 
		
		'wheel', 'strap', 'door', 'shelf', 'handle', 'cord', 'clip', 'stirrup',
		'wire', 'rail', 'crust'
	],

	'roles': [
		'director', 'CEO', 'president', 'ambassador', 'supervisor', 'pitcher',
		'guitarist', 'shareholder', 'lawyer', 

		'babysitter', 'planner', 'astronaut', 'purchaser', 'miner', 'gardener',
	],

	'self_test': [
		'foreigner', 'stranger', 'manufacturer', 'producer', 'discovery',
		'creation', 'loser', 'pair', 'supplier', 'energy', 'rapport', 'captain',
		'top', 'base', 

		'dispute', 'contract', 'reporter', 'gender', 'reconnaissance', 'tip',
		'war', 'operator', 'copper', 'nail', 'realization', 'writer', 'noon',
		'head', 'neck', 'delegate', 'investor', 'athlete', 'mathematician',
		'scientist', 'predecessor', 'mayor', 'blacksmith', 'partner',
		'assistant'
	]
}


question_specs = {
	'mother': {
		'query': 'mother',
		'correct': 'relational',
		'response': (
			'&ldquo;Mother&rdquo; expresses the mother-child '
			'relationship and refers to one of the relata (the mother), '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'aunt': {
		'query': 'aunt',
		'correct': 'relational',
		'response': (
			'&ldquo;Aunt&rdquo; expresses the aunt-nephew or aunt-niece '
			'relationship and refers to one of the relata (the aunt), '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'husband': {
		'query': 'husband',
		'correct': 'relational',
		'response': (
			'&ldquo;Husband&rdquo; expresses a spousal '
			'relationship, and refers to one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'nephew': {
		'query': 'nephew',
		'correct': 'relational',
		'response': (
			'&ldquo;Nephew&rdquo; expresses a aunt-nephew or uncle-nephew '
			'relationship, and refers to one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'relative': {
		'query': 'relative',
		'correct': 'relational',
		'response': (
			'&ldquo;Relative&rdquo; expresses a generic blood-relative '
			'relationship, and refers to one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'friend': {
		'query': 'friend',
		'correct': 'relational',
		'response': (
			'&ldquo;Friend&rdquo; expresses the friendship relationship '
			'and refers to one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'mentor': {
		'query': 'mentor',
		'correct': 'relational',
		'response': (
			'&ldquo;Mentor&rdquo; expresses the mentor-mentee relationship '
			'and refers to one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'ally': {
		'query': 'ally',
		'correct': 'relational',
		'response': (
			'&ldquo;Ally&rdquo; expresses an alliance relationship, and '
			'refers to one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'coworker': {
		'query': 'coworker',
		'correct': 'relational',
		'response': (
			'&ldquo;Coworker&rdquo; expresses the relationship between '
			'peers working in the same place, and refers to one of the '
			'relata, therefore it is '
			'relational'
			'.'
		)
	},
	'entourage': {
		'query': 'entourage',
		'correct': 'relational',
		'response': (
			'&ldquo;Entourage&rdquo; expresses a relationship between '
			'a person (usually a celebrity) and their friends and '
			'assistants who travel with them.  It refers to one of the '
			'relata (the friends and assistants as a group), '
			'so it is '
			'relational'
			'.'
		)
	},
	'confidante': {
		'query': 'confidante',
		'correct': 'relational',
		'response': (
			'&ldquo;Confidante&rdquo; expresses a relationship between '
			'one person and another with whom they share personal or '
			'sensitive information.  Also, it refers to one of the relata '
			'(the recipient of the information), therefore it is '
			'relational'
			'.'
		)
	},
	'sibling': {
		'query': 'sibling',
		'correct': 'relational',
		'response': (
			'&ldquo;Sibling&rdquo; expresses the siblinghood relationship '
			'between brothers and / or sisters, and when used it refers to '
			'one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'bread': {
		'query': 'bread',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Bread&rdquo; does not express a relationship, "
			'so it is '
			'non-relational'
			'.'
		)
	},
	'fever': {
		'query': 'fever',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Fever&rdquo; does not express a relationship, "
			'so it is '
			'non-relational'
			'.'
		)
	},
	'book': {
		'query': 'book',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Book&rdquo; does not express a relationship, '
			'so it is '
			'non-relational'
			'.'
		)
	},
	'box': {
		'query': 'box',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Box&rdquo; does not express a relationship, '
			'so it is '
			'non-relational'
			'.'
		)
	},
	'flower': {
		'query': 'flower',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Flower&rdquo; does not express a relationship, '
			'so it is '
			'non-relational'
			'.'
		)
	},
	'bus': {
		'query': 'bus',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Bus&rdquo; does not express a relationship, '
			'so it is '
			'non-relational'
			'.'
		)
	},
	'mannequin': {
		'query': 'mannequin',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Mannequin&rdquo; does not express a relationship, '
			'so it is '
			'non-relational'
			'.'
		)
	},
	'direction': {
		'query': 'direction',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Direction&rdquo; does not express a relationship, '
			'so it is '
			'non-relational'
			'.'
		)
	},
	'friendship': {
		'query': 'friendship',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Friendship&rdquo; does express a relationship, '
			'but it does not refer to one of the relata.  It refers to '
			"the relationship itself, so it is "
			'non-relational'
			'.'
		)
	},
	'conflict': {
		'query': 'conflict',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Conflict&rdquo; does express a relationship, but it '
			"doesn't refer to one of the relata of that relationship. "
			"It refers to the relationship itself, so it is "
			'non-relational'
			'.'
		)
	},
	'alliance': {
		'query': 'alliance',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Alliance&rdquo; does express a relationship, but it '
			"doesn't refer to one of the relata of that relationship. "
			"It refers to the relationship itself, so it is "
			'non-relational'
			'.'
		)
	},
	'connection': {
		'query': 'connection',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Connection&rdquo; does express a relationship, but "
			"it doesn't refer to one of the relata of the relationship. "
			"It refers to the relationship itself, so it is "
			'non-relational'
			'.'
		)
	},
	'agreement': {
		'query': 'agreement',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Agreement&rdquo; does express a relationship, but "
			"it doesn't refer to one of the relata of that relationship. "
			"It refers to the relationship itself, so it is "
			'non-relational'
			'.'
		)
	},
	'passenger': {
		'query': 'passenger',
		'correct': 'relational',
		'response': (
			'&ldquo;Passenger&rdquo; is inherently defined in terms of '
			'a relationship between a person and a mode of transport. '
			'Since it refers to one of the relata of that '
			"relationship, it's a "
			'relational'
			' noun.'
		)
	},
	'author': {
		'query': 'author',
		'correct': 'relational',
		'response': (
			'&ldquo;Author&rdquo; is typically used to establish a '
			'relationship between a written work and its creator.  Since '
			"it refers to one of the relata (the creator) it's a "
			'relational '
			' noun.'
		)
	},
	'subsidiary': {
		'query': 'subsidiary',
		'correct': 'relational',
		'response': (
			'&ldquo;Subsidiary&rdquo; expresses the relationship between '
			'a parent company and another company owned by the parent. '
			'It refers to one of the relata (the owned company), so it is '
			'a '
			'relational'
			' noun.'
		)
	},
	'replacement': {
		'query': 'replacement',
		'correct': 'relational',
		'response': (
			'&ldquo;Replacement&rdquo; expresses a relationship between '
			'an original object and a secondary object that '
			'substitutes the original.  It refers to one of the relata '
			"(the substitute) so it's a "
			'relational'
			' noun.'
		)
	},
	'successor': {
		'query': 'successor',
		'correct': 'relational',
		'response': (
			'&ldquo;Successor&rdquo; expresses a relationship '
			'between one object and second that follows '
			'in some kind of lineage.  It refers to one of '
			'the relata (the secondary object), so it is a '
			'relational'
			' noun.'
		)
	},
	'hole': {
		'query': 'hole',
		'correct': 'relational',
		'response': (
			'&ldquo;Hole&rdquo; expresses the relationship between some '
			'self-connected entity and an enclosed gap in that entity. '
			'It refers to one of the relata (the gap) so it is a '
			'relational'
			' noun.'
		)
	},
	'solution': {
		'query': 'solution',
		'correct': 'relational',
		'response': (
			'&ldquo;Solution&rdquo; expresses the relationship between '
			'some problem and the agent or technique that solves it. '
			"It refers to one of the relata, so it's a "
			'relational'
			' noun.'
		)
	},
	'province': {
		'query': 'province',
		'correct': 'relational',
		'response': (
			'&ldquo;Province&rdquo; expresses a relationship between a '
			'one geo-political entity and another that subsumes it. '
			'It refers to one of the '
			"relata (the subsumed geo-political entity) so it's "
			'a '
			'relational'
			' noun.'
		)
	},
	'heir': {
		'query': 'heir',
		'correct': 'relational',
		'response': (
			'&ldquo;Heir&rdquo; expresses the relationship between a '
			'person and their inheritance.  It refers to one of those '
			'relata (the person), so it is a '
			'relational'
			' noun.'
		)
	},
	'legacy': {
		'query': 'legacy',
		'correct': 'relational',
		'response': (
			'&ldquo;Legacy&rdquo; expresses a relationship between a '
			'person and the set of notable things that they will be '
			'remembered for.  It refers to one of the relata (the person) '
			"so it's a "
			'relational'
			' noun.'
		)
	},
	'distribution': {
		'query': 'distribution',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Distribution&rdquo; does not express a relationship. '
			'It expresses either the act of distributing something, or '
			'the manner in which it has been distributed.  Since it '
			"doesn't express a relationship, it isn't a "
			'relational'
			' noun.'
		)
	},
	'organization': {
		'query': 'organization',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Organization&rdquo; doesn't express a relationship. "
			'It expresses the act of organizing, the manner in which '
			'something is organized, or it expresses an administrative '
			"body like a company or institution.  Since it doesn't "
			"express a relationship, it is "
			'non-relational'
			'.'
		)
	},
	'delivery': {
		'query': 'delivery',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Delivery&rdquo; doesn't express a "
			'relationship, but rather expresses either the act of '
			'delivering, '
			'as in "She went home after she finished her delivery", or '
			'a performance (usually a show or presentation), '
			'as in "Her delivery was smooth and well-practiced". '
			"Since delivery does not express a relationship, it is "
			'non-relational'
			'.'
		)
	},
	'girder': {
		'query': 'girder',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Girder&rdquo; doesn't express a relationship so "
			"it is "
			'non-relational'
			'.'
		)
	},
	'roof': {
		'query': 'roof',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Roof&rdquo; doesn't express a relationship so "
			"it is "
			'non-relational'
			'.'
		)
	},
	'building': {
		'query': 'building',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Building&rdquo; either refers to a permanent "
			'construction made to hold people and physical objects, or '
			'the act of constructing something.  Neither meaning '
			'expresses a relationship, so &ldquo;building&rdquo; is '
			'non-relational'
			'.'
		)
	},
	'adherence': {
		'query': 'adherence',
		'correct': 'non-relational',
		'response': (
			'While &ldquo;adherence&rdquo; can, in a a sense, be used to '
			'express a '
			'relationship (as in "the adherence of the stickers to the '
			'page") it '
			"doesn't refer to one of the relata, so it is "
			'non-relational'
			'.'
		)
	},
	'corner': {
		'query': 'corner',
		'correct': 'relational',
		'response': (
			'&ldquo;Corner&rdquo; identifies a part of a physical object '
			'where edges or faces meet to form a sharp protrusion '
			'It establishes a physical / geometric relationship between '
			'the part to which it refers (a relatum) and the whole object '
			'so it is a '
			'relational'
			' noun.'
		)
	},
	'middle': {
		'query': 'middle',
		'correct': 'relational',
		'response': (
			'&ldquo;Middle&rdquo; identifies a part of an object extended '
			"in space or time based on it's spatial or temporal "
			'relationship to the whole.  Since it establishes a '
			"relationship while referring to one of the relata, it's a "
			'relational'
			' noun.'
		)
	},
	'part': {
		'query': 'part',
		'correct': 'relational',
		'response': (
			'&ldquo;Part&rdquo; expresses the whole-part relationship, and '
			'refers to one of the relata in that relationship (the part), '
			"so it's a "
			'relational'
			' noun.'
		)
	},
	'base': {
		'query': 'base',
		'correct': 'occasionally relational',
		'response': (
			'&ldquo;Base&rdquo; can be a relative part, signifying the '
			'bottom part of an object, '
			'but it can also refer to an outpost where operatives '
			'are located.  We consider '
			'&ldquo;base&rdquo; to be '
			'occasionally relational.'
		)
	},
	'top': {
		'query': 'top',
		'correct': 'relational',
		'response': (
			'&ldquo;Top&rdquo; can express a physical relationship between '
			'a whole object and its uppermost part.  It can also refer to '
			'a toy that played with by spinning it.  We think that the '
			'latter meaning, which satisfies the relational noun criteria '
			'is the most common meaning, so we consider &ldquo;top&rdquo; '
			'to be a '
			'relational'
			' noun.'
		)
	},
	'edge': {
		'query': 'edge',
		'correct': 'relational',
		'response': (
			'&ldquo;Edge&rdquo; identifies part of an object in terms of '
			'its physical relationship to the whole, so it is a '
			'relational'
			' noun.'
		)
	},
	'outside': {
		'query': 'outside',
		'correct': 'relational',
		'response': (
			'&ldquo;Outside&rdquo; identifies a region based on its '
			'relationship to some object, so it is a '
			'relational'
			' noun.'
		)
	},
	'stern': {
		'query': 'stern',
		'correct': 'relational',
		'response': (
			'&ldquo;Stern&rdquo; identifies a part of a ship based on '
			"its spatial relationship to the rest of the ship, so it's a "
			'relational'
			' noun.'
		)
	},
	'front': {
		'query': 'front',
		'correct': 'relational',
		'response': (
			'&ldquo;Front&rdquo; identifies part of an object based on '
			'its physical relationship to the whole, so it is a '
			'relational'
			' noun.'
		)
	},
	'wheel': {
		'query': 'wheel',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;wheel&rdquo; is usually part of another '
			"object, the meaning of wheel doesn't specifically denote "
			'such a relationship, so &ldquo;wheel&rdquo; is '
			'non-relational'
			'.'
		)
	},
	'strap': {
		'query': 'strap',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;strap&rdquo; is usually part of another '
			"object, the meaning of strap doesn't specifically denote "
			'such a relationship, so &ldquo;strap&rdquo; is '
			'non-relational'
			'.'
		)
	},
	'door': {
		'query': 'door',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;door&rdquo; is usually part of another '
			"object, the meaning of &ldquo;door&rdquo; doesn't "
			'specifically denote '
			'such a relationship, so it is '
			'non-relational'
			'.'
		)
	},
	'shelf': {
		'query': 'shelf',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;shelf&rdquo; can a be part of another '
			"object, the meaning of &ldquo;shelf&rdquo; doesn't "
			'specifically denote '
			'such a relationship, so it is '
			'non-relational'
			'.'
		)
	},
	'handle': {
		'query': 'handle',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;handle&rdquo; is typically part of another '
			"object, the meaning of &ldquo;handle&rdquo; doesn't "
			'specifically denote '
			'such a relationship, so it is '
			'non-relational'
			'.'
		)
	},
	'cord': {
		'query': 'cord',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;cord&rdquo; is typically part of another '
			"object, the meaning of &ldquo;cord&rdquo; doesn't "
			'specifically denote '
			'such a relationship, so it is '
			'non-relational'
			'.'
		)
	},
	'clip': {
		'query': 'clip',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;clip&rdquo; is typically part of another '
			"object, the meaning of &ldquo;clip&rdquo; doesn't "
			'specifically denote '
			'such a relationship, so it is '
			'non-relational'
			'.'
		)
	},
	'stirrup': {
		'query': 'stirrup',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Stirrup&rdquo; does not denote a relationship, so '
			"it is "
			'non-relational'
			'.'
		)
	},
	'wire': {
		'query': 'wire',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;wire&rdquo; is typically part of another '
			"object, the meaning of &ldquo;wire&rdquo; doesn't "
			'specifically denote '
			'such a relationship, so it is '
			'non-relational'
			'.'
		)
	},
	'rail': {
		'query': 'rail',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;rail&rdquo; can part of another '
			"object, the meaning of &ldquo;rail&rdquo; doesn't "
			'specifically denote '
			'such a relationship, so it is '
			'non-relational'
			'.'
		)
	},
	'crust': {
		'query': 'crust',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Crust&rdquo; identifies part of an object (usually '
			'a baked good that is firmer, crunchier, and/or more cooked '
			'One could make the argument that the physical relationship '
			'of a crust to the rest of the object is the salient aspect '
			"of the word's meaning, and that it is therefore a "
			'relational noun.  We consider it simply to be defined in '
			'terms '
			'of its properties, and that the fact that it is typically '
			'located at the outer surface or edge of baked food to be '
			'a typical property of crust but not the essential meaning, '
			'so we consider &ldquo;crust&rdquo; to be '
			'non-relational'
			'. Nevertheless, a good argument could be made to consider '
			'it relational.'
		)
	},
	'director': {
		'query': 'director',
		'correct': 'relational',
		'response': (
			'&ldquo;Director&rdquo; identifies a role, and is usually '
			'used to indicate a leadership relationship to an '
			'organizational unit, as in "Director of Finance". '
			'It can also be used in the sense of "the director of the film", '
			'which is also a relational usage. '
			'Therefore, we judge '
			'&ldquo;director&rdquo; to be a '
			'relational'
			' noun.'
		)
	},
	'CEO': {
		'query': 'CEO',
		'correct': 'relational',
		'response': (
			'&ldquo;CEO&rdquo; identifies a role, and is usually '
			'used to indicate a leadership relationship to a '
			'company, as in "The CEO of Walmart".  '
			'But, CEO can also be used in a generic non-relational sense, '
			'as in &ldquo;CEO salaries have skyrocketed.&rdquo; '
			'We think '
			'the relational usage is more common, and judge '
			'&ldquo;CEO&rdquo; to be a '
			'relational'
			' noun.'
		)
	},
	'president': {
		'query': 'president',
		'correct': 'relational',
		'response': (
			'&ldquo;President&rdquo; identifies a role, and is usually '
			'used to indicate a leadership relationship to country '
			'as in "President of Italy".  We think '
			'the relational usage is most common, and judge '
			'&ldquo;president&rdquo; to be a '
			'relational'
			' noun.'
		)
	},
	'ambassador': {
		'query': 'ambassador',
		'correct': 'relational',
		'response': (
			'&ldquo;Ambassador&rdquo; identifies a role, and is usually '
			'used to indicate a relationship between that person and '
			'either their country of nationality, and / or the foreign '
			'country in which they are posted, '
			'as in "US Ambassador to France".  We think '
			'the relational usage is most common, and judge '
			'&ldquo;ambassador&rdquo; to be a '
			'relational'
			' noun.'
		)
	},
	'supervisor': {
		'query': 'supervisor',
		'correct': 'relational',
		'response': (
			'&ldquo;Supervisor&rdquo; identifies a role, and is usually '
			'used to indicate a leadership and oversight relationship to '
			'another person or group of people, '
			'as in "John\'s supervisor never lets him take breaks." '
			'We think the relational usage is most common, and judge '
			'&ldquo;supervisor&rdquo; to be a '
			'relational'
			' noun.'
		)
	},
	'pitcher': {
		'query': 'pitcher',
		'correct': 'relational',
		'response': (
			'&ldquo;Pitcher&rdquo; identifies a role on a baseball team, '
			'as in "Pitcher for the Mets".  We think '
			'the relational usage is most common, and judge '
			'&ldquo;pitcher&rdquo; to be a '
			'relational'
			' noun.'
		)
	},
	'guitarist': {
		'query': 'guitarist',
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Guitarist&rdquo; can identify a role in a band, '
			'as in "Jimi Hendrix started as a guitarist for the Isley '
			'Brothers".  However, its non-relational usage, as in '
			'"Jimi Hendrix is a legendary guitarist" is also probably '
			'very common, and we judge &ldquo;guitarist&rdquo; to be '
			'partly relational'
			'.'
		)
	},
	'shareholder': {
		'query': 'shareholder',
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Shareholder&rdquo; expresses the relationship '
			'between the owner of shares and the company or shares that '
			'are owned. But the word is also frequently used in a '
			'non-relational sense, meaning generically one who owns '
			'shares, without reference to the particular shares or '
			"company.  We don't think the relational usage is necessarily "
			'the most common, so we judge &ldquo;shareholder&rdquo; to be '
			'partly relational'
			'.'
		)
	},
	'lawyer': {
		'query': 'lawyer',
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Lawyer&rdquo; can refer to a particular vocation. '
			'But it is also often used to indicate the relationship '
			'between the lawyer and the person (s)he is representing, '
			'as in "Julian Assange\'s lawyer was not available for '
			'comment." '
			'We don\'t think the relational usage is sufficiently common '
			'to consider &ldquo;lawyer&rdquo; a relational noun, so we '
			'judge it to be '
			'partly relational'
			'.'
		)
	},
	'babysitter': {
		'query': 'babysitter',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Babysitter&rdquo; identifies a person that watches '
			'over children.  While it could be used relationally, '
			'establishing a relationship to the children, we think that '
			'its core of the meaning is about the performance of duty '
			'rather than the relationship it implies with the children. '
			'Therefore we judge &ldquo;babysitter&rdquo; to be '
			'non-relational'
			', although an argument could be made otherwise.'
		)
	},
	'planner': {
		'query': 'planner',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Planner&rdquo; identifies a person who determines '
			'a sequence of actions or allocation of resources to achieve '
			'some goal.  The meaning implies the undertaking of an '
			"activity, but doesn't centrally express a relationship. "
			'Therefore we consider &ldquo;planner&rdquo; to be a '
			'non-relational'
			' noun.'
		)
	},
	'astronaut': {
		'query': 'astronaut',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Astronaut&rdquo; identifies a particular vocation, '
			'but does not centrally express a relationship.  Therefore '
			'we consider &ldquo;astronaut&rdquo; to be a '
			'non-relational'
			' noun.'
		)
	},
	'purchaser': {
		'query': 'purchaser',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Purchaser&rdquo; identifies a particular vocation, '
			'or generally a person responsible for the act of purchasing '
			'but does not centrally express a relationship.  Therefore '
			'we consider &ldquo;purchaser&rdquo; to be a '
			'non-relational'
			' noun.'
		)
	},
	'miner': {
		'query': 'miner',
		'correct': 'non-relational',
		'response': (
			'&ldquo;miner&rdquo; identifies a particular vocation, '
			'but does not centrally express a relationship.  Therefore '
			'we consider &ldquo;miner&rdquo; to be a '
			'non-relational'
			' noun.'
		)
	},
	'gardener': {
		'query': 'gardener',
		'correct': 'non-relational',
		'response': (
			'&ldquo;gardener&rdquo; identifies a particular vocation, '
			'but does not centrally express a relationship.  Therefore '
			'we consider &ldquo;gardener&rdquo; to be a '
			'non-relational'
			' noun.'
		)
	},
	'foreigner': {
		'query': 'foreigner',
		'correct': 'relational',
		'response': (
			'&ldquo;Foreigner&rdquo; expresses a relationship between '
			'a person and a country, wherein the person does not normally '
			'live in the country and was not born in the country. '
			'Since it expresses a relationship, and refers to one of the '
			'relata (the person), &ldquo;foreigner&rdquo; is a '
			'relational'
			' noun.'
		)
	},
	'stranger': {
		'query': 'stranger',
		'correct': 'relational',
		'response': (
			'&ldquo;Stranger&rdquo; expresses a relationship between '
			'people who are not familiar to each other, and when used it '
			'refers to one of those people (one of the relata).'
			'Therefore &ldquo;foreigner&rdquo; is a '
			'relational'
			' noun.'
		)
	},
	'manufacturer': {
		'query': 'manufacturer',
		'correct': 'relational',
		'response': (
			'&ldquo;Manufacturer&rdquo; expresses the relationship '
			'between a product and the entity (usually a company) that '
			'makes it.  While that meaning is relational, '
			'&ldquo;manufacturer&rdquo; can also be used in a generic '
			'non-relational sense, as in '
			'&ldquo;Manufacturers and exporters will be negatively '
			'affected this quarter&rdquo;.  We think the relational usage '
			'is most common, and judge &ldquo;manufacturer&rdquo; to be '
			'a '
			'relational'
			' noun.'
		)
	},
	'producer': {
		'query': 'producer',
		'correct': 'relational',
		'response': (
			'&ldquo;Producer&rdquo; expresses the relationship '
			'between a product and the entity (usually a company) that '
			'produces it.  It can also describe the role in the creation '
			'of a film.  Both usages are relational.  While there is also '
			'a generic non-relational usage, as in &ldquo;Producers and '
			'consumers are co-dependent in the economy,&rdquo; we think '
			'the relational usages are most commonly used, and judge '
			'&ldquo;producer&rdquo; to be '
			'a '
			'relational'
			' noun.'
		)
	},
	'discovery': {
		'query': 'discovery',
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Discovery&rdquo; can refer to the act of '
			'discovering, as in &ldquo;The discover of Arsenic&rdquo;, '
			'or it can refer to the thing discovered, as in '
			'&ldquo;The law of the photoelectric effect is one of '
			'Einstein&rsquo;s important discoveries.&rdquo;. '
			'As shown, when used to refer to the thing discovered, it can '
			'be used to express the relationship between the person '
			'who did the discovering and the thing discovered. '
			'However, we think that the relational usage is not very '
			'common, and '
			'so judge &ldquo;discovery&rdquo; to be '
			'partly relational'
			'.'
		)
	},
	'creation': {
		'query': 'creation',
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Creation&rdquo; can refer to the act of creating, or '
			'to the thing created, as in '
			'&ldquo;According to the Bible,&rsquo; the Universe is '
			'God&rsquo;s creation.&rdquo;  As shown, when it refers '
			'to the thing created, it can be used to establish the '
			'relationship between the creator and the thing created. '
			"We don't think the relational usage is dominant, so we "
			'judge creation to be only '
			'partly relational'
			'.'
		)
	},
	'loser': {
		'query': 'loser',
		'correct': 'relational',
		'response': (
			'&ldquo;Loser&rdquo; can be used relationally to designate an '
			'entrant in '
			'a competition that lost, as in: '
			'&ldquo;The losers were given a certificate for their '
			'participation as a consolation.&rdquo;, or it can be used '
			'in a pejorative, non-relationally way, as in ' 
			'&ldquo;What a bunch of losers, I\'m embarrassed to be seen '
			'with them.&rdquo; '
			'We think the relational usage is more common, so judge '
			'&ldquo;loser&rdquo; to be '
			'relational'
			'.'
		)
	},
	'pair': {
		'query': 'pair',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Pair&rdquo; denotes a quantity or grouping, and '
			"doesn't express a relationship, so it is "
			'non-relational'
			'.'
		)
	},
	'supplier': {
		'query': 'supplier', 
		'correct': 'relational',
		'response': (
			'&ldquo;Supplier&rdquo; is used to establish a relationship '
			'between companies, where the supplier supplies parts or '
			'materials to the other company (the relatum).  Therefore it '
			'is '
			'relational'
			'.'
		)
	},
	'energy': {
		'query': 'energy',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Energy&rdquo; does not indicate a relationship, so it '
			'is '
			'non-relational'
			'.'
		)
	},
	'rapport': {
		'query': 'rapport',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Rapport&rdquo; is a bridge-noun: '
			'&ldquo;the rapport between the students and teacher was '
			'breaking down.&rdquo;  Therefore it is '
			'non-relational'
			'.'
		)
	},
	'captain': {
		'query': 'captain', 
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Captain&rdquo; can either refer to the rank, or to '
			'the leadership position in command of a ship. The former '
			'usage is non-relational, while the latter is relational. '
			"The relational usage doesn't seem distinctly more common, so "
			'we label &ldquo;captain&rdquo; '
			'partly relational'
			'.'
		)
	},
	'dispute': {
		'query': 'dispute', 
		'correct': 'non-relational',
		'response': (
			'&ldquo;Dispute&rdquo; is a bridge-noun: '
			'&ldquo;the dispute between business partners would not be '
			'resolved easily.&rdquo; Therefore it is '
			'non-relational'
			'.'
		)
	},
	'contract': {
		'query': 'contract',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Contract&rdquo; is a bridge-noun: '
			'&ldquo;the contract between the parties is binding&rdquo;. '
			'Therefore it is '
			'non-relational'
			'.'
		),
	},
	'reporter': {
		'query': 'reporter', 
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Reporter&rdquo; is a role, which can certainly be '
			'used non-relationally, as in : &ldquo;What do I do for a '
			"living? I'm a reporter.&rdquo; "
			'However, it is very often used to designate a person&rsquo;s '
			'affiliation to a particular news organization, as in: '
			'&ldquo;Neha Thirani Bagri is a reporter with The New York '
			"Times.&rdquo; It isn't clear which is more "
			'common, so we consider &ldquo;reporter&rdquo; to be '
			'partly relational'
			'.'
		)
	},
	'gender': {
		'query': 'gender', 
		'correct': 'non-relational',
		'response': (
			"Gender is a property, so it is "
			"non-relational"
			'.'
		)
	},
	'reconnaissance': {
		'query': 'reconnaissance',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Reconnaissance&rdquo; does not indicate a '
			'relationship.  Therefore it is '
			'non-relational'
			'.'
		)
	},
	'tip': {
		'query': 'tip',
		'correct': 'partly-relational',
		'response': (
			'Tip is partly-relational.  It can describe the furthest '
			'point along some extended part of an object.  Or it can refer '
			"to pointers, as in &ldquo;tips and tricks&rdquo;.  We don't "
			'think the '
			'relational usage is the most common, so we '
			'judge tip to be '
			'partly relational'
			'.'
		)
	},
	'war': {
		'query': 'war', 
		'correct': 'non-relational',
		'response': (
			'&ldquo;War&rdquo; is a bridge-noun: '
			'&ldquo;The war between the '
			'countries raged on for decades&rdquo;. '
			'Therefore it is '
			'non-relational'
			'.'
		)
	},
	'operator': {
		'query': 'operator', 
		'correct': 'non-relational',
		'response': (
			'&ldquo;Operator&rdquo; is a role, designating someone '
			'controlling machinery or a production process. As a role, '
			'it is conceivable that it could be used relationally, '
			'however that seems like a stretch, and such a usage is '
			'probably very rare.  Therefore, we consider '
			'&ldquo;operator&rdquo; to be '
			'non-relational'
			'.'
		)
	},
	'copper': {
		'query': 'copper', 
		'correct': 'non-relational',
		'response': (
			'&ldquo;Copper&rdquo; does not indicate a relationship.  '
			'Therefore it is '
			'non-relational'
			'.'
		),
	},
	'nail': {
		'query': 'nail', 
		'correct': 'non-relational',
		'response': (
			'&ldquo;Nail&rdquo; does not indicate a relationship.  '
			'Therefore it is '
			'non-relational'
			'.'
		),
	},
	'realization': {
		'query': 'realization',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Realization&rdquo; does not indicate a relationship.  '
			'Therefore it is '
			'non-relational'
			'.'
		)
	},
	'writer': {
		'query': 'writer', 
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Writer&rdquo; has a few meanings: it can refer to the '
			'vocation, it can '
			'refer to the role at a newspaper company, or it can refer to '
			'the person who wrote the screenplay of a movie (although '
			'screenwriter isolates that meaning). The first non-relational '
			'usage is probably the most common. The latter two usages, as '
			'roles, can be used relationally, and it is not excessively '
			'rare to see them used that way. '
			'Therefore, we consider &ldquo;writer&rdquo; to be '
			'partly relational'
			'.'
		)
	},
	'noon': {
		'query': 'noon',
		'correct': 'non-relational',
		'response': (
			'Noon is '
			'non-relational'
			' because it does not denote a relationship.'
		)
	},
	'head': {
		'query': 'head', 
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Head&rdquo; has a few prominent meanings: it can '
			'signify the body part, it can signify a distal and bulbous '
			'part of an object, or it can signify a position of '
			'leadership.  The latter two meanings are relational, but the '
			'first meaning is probably more common.  We therefore label '
			'&ldquo;head&rdquo; as '
			'partly relational'
			'.'
		)
	},
	'neck': {
		'query': 'neck', 
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Neck&rdquo; can mean the body part of an animal '
			'joining the head to the body, or it can mean an elongate '
			'constricted part of an object, such as a bottle or vase. '
			'The former usage is '
			'non-relational'
			', while the latter is '
			'relational.  We think that the latter usage is less '
			'frequent, so we label &ldquo;neck&rdquo; as '
			'partly relational'
			'.'
		)
	},
	'delegate': {
		'query': 'delegate',
		'correct': 'relational',
		'response': (
			'&ldquo;Delegate&rdquo; is '
			'relational'
			', because a delegate is defined in '
			'terms of that which it is appointed for (e.g. a delegate of '
			'the internal committee).'
		)
	},
	'investor': {
		'query': 'investor', 
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Investor&rdquo; is a role. It can be used to '
			'designate someone who frequently invests in companies and '
			'projects generally, or it can be used to signify the '
			'relationship between the person who invests and the '
			'company / project '
			'in which she invests&mdash;the latter being a relational '
			'usage.  We '
			'think that the relational usage is roughly on par with the '
			'generic usage, and since it is not clearly more common, we '
			'choose to label &ldquo;investor&rdquo; as '
			'partly relational'
			'.'
		)
	},
	'athlete': {
		'query': 'athlete',
		'correct': 'non-relational',
		'response': (
			'Athlete is '
			'non-relational'
			'.  It is not essentially defined as '
			'a relationship to something / someone else.'
		)
	},
	'mathematician': {
		'query': 'mathematician',
		'correct': 'non-relational',
		'response': (
			'Mathematician is '
			'non-relational'
			', it is not essentially '
			'defined as a relationship to something / someone else.'
		)
	},
	'scientist': {
		'query': 'scientist',
		'correct': 'non-relational',
		'response': (
			'Scientist is '
			'non-relational'
			'.  It is not essentially defined '
			'as a relationship to something / someone else.'
		)
	},
	'predecessor': {
		'query': 'predecessor',
		'correct': 'relational',
		'response': (
			'Predecessor is inherently '
			'relational'
			', being defined as the '
			'thing which came before something else.'
		)
	},
	'mayor': {
		'query': 'mayor',
		'correct': 'relational',
		'response': (
			'Mayor is '
			'relational'
			', expressing the relationship between a city and its '
			'administrative leader.'
		)
	},
	'blacksmith': {
		'query': 'blacksmith', 
		'correct': 'non-relational',
		'response': (
			'Blacksmith is '
			'non-relational'
			'.  Although it is a vocation, '
			"it isn't essentially defined in terms of a relationship to "
			'something else.'
		)
	},
	'partner': {
		'query': 'partner',
		'correct': 'relational',
		'response': (
			'Partner is inherently '
			'relational'
			', describing a person who '
			"is cooperating with someone else."
		)
	},
	'assistant': {
		'query': 'assistant',
		'correct': 'relational', 
		'response': (
			'Assistant is inherently '
			'relational'
			', describing a person who '
			"is supporting someone else."
		)
	}
}

TERNARY_OPTIONS = [
	{
		'text': 'almost never relational',
		'class': 'non-relational'
	}, {
		'text': 'occasionally relational',
		'class': 'partly-relational'
	}, 
	{
		'text': 'usually relational', 
		'class': 'relational'
	}
]
BINARY_OPTIONS = [
	{'text': 'non-relational', 'class': 'non-relational'}, 
	{'text': 'relational', 'class': 'relational'}
]

def make_quiz_questions(grouping, arity):
	"""
	Make the HTML for practice questions appearing in the Crowdflower task.
	``grouping`` should be one of the keys in the global dict 
	``question_groupings``, which is used to select a subset of questions
	to be created.
	"""
	random.seed(0)
	questions_container = div()
	randomized_questions = random.sample(
		question_groupings[grouping], len(question_groupings[grouping])
	)
	for i, query in enumerate(randomized_questions):
		spec = question_specs[query]
		questions_container.appendChild(make_quiz_question(
			i, spec, arity, grouping
		))

	return unescape(questions_container.toprettyxml())

def unescape(string):
	return string.replace('&amp;', '&')

def make_quiz_question(i, spec, arity, grouping):
	question_wrapper = span({'class':'quiz'})

	# Make query part
	query_line = question_wrapper.appendChild(span({'class':'queryline'}))
	query = query_line.appendChild(span({'class': 'query'}))
	query_word = query.appendChild(span({'class':'query-word'}))
	query_word.appendChild(text(spec['query']))

	# Answer
	correct = query_line.appendChild(span({'class':'correct-answer'}))
	correct.appendChild(text(spec['correct']))

	# Make and append options
	option_wrapper = query_line.appendChild(div({'class':'option-wrapper'}))
	for option in make_options(i, arity, grouping):
		option_wrapper.appendChild(option)

	# Make response portion
	response_line = question_wrapper.appendChild(span({'class':'responseline'}))
	response = response_line.appendChild(span({'class':'response'}))
	prefix = response.appendChild(span({'class':'prefix'}))
	prefix.appendChild(text('prefix'))
	use_response = spec['response']
	response.appendChild(text(use_response))

	return question_wrapper


def make_options(i, arity, grouping):
	options = []

	option_types = BINARY_OPTIONS if arity=='binary' else TERNARY_OPTIONS
	for j, option in enumerate(option_types):
		option_elm = span({'class':option['class']+'-option option'})
		option_elm.appendChild(element(
			'input', 
			{
				'type':'radio',
				'id':'%s.%s.%s'%(grouping, i,j),
				'name':'%s.%s'%(grouping, i)
			}
		))
		label = option_elm.appendChild(element(
			'label', {'for':'%s.%s.%s'%(grouping,i,j)}
		))
		label.appendChild(text(option['text']))
		options.append(option_elm)

	return options


DOM = minidom.Document()

def element(tag_name, attributes={}):
    elm = DOM.createElement(tag_name)
    bind_attributes(elm, attributes)
    return elm 

def bind_attributes(element, attributes):
    for attribute in attributes:
        element.setAttribute(attribute, attributes[attribute])
    return element

def div(attributes={}):
    return element('div', attributes)

def span(attributes={}):
    return element('span', attributes)

def text(text_content):
    return DOM.createTextNode(text_content)

