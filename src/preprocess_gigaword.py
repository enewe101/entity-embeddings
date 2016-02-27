'''
Preprocess gigaword, a large corpus of news stories.
1) Only take articles of type "story"
2) Detect and eliminate directives to editors, which appear as a series of
	all-caps words.
3) Split the data into separate files for each article.
	a) Only put the plain text into those files
	b) Extract metadata for articles and Use a progress tracker to 
		associate metadata to the article files.
4) Convert SGML entities &amp; &gt; &lt;
5) Eliminate spanish articles and articles misclassified as being of type
	story using the corrections provided as metadata (spanish.file-doc.map
	and other.file-doc.map)
6) Keep the headline?
'''


