# automatic_literary_analysis
This is a little script that heaps a few methods to automatically analyse a literary text.
This can be used for machine learning feature engineering or simply for authors or scholars wanting to see how one book differs from another.
The features are by no means exhaustive. I’ll try to add some more.


# Requirements:

Python 3
spacy
nltk
numpy
collections

# How to use:
The following code is under a class. All you need to do is add a name and the path to your favourite book’s url to the class.

```
name = ‘Cinderilla’
url= ‘mypath’
book = Literary_analysis(name, url )

```
then you can either call methods you find interesting or call ‘oneforall’ which will treat the whole text.

```
book.oneforall()
print(book.__dict__)
{'book': 'cind', 'url': 'cinderilla.txt', 'frac_rep': 0.014265335235378032, 't_score': 13.846808510638297, 'fraction_gradable': 0.04797047970479705, 'verb_ratio': 0.15611555009219422, 'punct_ratio': 0.1253841425937308, 'symbol_ratio': 0.0, 'adjective_ratio': 0.08328211432083589, 'conjunction_ratio': 0.0, 'number_ratio': 0.009834050399508297, 'determinent_ratio': 0.0636140135218193, 'adverb_ratio': 0.06023355869698832, 'foreign_ratio': 0.001229256299938537, 'preposition_ratio': 0.08451137062077443, 'noun_ratio': 0.13214505224339274, 'proper_noun_ratio': 0.06515058389674247, 'particle_ratio': 0.023355869698832205, 'space_ratio': 0.08543331284572833, 'interjection_ratio': 0.004609711124769515, 'frac_unusual': 0.03525046382189239, 'simple_clause_ratio': 0.1826086956521739, 'complex_compound_clause_ratio': 0.2, 'complex_clause_ratio': 0.6260869565217392, 'compound_clause_ratio': 0.22608695652173913, 'lexical_diversity': 0.0061096291196971, 'size_sentence_mean': 28.295652173913044, 'size_sentence_var': 552.01693761814749, 'avg_syllable': 0.753841425937308, 'flesh': 22.310015365703755, 'mean_sentences_aliteration': 0.40204158790170136, 'var_sentences_aliteration': 0.17256688476670679, 'passive_voice': 0.23478260869565218}

```

# Features:

. Alliteration: frequency of alliterations in a text. An alliteration is the occurrence of the same letter or sound at the beginning of adjacent or closely connected words.
returns mean_sentences_aliteration  and  var_sentences_aliteration

.flesch_reading_test; [the flesch kincaid](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests) reading test score. The lower the score the more difficult the style of the writer.
returns flesh
.Language diversity returns the frequency of unique words in the text
returns lexical_diversity

.size_sentence simple: simple mean and variance of the number of words per sentence.
returns size_sentence_mean
size_sentence_var

.syllable: the average number of syllables per words
returns: avg_syllable

.pos: returns the various frequencies of spacy’s part of speech in the text.
returns : verb_ratio,
punct_ratio,
symbol_ratio adjective_ratio,
conjunction_ratio,
number_ratio,
determinent_ratio,
adverb_ratio,
foreign_ratio,
preposition_ratio,
noun_ratio,
proper_noun_ratio,
particle_ratio,
space_ratio,
interjection_ratio

.ratio_clause : returns the ratio throughout the text of simple clause, complex clause, compound clause, and complex-compound clauses to all sentences.
returns simple_clause_ratio
complex_compound_clause_ratio
complex_clause_ratio
compound_clause_ratio

. un_and_usual_words returns the ratio of unusual words in the text to total words
returns frac_unusual

.gradable checks the ratio of adjective having gradable words associated to total adjectives. The underlying thought is that writers using phrases such as ‘very big’ rather than ‘enormous’ have a poorer vocabulary.
returns fraction_gradable

.tunits score as explained [here]( https://en.wikipedia.org/wiki/T-unit)
returns t_score

.repetition: fraction of repeating words over total words within a window. The window is fixed at 20 words, but you are welcome to expand or shrink it.
returns frac_rep

.passive_markers: the fraction of passive sentences to total sentences.
returns passive_voice
