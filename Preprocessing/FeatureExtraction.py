# Loading features from dicts
measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Francisco', 'temperature': 18.},
]
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
vec.fit_transform(measurements).toarray()
vec.get_feature_names_out()

# DictVectorizer
movie_entry = [{'category': ['thriller', 'drama'], 'year': 2003},
               {'category': ['animation', 'family'], 'year': 2011},
               {'year': 1974}]
vec.fit_transform(movie_entry).toarray()
vec.get_feature_names_out()
vec.transform({'category': ['thriller'],
               'unseen_feature': '3'}).toarray()

#TFidf Trasformer
vec = DictVectorizer()
pos_vectorized = vec.fit_transform(pos_window)
pos_vectorized
pos_vectorized.toarray()
vec.get_feature_names_out()

#Feature hashing
def token_features(token, part_of_speech):
    if token.isdigit():
        yield "numeric"
    else:
        yield "token={}".format(token.lower())
        yield "token,pos={},{}".format(token, part_of_speech)
    if token[0].isupper():
        yield "uppercase_initial"
    if token.isupper():
        yield "all_uppercase"
    yield "pos={}".format(part_of_speech)

raw_X = (token_features(tok, pos_tagger(tok)) for tok in corpus)
hasher = FeatureHasher(input_type='string')
X = hasher.transform(raw_X)

#Text feature extraction
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)
X
analyze = vectorizer.build_analyzer()
analyze("This is a text document to analyze.") == (
    ['this', 'is', 'text', 'document', 'to', 'analyze'])
vectorizer.get_feature_names_out()

X.toarray()
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
analyze('Bi-grams are cool!') == (
    ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])
X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
X_2
feature_index = bigram_vectorizer.vocabulary_.get('is this')
X_2[:, feature_index]

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
transformer

#Limitations of the Bag of Words representation
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
counts = ngram_vectorizer.fit_transform(['words', 'wprds'])
ngram_vectorizer.get_feature_names_out()
counts.toarray().astype(int)

ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5, 5))
ngram_vectorizer.fit_transform(['jumpy fox'])
ngram_vectorizer.get_feature_names_out()

ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5))
ngram_vectorizer.fit_transform(['jumpy fox'])
ngram_vectorizer.get_feature_names_out()

from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer(n_features=10)
hv.transform(corpus)

hv = HashingVectorizer()
hv.transform(corpus)

def my_tokenizer(s):
    return s.split()
vectorizer = CountVectorizer(tokenizer=my_tokenizer)
vectorizer.build_analyzer()(u"Some... punctuation!") == (
    ['some...', 'punctuation!'])

#Image feature extraction
import numpy as np
from sklearn.feature_extraction import image

one_image = np.arange(4 * 4 * 3).reshape((4, 4, 3))
one_image[:, :, 0]  # R channel of a fake RGB picture

patches = image.extract_patches_2d(one_image, (2, 2), max_patches=2,
    random_state=0)
patches.shape
patches[:, :, :, 0]

patches = image.extract_patches_2d(one_image, (2, 2))
patches.shape
patches[4, :, :, 0]

reconstructed = image.reconstruct_from_patches_2d(patches, (4, 4, 3))
np.testing.assert_array_equal(one_image, reconstructed)

five_images = np.arange(5 * 4 * 4 * 3).reshape(5, 4, 4, 3)
patches = image.PatchExtractor(patch_size=(2, 2)).transform(five_images)
patches.shape
