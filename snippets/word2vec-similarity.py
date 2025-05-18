# Word Similarity (Word2Vec)

# Download dataset from Kaggle
import kagglehub
path = kagglehub.dataset_download(
  'leadbest/googlenewsvectorsnegative300', 
  path='GoogleNews-vectors-negative300.bin.gz'
)

# Prepare word2vec model
from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format(path, binary=True)
pairs = [
    ('car', 'minivan'),   # a minivan is a kind of car
    ('car', 'bicycle'),   # still a wheeled vehicle
    ('car', 'airplane'),  # ok, no wheels, but still a vehicle
    ('car', 'cereal'),    # ... and so on
    ('car', 'communism'),
]

# Print word similarity
for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))