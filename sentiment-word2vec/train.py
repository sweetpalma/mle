# Sentiment Analysis (Word2Vec)
import numpy as np

# This example demonstrates how to use Linear Regression with Word2Vec (Google News).
# Achieved 64% accuracy - highlights limitations of simple averaging for this task and dataset.

# Articles used:
# https://medium.com/swlh/sentiment-classification-using-word-embeddings-word2vec-aedf28fbb8ca

# Download sentiment dataset
import kagglehub
df = kagglehub.dataset_load(
  kagglehub.KaggleDatasetAdapter.PANDAS,
  'jp797498e/twitter-entity-sentiment-analysis',
  'twitter_training.csv',
  pandas_kwargs={'encoding': 'ISO-8859-1'},
)

# Download word2vec dataset
import kagglehub
path = kagglehub.dataset_download(
  'leadbest/googlenewsvectorsnegative300', 
  path='GoogleNews-vectors-negative300.bin.gz'
)

# Prepare Word2Vec model
from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format(path, binary=True)

# Prepare vectorizer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
def vectorize(data):
  text_without_stopwords = remove_stopwords(data.lower())
  tokens = simple_preprocess(text_without_stopwords, deacc=True)
  token_vectors = [wv.get_vector(x) for x in tokens if x in wv]
  if token_vectors:
    return np.mean(token_vectors, axis=0)
  else:
    return np.zeros(wv.vector_size)

# Pin column names
df = df[df.columns[[2, 3]]]
df.columns = ['sentiment', 'text']

# Pin column data types
df['text'] = df['text'].astype(str)
df['sentiment'] = df['sentiment'].astype(str)
df = df.dropna()

# Clean data
df = df.loc[df['sentiment'] != 'Irrelevant']

# Split data
# It is done before vectorization to prevent vocabulary contamination and overfitting
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
x_train = train['text']
y_train = train['sentiment']
x_test = test['text']
y_test = test['sentiment']

# Vectorize data
x_train = np.array(train['text'].map(vectorize).tolist())
x_test = np.array(test['text'].map(vectorize).tolist())

# Prepare classifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', max_iter=1000)

# Train
classifier.fit(x_train, y_train)

# Test
from sklearn.metrics import classification_report
prediction = classifier.predict(x_test)
print(classification_report(y_test, prediction))
