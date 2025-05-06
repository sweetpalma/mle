# Sentiment Analysis
# coding: utf-8
import re

# Download dataset from Kaggle
import kagglehub
df = kagglehub.load_dataset(
  kagglehub.KaggleDatasetAdapter.PANDAS,
  'jp797498e/twitter-entity-sentiment-analysis',
  'twitter_training.csv',
  pandas_kwargs={'encoding': 'ISO-8859-1'},
)

# Pin the dataset column names
df = df[df.columns[[2, 3]]]
df.columns = ['sentiment', 'text']

# Prepare stop words
from spacy.lang.en import stop_words
stop_words = {'.', ',', '!'}.union(stop_words.STOP_WORDS)
def clean(data):
  tokens = [x.lower() for x in re.split(r'\s', str(data))]
  important_tokens = [x for x in tokens if not x in stop_words]
  return ' '.join(important_tokens)

# Clean the dataset
df['text'] = df['text'].map(clean)
df['sentiment'] = df['sentiment'].str.lower()
df = df.dropna()

# Split the training and test datasets
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
x_train = train['text']
y_train = train['sentiment']
x_test = test['text']
y_test = test['sentiment']

# Prepare classification pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([
  ('vectorizer', CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85)),
  ('classifier', LogisticRegression(C=10, solver='lbfgs', max_iter=500)),
])

# Run training
pipeline.fit(x_train, y_train)

# Test classifier results
from sklearn.metrics import classification_report
prediction = pipeline.predict(x_test)
print(classification_report(y_test,prediction))
