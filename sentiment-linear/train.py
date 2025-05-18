# Sentiment Analysis (Linear Regression)

# This example demonstrates how to use Linear Regression with CountVectorizer.
# Achieved 92% accuracy using RandomizedSearchCV (check `train_cross.py`).

# Articles used:
# https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c/
# https://towardsdatascience.com/linear-regression-explained-1b36f97b7572/

# Download sentiment dataset
import kagglehub
df = kagglehub.dataset_load(
  kagglehub.KaggleDatasetAdapter.PANDAS,
  'jp797498e/twitter-entity-sentiment-analysis',
  'twitter_training.csv',
  pandas_kwargs={'encoding': 'ISO-8859-1'},
)

# Prepare the cleaning function
# It performs text lowering and common (stop) word removal.
from gensim.parsing.preprocessing import remove_stopwords
def clean(text):
  return remove_stopwords(text.lower())

# Pin column names
df = df[df.columns[[2, 3]]]
df.columns = ['sentiment', 'text']

# Pin column data types
df['text'] = df['text'].astype(str)
df['sentiment'] = df['sentiment'].astype(str)
df = df.dropna()

# Clean data
df = df.loc[df['sentiment'] != 'Irrelevant']
df['text'] = df['text'].map(clean)

# Split data
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
x_train = train['text']
y_train = train['sentiment']
x_test = test['text']
y_test = test['sentiment']

# Prepare count vectorizer
# It builds a feature vocabulary first, and then uses it to turn a list of tokens into a 
# vector representing their respective count.
#
# Important:
# N-gram range is extremely important here - it allows the vectorizer to combine 
# some terms for much better accuracy - e.g. "cool" and "not cool" *may* have completely
# opposite meaning with this approach.
#
# Example: 
# Input: 'this sucks. really sucks'
# Output: { features: ['this', 'sucks', 'really', ...], result: [1, 2, 1, ...] }
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)

# Prepare logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=0.1, solver='lbfgs', max_iter=500)

# Prepare scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)

# Scaler was added because LBFGS may struggle with input features with vastly different 
# scales. It happens because features with larger values (like WordCount results) can 
# dominate the gradient updates, making the optimization path unstable or very slow.

# Prepare pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
  ('vectorizer', vectorizer),
  ('scaler', scaler),
  ('classifier', classifier),
])

# Train
pipeline.fit(x_train, y_train)

# Test
from sklearn.metrics import classification_report
prediction = pipeline.predict(x_test)
print(classification_report(y_test, prediction))
