# Sentiment Analysis

# This example shows a simple CountVectorizer with n-grams and Linear Regression.
# Achieved 90% accuracy using GridSearchCV.

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
def clean(data):
  return remove_stopwords(str(data).lower())

# Assign column names
df = df[df.columns[[2, 3]]]
df.columns = ['sentiment', 'text']

# Clean data
df['text'] = df['text'].map(clean)
df['sentiment'] = df['sentiment'].str.lower()
df = df.dropna()

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
# Output: { features: ['this', 'really', 'sucks', ...], result: [1, 2, 1, ...] }
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2))

# Prepare classifier
# Slightly bigger 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', max_iter=1500)

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

# Parameter "classifier__C"
# Regularization strength of the LogisticRegression classifier.
# Smaller values make it stronger (less prone to overfitting), bigger - weaker (able to capture more 
# nuances in data).
#
# Parameter "vectorizer__min_df"
# Minimum document frequency - ignore terms that appear in fewer than 'min_df' documents.
# Smaller values may lead to huge noisy vocabularies, bigger may result in losing specific signals.
#
# Parameter "vectorizer__max_df"
# Maximum document frequency - ignore terms that appear in more than 'max_df' % of documents.
# Smaller values exclude more common terms (good for noise reduction), but too small may result 
# in losing important common signals (underfitting).
param_grid = {
  'classifier__C': [0.1, 1, 10],
  'vectorizer__min_df': [1, 2, 3, 5],
  'vectorizer__max_df': [0.85, 0.90, 0.95, 1.0],
}

# Run training with cross-validation
from sklearn.model_selection import RandomizedSearchCV
grid = RandomizedSearchCV(pipeline, verbose=3, param_distributions=param_grid, scoring='accuracy', n_jobs=-1, cv=3)
grid.fit(x_train, y_train)

# Determine best possible parameters
best_params = grid.best_params_
best_model = grid.best_estimator_

# Test classifier results
from sklearn.metrics import classification_report
prediction = best_model.predict(x_test)
print(classification_report(y_test, prediction))
print(f'Best parameters: {best_params}')
