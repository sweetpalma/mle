# Sentiment Analysis (Word2Vec + Multi-layered Perceptron)
import pandas as pd
import numpy as np

# This example demonstrates how to use Keras Multi-Layer Perceptron with Word2Vec (Google News).
# Achieved 85% accuracy - better than a LinearClassifier with Word2Vec (64%), but still worse 
# than a simple CountVectorizer (92%).

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
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
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

# Seems like the fourth possible class ("Irrelevant") was confusing the model, causing
# too much noise and unrelevant data. Removing it greatly increased the accuracy - from 
# 78% to 85%.

# Split data
# It is done before vectorization to prevent vocabulary contamination and overfitting
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
x_train = train['text']
y_train = train['sentiment']
x_test = test['text']
y_test = test['sentiment']

# Vectorize X
x_train = np.array(train['text'].map(vectorize).tolist())
x_test = np.array(test['text'].map(vectorize).tolist())

# Encode Y
# Transform textual labels into their matrix representation.
#
# Example
# Input: ['Negative', 'Neutral', 'Positive']
# Output: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
from sklearn.preprocessing import LabelBinarizer 
encoder = LabelBinarizer()
encoder.fit(df['sentiment'])
y_train_encoded = pd.DataFrame(encoder.fit_transform(y_train))
y_test_encoded = pd.DataFrame(encoder.transform(y_test))

# Prepare model
# Layer 1: Input layer with shape matching Word2Vec size
# Layer 2, 4: Hidden layers with 128 neurons, non-linear (ReLU) activation
# Layer 3, 5: Dropout layers to reduce potential overfitting
# Layer 5: Output layer with softmax for multi-class probability (sum of neurons is 1.0)
from tensorflow.keras import layers, Sequential
num_classes = 3 # Number of possible sentiments
model = Sequential([
  layers.Input(shape=(300,)),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.1),
  layers.Dense(num_classes, activation='softmax')
])

# Originally, there were only three layers - input, hidden, and output. But stacking two
# more hidden layers helped to uplift the original 82% accuracy to 85%. That happened due
# to increased model capacity, and (theoretically) hierarchical feature learning.

# Compile
# Uses Adam optimizer 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
# Performs 50 passes (epochs), re-evaluating the weights after each batch (32 samples).
# Initially it worked fine with 10 epochs, but dropout layer needed more passes.
model.fit(x_train, y_train_encoded, epochs=50, batch_size=32, validation_split=0.1) 

# Test
loss, accuracy = model.evaluate(x_test, y_test_encoded, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
