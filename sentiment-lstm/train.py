# Sentiment Analysis (Word2Vec + LSTM)
import pandas as pd
import numpy as np

# This example demonstrates how to use Keras LSTMS with Word2Vec as an embedding layer.
# Achieved 92% accuracy - reaching the same score as LinearClassifier, but using a
# completely different technique.

# Articles used:
# https://medium.com/@hsinhungw/understanding-word-embeddings-with-keras-dfafde0d15a4

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

# Prepare word2vec model
from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format(path, binary=True)

# Pin column names
df = df[df.columns[[2, 3]]]
df.columns = ['sentiment', 'text']

# Pin column data types
df['text'] = df['text'].astype(str)
df['sentiment'] = df['sentiment'].astype(str)
df = df.dropna()

# Clean data
df = df.loc[df['sentiment'] != 'Irrelevant']

# Prepare tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])

# Split data
# It is done before vectorization to prevent vocabulary contamination and overfitting
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
x_train = train['text']
y_train = train['sentiment']
x_test = test['text']
y_test = test['sentiment']

# Determine maximal sequence length
maxlen = 96

# Tokenize and pad X
from tensorflow.keras.utils import pad_sequences
x_train = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=maxlen)
x_test = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=maxlen)

# Encode Y
from sklearn.preprocessing import LabelBinarizer 
encoder = LabelBinarizer()
encoder.fit(df['sentiment'])
y_train_encoded = pd.DataFrame(encoder.fit_transform(y_train))
y_test_encoded = pd.DataFrame(encoder.transform(y_test))

# Prepare embedding matrix
embedding_matrix_shape = (len(tokenizer.word_index) + 1, wv.vector_size)
embedding_matrix = np.zeros(shape=embedding_matrix_shape)
for word, index in tokenizer.word_index.items():
  if word in wv:
    embedding_matrix[index] = wv.get_vector(word)
  else:
    embedding_matrix[index] = np.zeros(wv.vector_size)

# Prepare model
#
# Layer 1:
# Turns padded token index sequences into their word2vec representations, preserving
# their order. Essentially, a trainable look-up table.
#
# Layer 2: 
# Main learning layer - unlike Dense layer, each unit here has an internal state and
# data from the previous layer step-by-step, allowing model to perceive the possible 
# word order, making predictions more accurate.
#
# Layer 3: 
# Dropout layer to reduce potential overfitting.
#
# Layer 4: 
# Output layer with softmax for multi-class probability (sum of neurons is 1.0).
from tensorflow.keras import layers, Sequential
num_classes = len(encoder.classes_)
model = Sequential([
  layers.Embedding(
    weights=[embedding_matrix],
    input_dim=embedding_matrix_shape[0], 
    output_dim=embedding_matrix_shape[1],
    trainable=True,
  ),
  layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
  layers.Dropout(0.2),
  layers.Dense(num_classes, activation='softmax'),
])

# Prepare early stopping
# Prevents model from overfitting and stops when learning plateau is reached
from tensorflow.keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train_encoded, epochs=20, batch_size=128, validation_split=0.1, callbacks=[earlystop]) 

# Test
loss, accuracy = model.evaluate(x_test, y_test_encoded, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
