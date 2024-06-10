# First NLP Kaggle project

## Intro | Objectives | Main goals
Natural Language Processing (NLP) is a critical area of artificial intelligence (AI) that enables computers to understand, interpret, and respond to human language. Therefore, this project has the goal of studying the Kaggle competition "Natural Language Processing with Disaster Tweets" as way of understanding more about NLP.

## Studying the data
To solve this problem, as a common data science project, it is essential to understand the data. As a NLP problem, the data has to be cleaned and processed to get studied. With the cleaning of words 'http', '@user', etc.

Then, the following steps were as listed:

1. Plot word-clouds to get as much information to get a visual representation of real disaster tweets and the other tweets.
1. Get the top 10 words of each group.
1. Plot the most common keywords for the 


## DL model
After sudying the problem's DF it was essential to start building the DL model from the information presented in the last section. Tensorflow and Keras was used to build this model, as Keras provides an user-friendly interface to work with TensorFlow.

- Using padding as it is common in NLP DL models to get a uniform input size, efficient batch processing, and compatibility with libraries.
### Bidirectional LSTM with Conv1D Model

This Python script builds and trains a sequential model using Keras, incorporating an embedding layer, convolutional layers, bidirectional LSTM layers, and dense layers with dropout for regularization.

### Bidirectional LSTM layer

### Dense layer

### Dropout layer

### ReLU ACTIVATION 
Rectifier 

$$
f(x)=x^+=max(0,x)=\frac{x+|x|}{2} = \begin{cases}
    x, & \text{if } x > 0 \\
    0, & \text{otherwise}
\end{cases}
$$

$$ReLU(x) = max(0,) $$


### LSTM 
## Building DL model
### Code
 1. Importing the models, layers and optimizers used on the deep learning model.

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Dense, Dropout
from keras.optimizers import Adam
```

2. Starting the sequential model and adding the first layer, the embedding layer. That is commonly used in natural language processing (NLP) tasks. It turns positive integers into dense vectors of fixed size. Converting words (represented as integers) into vectors, building the input of the neural network.

- input_dim=num_words: This is the size of the vocabulary. It's the number of unique words (or tokens) in your dataset. Each word in the vocabulary will be represented by a unique integer index.

- output_dim=500: This is the size of the dense embedding vector. Each word will be represented by a vector of this length. In this case, each word will be mapped to a 500-dimensional vector.

- input_length=max_sequence_length: This is the length of input sequences. It defines the length of input sequences to be expected by the model. All sequences in a batch will need to have the same length (padding or truncating may be necessary).

- trainable=True: This means that the weights of the Embedding layer will be updated during the training process. If set to False, the weights will be fixed and not updated during training.


```python
# Assuming num_words and max_sequence_length are defined
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=500, input_length=max_sequence_length, trainable=True))
```
3. 1D convolutional beural network are used in NLP, as they work well with sequential data, as is the case in NLP. 
```python
# Add Conv1D layer
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
```
4.
```python
# Use Bidirectional LSTM
model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2)))
```
5.
```python
# Add more Dense layers with regularization
model.add(Dense(1024, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
```
6.
```python
# Final output layer
model.add(Dense(2, activation='softmax'))

# Compile model with a different optimizer and learning rate
optimizer = Adam(learning_rate=1e-4)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```