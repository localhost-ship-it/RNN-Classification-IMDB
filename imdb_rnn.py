import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense #dense of hidden layer, simple rnn for rnn layer, embedding for word embedding layer


#SimpleRNN for IMDB sentiment classification

#load the imdb dataset
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#inspect the sample review and its label
print(x_train[0])
print(y_train[0]) # 1 means positive review, 0 means negative review

##mapping the word index to actual words (for understanding purpose)

word_index = imdb.get_word_index()
#print(word_index) #dictionary of words with their corresponding index

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(reverse_word_index)

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
#print(decoded_review)

##padding the sequences to make them of equal length
maxlen = 500
x_train = sequence.pad_sequences(x_train, maxlen=maxlen) #default is pre padding
x_test = sequence.pad_sequences(x_test, maxlen=maxlen) 
#print(x_train.shape, x_test.shape)
#print(x_train[0])


##training the SimpleRNN model

#vocab_size → total number of unique tokens (e.g., size of your tokenizer’s vocabulary)

#128 → dimension of the embedding vectors

#max_len → length of each input sequence (e.g., after padding)

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=maxlen)) #128 is the dimension (features , could be any number) of embedding vector
#this embedding layer will convert each word index to a 128 dimension vector
model.add(SimpleRNN(128, activation='tanh')) #128 is number of units (neurons) in the rnn layer
model.add(Dense(1, activation='sigmoid')) #output layer for binary classification

model.build(input_shape=(None, maxlen))
model.summary()


#create an instance of earlystopping callback
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#restore_best_weights=True will restore the model weights from the epoch with the best value of the monitored quantity.

#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#compilation is for preparing the model for training by configuring the optimizer, loss function, and metrics to monitor

#train the model with early stopping
history = model.fit(x_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping]
            )
#validation_split=0.2 means 20% of training data will be used for validation

#save the model
model.save('imdb_rnn_model.h5')

