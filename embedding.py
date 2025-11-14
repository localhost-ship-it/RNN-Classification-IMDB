
from tensorflow.keras.preprocessing.text import one_hot

sent = ['the glass of milk',
        'the glass of juice',
        'the cup of tea',
        'I am a good boy'
        'I am a good developer'
        'understand the meaning of words'
        'your videos are good']

vocab_size = 10000

encoded_sentences = [one_hot(i, vocab_size) for i in sent]
print(encoded_sentences)

##word embedding representation

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential


#-> One important thing to note is we need to make all the sentences in dataset of equal 
# size otherwise we will not be able to train it in our RNN bcz all the words that 
# will be going it will be going for a fixed number of timestamp based on the sentence size that's
#why we have imported pad_sequences from keras.preprocessing.sequence to make all the sentences of equal size

sent_length = 8
#padded_sentences = pad_sequences(encoded_sentences, maxlen=sent_length, padding='post')
padded_sentences = pad_sequences(encoded_sentences, maxlen=sent_length, padding='pre')
print(padded_sentences)

##feature representation
dim = 10
model = Sequential()
model.add(Embedding(vocab_size, dim, input_length=sent_length))
model.compile('adam', 'mse')
print(model.summary())
