#import libraries and load model
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
import tensorflow.keras.preprocessing.sequence as sequence
from keras.models import load_model

#load the imdb dataset
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

##load the pretrained model
model = load_model('imdb_rnn_model.h5')

#helper functions
#function to decode the review
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

#function to pre-process the input review
def preprocess_review(review, maxlen=500):
    #tokenize the review
    encoded_review = []
    for word in review.split():
        if word in word_index and word_index[word] < 10000:
            encoded_review.append(word_index[word] + 3) #+3 bcz 0,1,2 are reserved indices
        else:
            encoded_review.append(2) #2 is for unknown words
    #pad the review
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review

#predict sentiment for a new review
def predict_sentiment(review):
    processed_review = preprocess_review(review)
    prediction = model.predict(processed_review)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    return sentiment, prediction[0][0]


#streamlit app for sentiment prediction
import streamlit as st
st.title("IMDB Movie Review Sentiment Prediction")
st.write("Enter a movie review to predict its sentiment.")
#input field
user_review = st.text_area("Movie Review", "Type your review here...")

if st.button("Predict Sentiment"):
    sentiment, confidence = predict_sentiment(user_review)
    st.write(f"Predicted Sentiment: **{sentiment}** (Confidence: {confidence:.4f})")

else:
    st.write("Please enter a review and click the 'Predict Sentiment' button.")

    