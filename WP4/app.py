import streamlit as st
import tensorflow as tf
from tensorflow import keras

# Load the pre-trained model
model = keras.models.load_model("movie_review_model.h5")  # Replace with the actual path to your saved model

# Create a Streamlit app
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review, and we'll predict its sentiment!")

# User input for text
user_input = st.text_area("Enter a movie review:")

if st.button("Classify"):
    if user_input:
        # Preprocess the user input
        max_sequence_length = 10000  # Use the same sequence length as in training
        word_index = keras.datasets.imdb.get_word_index()
        user_input = user_input.lower().split()
        user_input = [word_index.get(word, 2) for word in user_input]  # Use 2 for out-of-vocabulary words
        user_input = keras.preprocessing.sequence.pad_sequences([user_input], value=0, padding='post', maxlen=max_sequence_length)

        # Make predictions
        prediction = model.predict(user_input)
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        confidence = round(float(prediction[0, 0]) * 100, 2)
        st.success(f"{sentiment} sentiment with {confidence}% confidence.")
    else:
        st.warning("Please enter a movie review.")
