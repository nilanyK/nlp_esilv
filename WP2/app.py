import os
import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download the "movie_reviews" dataset from NLTK
nltk.download('movie_reviews')

# Load movie reviews from NLTK's movie_reviews corpus
from nltk.corpus import movie_reviews
reviews = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

# Create a DataFrame
df = pd.DataFrame(reviews, columns=['text', 'sentiment'])

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf_vectorizer.fit_transform(df['text'].apply(' '.join))

# Split the data into training and testing sets
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, df['sentiment'], test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
svm_classifier = SVC()
svm_classifier.fit(X_train_tfidf, y_train)

# Evaluate the SVM classifier
y_pred_svm = svm_classifier.predict(X_test_tfidf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Streamlit app
st.title("Movie Review Sentiment Analyzer")

# Input text box for user reviews
user_input = st.text_area("Enter a movie review:")

# Add a "Submit" button
if st.button("Analyze"):
    if user_input:
        # Vectorize the user input using the same TF-IDF vectorizer
        user_input_vectorized = tfidf_vectorizer.transform([user_input])

        # Predict the sentiment
        sentiment = svm_classifier.predict(user_input_vectorized)

        # Display the result
        st.write("Sentiment:", sentiment[0])

        # Display accuracy of the SVM model (optional)
        st.write(f"Model Accuracy: {accuracy_svm:.2f}")
