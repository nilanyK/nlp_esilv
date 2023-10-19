import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

nltk.download('movie_reviews')
# Load the movie reviews dataset
from nltk.corpus import movie_reviews
reviews = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

# Create a DataFrame
df = pd.DataFrame(reviews, columns=['text', 'sentiment'])

# Create a Bag of Words model
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(df['text'].apply(' '.join))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['sentiment'], test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.title("Movie Review Sentiment Analyzer")

# Input text box for user reviews
user_input = st.text_area("Enter a movie review:")

if user_input:
    # Vectorize the user input
    user_input_vectorized = cv.transform([user_input])

    # Predict the sentiment
    sentiment = nb_classifier.predict(user_input_vectorized)

    # Display the result
    st.write("Sentiment:", sentiment[0])

    # Display accuracy of the model (optional)
    st.write(f"Model Accuracy: {accuracy:.2f}")

