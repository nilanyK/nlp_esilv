import os
import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to load movie reviews from text files in a folder
def load_txt_files(folder_path):
    """Loads all of the txt files in a folder into a list of strings.

    Args:
      folder_path: The path to the folder containing the txt files.

    Returns:
      A list of strings, where each string contains a movie review.
    """
    reviews = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            review = f.read()
            reviews.append(review)
    return reviews

# Load movie reviews from text files
neg_reviews = load_txt_files("neg")
pos_reviews = load_txt_files("pos")

# Combine the positive and negative reviews into a single DataFrame
reviews_df = pd.DataFrame(data={'review': neg_reviews + pos_reviews, 'sentiment': ['negative'] * len(neg_reviews) + ['positive'] * len(pos_reviews)})

# Create a Bag of Words model
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(reviews_df['review'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, reviews_df['sentiment'], test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
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
