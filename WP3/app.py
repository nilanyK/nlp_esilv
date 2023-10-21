import streamlit as st
import os
import sys
sys.path.insert(1, "CNN")
import loadCNN as cnn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load CNN dataset
articles, abstracts = cnn.loadCNN()

# Create a TfidfVectorizer with specific settings
# It will convert text data into TF-IDF feature vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Apply TF-IDF vectorization to the "articles" text data and "abstracts" text data
# This computes TF-IDF values for each term in the articles and produces a TF-IDF matrix
tfidf_articles = tfidf_vectorizer.fit_transform(articles)
# This computes TF-IDF values for each term in the abstracts and produces a TF-IDF matrix
tfidf_abstracts = tfidf_vectorizer.transform(abstracts)

# Calculate similarity scores using linear_kernel
scores = linear_kernel(tfidf_abstracts, tfidf_articles)

# Define a function to retrieve the top-k documents for a given summary
def retrieve_top_documents(query_summary, k=10):
    # Transform the query summary into a TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query_summary])
    
    # Calculate cosine similarity between the query and all articles
    similarity_scores = linear_kernel(query_vector, tfidf_articles)
    
    # Sort document indices by similarity score in descending order and get the top-k indices
    document_indices = similarity_scores.argsort()[0][::-1][:k]
    
    # Retrieve the top-k documents based on their indices
    top_documents = [articles[i] for i in document_indices]
    
    # Return the top-k documents
    return top_documents



st.title("📚 CNN Document Retrieval 📚")
st.subheader("By Nilany Karunathasan 👨‍💻")

query_summary = st.text_area("✏️ Enter your request summary:")

if st.button("🔍 Retrieve Documents"):
    if not query_summary:
        st.warning("Please enter a request summary.")
    else:
        top_k_documents = retrieve_top_documents(query_summary, k=10)
        st.header("📜 Top Documents 📜")
        for i, document in enumerate(top_k_documents, start=1):
            st.subheader(f"🏆 Rank {i}:")
            st.write(document)

i = 10  # Customize the index
st.header(f"Example: Retrieve the most similar document for abstract {i} ")
st.write("Query abstract:")
st.write(abstracts[i])

best_similarity = np.max(scores[i])
best_location = np.argmax(scores[i])
st.write(f"📊 Best similarity with abstract {best_location} : {best_similarity} ")
st.write("Matching Document:")
st.write(articles[best_location])
