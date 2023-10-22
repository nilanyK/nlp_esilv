import streamlit as st
import os
import sys
sys.path.insert(1, "CNN")
import loadCNN as cnn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Suppress the deprecation message
st.set_option('deprecation.showfileUploaderEncoding', False)

# Load CNN dataset
articles, abstracts = cnn.loadCNN()

@st.cache(persist=True)
def get_tfidf_vectorizer(articles):
    # Create a TF-IDF vectorizer with specific settings
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    return tfidf_vectorizer.fit(articles)

# Retrieve the TF-IDF vectorizer, cached for efficiency
tfidf_vectorizer = get_tfidf_vectorizer(articles)

@st.cache(persist=True)
def retrieve_top_documents(query_summary, k=10):
    # Transform the query summary into a TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query_summary])
    
    # Calculate cosine similarity between the query and all articles
    similarity_scores = linear_kernel(query_vector, tfidf_vectorizer.transform(articles))
    
    # Sort document indices by similarity score in descending order
    document_indices = similarity_scores[0].argsort()[:-k-1:-1]
    
    # Retrieve the top-k documents based on their indices
    top_documents = [articles[i] for i in document_indices]
    
    return similarity_scores, top_documents  # Return similarity_scores

st.title("📚 CNN Document Retrieval 📚")
st.subheader("By Nilany Karunathasan 👨‍💻")

query_summary = st.text_area("✏️ Enter your request summary:")

if st.button("🔍 Retrieve Documents"):
    if not query_summary:
        st.warning("Please enter a request summary.")
    else:
        similarity_scores, top_k_documents = retrieve_top_documents(query_summary, k=10)
        st.header("📜 Top Documents 📜")
        for i, document in enumerate(top_k_documents, start=1):
            st.subheader(f"🏆 Rank {i}:")
            st.write(document)
