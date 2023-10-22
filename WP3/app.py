import streamlit as st
import os
import sys
sys.path.insert(1, "CNN")
import loadCNN as cnn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_option('deprecation.showfileUploaderEncoding', False)

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

@st.cache(persist=True)
def get_tfidf_vectorizer(articles):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    return tfidf_vectorizer.fit(articles)

# Retrieve the TF-IDF vectorizer, cached for efficiency
tfidf_vectorizer = get_tfidf_vectorizer(articles)

@st.cache(persist=True)
def retrieve_top_documents(query_summary, k=10):
    query_vector = tfidf_vectorizer.transform([query_summary])
    similarity_scores = linear_kernel(query_vector, tfidf_vectorizer.transform(articles))
    document_indices = similarity_scores[0].argsort()[:-k-1:-1]
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

i = 10  # Customize the index
st.header(f"Example: Retrieve the most similar document for abstract {i} ")
st.write("Query abstract:")
st.write(abstracts[i])

best_similarity = np.max(scores[i])
best_location = np.argmax(scores[i])
st.write(f"📊 Best similarity with abstract {best_location} : {best_similarity} ")
st.write("Matching Document:")
st.write(articles[best_location])
