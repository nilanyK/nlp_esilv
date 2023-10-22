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

@st.cache
def get_tfidf_vectorizer(articles):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    return tfidf_vectorizer.fit_transform(articles)

# Retrieve the TF-IDF vectorizer, cached for efficiency
tfidf_articles = get_tfidf_vectorizer(articles)

@st.cache
def retrieve_top_documents(query_summary, tfidf_articles, articles, k=10):
    query_vector = tfidf_vectorizer.transform([query_summary])
    similarity_scores = linear_kernel(query_vector, tfidf_articles)
    document_indices = similarity_scores.argsort()[0][::-1][:k]
    top_documents = [articles[i] for i in document_indices]
    return top_documents

st.title("📚 CNN Document Retrieval 📚")
st.subheader("By Nilany Karunathasan 👨‍💻")

query_summary = st.text_area("✏️ Enter your request summary:")

if st.button("🔍 Retrieve Documents"):
    if not query_summary:
        st.warning("Please enter a request summary.")
    else:
        top_k_documents = retrieve_top_documents(query_summary, tfidf_articles, articles, k=10)
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
