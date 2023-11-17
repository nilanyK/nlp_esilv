import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import loadNFCorpus as nfc  

# Suppress the deprecation message
st.set_option('deprecation.showfileUploaderEncoding', False)

# Load NFCorpus data using the imported function
dicDoc, dicReq, dicReqDoc = nfc.loadNFCorpus()

# Preprocess text
@st.cache(persist=True)
def text_preprocessing(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(filtered_tokens)

# Customized TF-IDF retrieval function
@st.cache(persist=True)
def run_custom_tfidf(dicDoc, user_query, nb_docs): 
    dicReqDocToKeep = defaultdict(dict)

    docsToKeep = []

    # Pre-processing and vectorization
    corpus = [text_preprocessing(doc) for doc in dicDoc.values()]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Calculate TF-IDF scores for the user query and documents
    user_query_vector = tfidf_vectorizer.transform([user_query])

   for docId, doc_text in dicDoc.items():
      if i >= (nb_docs):
            break
      doc_vector = tfidf_vectorizer.transform([text_preprocessing(doc_text)])
      cos_sim = (user_query_vector * doc_vector.T).toarray()[0][0]
      docsToKeep.append((docId, cos_sim))
      i=i+1
       
    # Sort the documents by their scores in descending order
    docsToKeep.sort(key=lambda x: x[1], reverse=True)

    # Take the top nb_docs documents (as specified by the user)
    top_documents = docsToKeep[:nb_docs]

    return top_documents

# Calculate NDCG@5
@st.cache(persist=True)
def calculate_ndcg(doc_scores, true_labels):
    return ndcg_score([true_labels], [doc_scores], k=5)

st.title("🔍 Custom Information Retrieval System on NFCorpus 🔍")
st.sidebar.header("📌 Project Recap")
st.sidebar.markdown("""
The goal was to develop an original information retrieval system on NFCorpus. We were allowed to use any kind of pre-treatment and manipulate the vocabulary of the documents. 

**BM25 was our baseline** and we needed to find a way to improve the result. The metric used is the NDCG@5, which evaluates the top 5 results returned by the model.

👨‍💻 Authors : [Nilany KARUNATHASAN](https://www.linkedin.com/in/nilany-karunathasan-7b49691ba?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app) and [Sïndoumady SAMBATH](https://www.linkedin.com/in/s%C3%AFndoumady-sambath-a7519a209?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)
""")

# Create a form to enter the user query and trigger the button with Enter key
with st.form(key='query_form'):
    query_input = st.text_area("📝 Enter your query", height=100)
    nb_docs = st.slider("📄 Number of Documents to Consider", min_value=0, max_value=500, value=150)  # Add a slider for nb_docs
    submitted = st.form_submit_button("🔍 Retrieve Top Documents")

    if submitted or query_input:
        top_documents = run_custom_tfidf(dicDoc, query_input, nb_docs)

        st.subheader("📜 Top Documents :")

        for rank, (docId, cos_sim) in enumerate(top_documents, start=1):
            st.write(f"🏆 Rank {rank} - Document {docId} :")
            st.write(dicDoc[docId])
