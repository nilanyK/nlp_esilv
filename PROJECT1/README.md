ESILV - Machine Learning for NLP - Project 2023 <br>
[KARUNATHASAN Nilany](https://www.linkedin.com/in/nilany-karunathasan-7b49691ba/) <br>
[SAMBATH Sïndoumady](https://www.linkedin.com/in/s%C3%AFndoumady-sambath-a7519a209/) <br>

<br>
<h1 align="center">Information Retrieval Challenge Beating BM25</h1>


### Introduction
This repository contains all files related to the first project of the course Machine Learning for NLP. This project focuses on enhancing information retrieval on the NFCorpus dataset, a collection of medical abstracts from PubMed publications. The primary goal was to beat the performance of the baseline BM25 model. We also developped a custom TF-IDF based retrieval system and a Streamlit app to provide an interactive interface for users to retrieve relevant documents.
 

### Files

The project repository contains the following files :

- **NLP_PROJECT1_KARUNATHASAN_SAMBATH.ipynb** : A Jupyter Notebook containing the code for the project. It includes the implementation of the custom TF-IDF retrieval system and the analysis of the results.

- **app.py** : The Python script for the Streamlit app, which allows users to interact with the custom retrieval system.

- **dev.2-1-0.qrel** : Data file containing relevance scores for queries and documents.

- **dev.all.queries** : Data file with queries related to PubMed articles.

- **dev.docs** : Data file containing abstracts of medical publications from PubMed.

- **loadNFCorpus.py** : Python script for loading NFCorpus data and performing text preprocessing.

- **requirements.txt** : A list of required Python packages and dependencies for the project.


### How to Use

1. **Custom TF-IDF Retrieval**
   - Open the Jupyter Notebook (`NLP_PROJECT1_KARUNATHASAN_SAMBATH.ipynb`) to explore the code.
   - Run the code to analyze the performance of the custom TF-IDF retrieval system on the NFCorpus dataset.

2. **Streamlit App** <br>
     **• Public Access** <br>
       You can acces the app through this link : <br>
     **• Local Access** <br>
     - Clone this GitHub repository to your local machine using the following command:
       ```bash
       git clone https://github.com/nilanyK/nlp_esilv.git
       ```
     - Change to the project directory:
       ```bash
       cd PROJECT1
       ```
     - Run the Streamlit app by executing `app.py` :
       ```python
       streamlit run app.py
       ```
     - The output will display the link on which the server is running.  Click on it if it doesn't directly open the link.
     - Enter your query and choose the number of documents to consider.
     - Click the "Retrieve Top Documents" button to initiate the retrieval process and view the top documents. <br>
   <br>

### Colab Notebook

You can find the Colab Notebook for this project [here](insert_your_colab_notebook_link_here).
