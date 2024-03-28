ESILV - Machine Learning for NLP - Project 2024 <br>
[KARUNATHASAN Nilany](https://www.linkedin.com/in/nilany-karunathasan-7b49691ba/) <br>
[SAMBATH Sïndoumady](https://www.linkedin.com/in/s%C3%AFndoumady-sambath-a7519a209/) <br>
<br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1wefYGTtTMoAVfRXJNx4I3EHMjxUJvqMA/view?usp=sharing) <br>


<br>
<h1 align="center">Data Exploration and NLP Modeling</h1>
<br>
<br>
<div style="text-align:center;">
    <span style="display:inline-block; margin-right: 20px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/5/5c/Trustpilot_logo.png" alt="Trustpilot Logo" width="100"/>
    </span>
    <span style="display:inline-block; margin-right: 20px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/ad/Yelp_Logo.svg" alt="Yelp Logo" width="100"/>
    </span>
</div>



### Overview
This repository encompasses all files pertinent to the second project in the "Machine Learning for NLP" course. The project aims to utilize NLP techniques to extract valuable insights from customer reviews sourced from prominent review platforms. <br>
To establish a comprehensive database, we chose to gather information from two main sources: Trustpilot and Yelp. Trustpilot, a customer review platform, and Yelp, a restaurant review site, offer an extensive dataset of restaurant reviews.

## Table of Contents
- [Files](#files)
- [Installation](#installation)
- [Features](#features)
- [Data Collection and Preparation](#data-collection-and-preparation)
- [Contributors](#contributors)
  
## Files

The project repository contains the following main files :

- **NLP_PROJECT2_KARUNATHASAN_SAMBATH.ipynb** : A Jupyter Notebook containing all the code generated for the project. (also accessible via the colab link)

- **app.py** : The Python script for the Streamlit app, which allows users to explore data insights, view sentiment analysis results, and interact with the QA system.


## Installation
Our interactive application allows users to explore data insights, view sentiment analysis results, and interact with the QA system. <br>
<br>
**Streamlit App** <br>
     **• Public Access** <br>
       You can acces the app through this link : https://project2nlp.streamlit.app/ <br>
     **• Local Access** <br>
     - Clone this GitHub repository to your local machine using the following command: 
       ```
       git clone https://github.com/nilanyK/nlp_esilv.git
       ```
       <br>
     - Change to the project directory: 
       ```
       cd PROJECT2
       ```
       <br>
     - Run the Streamlit app by executing `app.py` : 
       ```
       streamlit run app.py
       ```
       <br>
     - The output will display the link on which the server is running.  Click on it if it doesn't directly open the link.
   <br>
   <br>

## Features
- **Summary** :
  - Diplay concise summaries of restaurant reviews, providing quick insights into the overall customer feedback.

- **Topic Modeling (Read reviews that mention...)** :
  - Extract common themes and topics from review data, offering a deeper understanding of what aspects customers frequently discuss.

- **Sentiment Analysis & Explanation** :
  - Analyze the sentiment of restaurant reviews to understand customer opinions. This feature helps identify general sentiment trends such as positivity, negativity, or neutrality within the reviews.

- **Rating Prediction** :
  - Utilize deep learning models to predict the star ratings of reviews based on their content, aiding in the understanding of customer satisfaction levels.

- **Information Retrieval** :
  - Search reviews by specific search terms, enabling users to find detailed opinions and experiences related to particular aspects of a restaurant.

- **QA System** :
  - A chatbot for answering user queries based on the review dataset. This interactive tool uses NLP to understand and respond to user inquiries, providing personalized recommendations and insights.


## Data Collection and Preparation
Data was scraped from Trustpilot and Yelp. Rigorous data cleaning and preparation were performed to ensure the quality and consistency of the dataset.

### Contributors
[KARUNATHASAN Nilany](https://www.linkedin.com/in/nilany-karunathasan-7b49691ba/) <br>
[SAMBATH Sïndoumady](https://www.linkedin.com/in/s%C3%AFndoumady-sambath-a7519a209/) <br>
