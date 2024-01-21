import streamlit as st
import pandas as pd
import re
import math
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from numpy import mean, zeros
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
nltk.download('punkt')
import gdown
from transformers import pipeline
import keras
from keras.models import load_model
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Download NLTK stopwords data (if not already downloaded)
nltk.download('stopwords')
from pathlib import Path

# Get the directory where the script is located
script_directory = Path(__file__).parent

# Function to load data from CSV files
def load_csv(file_name):
    # Construct the full path for the CSV file
    file_path = script_directory / file_name
    # Read the CSV file using the full path
    return pd.read_csv(file_path, sep=',')
# Define the words to remove
custom_stopwords = set(stopwords.words('english'))
custom_stopwords.discard('not')
custom_stopwords.discard('no')
custom_stopwords.discard('but')
custom_stopwords.discard("won't")

def preprocess(text):
    tokens = word_tokenize(text)
    table = str.maketrans('', '', punctuation)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]

    # Remove custom stopwords
    tokens = [word for word in tokens if word not in custom_stopwords]

    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)


script_directory = Path(__file__).parent
sentiment_analysis_path = script_directory / 'sentiment_analysis_model.pkl'

# Open the file and load the model
with open(sentiment_analysis_path, 'rb') as file:
    sentiment_model = pickle.load(file)

script_directory = Path(__file__).parent
# Construct the full path for the word2vec model file
rating_model_path = script_directory / 'rating_model.h5'
    
# Load the model from the file using the full path
rating_model = load_model(str(rating_model_path))

script_directory = Path(__file__).parent
tfidf_vectorizer_path = script_directory / 'tfidf_vectorizer_rating.pkl'

# Open the file and load the model
with open(tfidf_vectorizer_path, 'rb') as file:
    tfidf_vectorizer_rating = pickle.load(file)

encoder_path = script_directory / 'encoder.pkl'

# Open the file and load the model
with open(encoder_path, 'rb') as file:
    encoder = pickle.load(file)

df = load_csv('preprocess_df.csv')

# Remove rows with NaN values in the 'Processed_Review' column
df = df.dropna(subset=['Processed_Review'])

# Fit the TF-IDF vectorizer
tfidf_vectorizer.fit_transform(df['Processed_Review']).toarray()

# Set the page layout to wide
st.set_page_config(layout="wide")



def local_css():
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

        /* This CSS hides the Streamlit header and footer */
        .css-1d391kg {display:none;}
        .css-1e5imcs {display:none;}
        /* Apply the Montserrat font family to the entire app */
        html, body, [class^="st-"] {
            font-family: 'Montserrat', sans-serif;
        }

         /* Custom styles for the title */
        h1 {
            font-size: 1.75em; /* Smaller font size for the title */
        }

        /* Style adjustments */
        .stButton>button {
            color: #333;
            background-color: #9CDABC; /* Light green pastel color */
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            font-size: 16px;
        }

        .stSelectbox {
            font-size: 16px;
        }

        .review-container {
            padding: 15px;
            margin-bottom: 15px; /* Increase space between reviews */
            background-color: #f8f9fa; /* Light background for the review container */
            border-radius: 5px; /* Rounded corners for the container */
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Subtle shadow for separation */
            text-align: center; /* Center the text */
        }

        review-text {
            font-size: 18px; /* Size of the review text */
            color: #333; /* Color of the review text */
            margin: 0; /* Reset margin */
            padding-bottom: 5px; /* Space at the bottom of the text */
        }

        .star-rating {
            color: gold;
            font-size: 24px; /* Size of stars */
            display: block; /* Display stars on their own line */
            margin: 0 auto; /* Center the stars with no additional space */
        }

        .rating-number {
            font-size: 32px; /* Larger font size for the rating number */
            color: #333;
            display: block; /* Display rating number on its own line */
            margin: 0 auto; /* Center the rating number with no additional space */
            line-height: 0.5; /* Default line height to ensure it's not adding extra space */
        }

        .reviews-number {
            font-size: 16px; /* Smaller font size for number of reviews */
            color: #333;
            display: block; /* Display number of reviews on its own line */
            margin: 5px auto; /* Center the number of reviews and reduce space */
            line-height: 0.5; /* Default line height to ensure it's not adding extra space */
        }


        /* Add custom styles for the "All Reviews" title */
        .all-reviews-title {
            text-align: center;
            font-size: 24px;
            margin-top: 30px;
            margin-bottom: 20px;
        }

        /* Separate style for the horizontal line for clarity */
        .review-separator {
            border-top: 1px solid #e1e4e8;
            margin: 20px 0; /* Spacing above and below the separator */
        }

        

    </style>
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Run the local CSS function to apply styles
local_css()

def load_data():
    # Replace with your DataFrame path
    data = load_csv("lda_results.csv")
    data['Restaurant'] = data['Restaurant'].str.title()  # Convert restaurant names to title case
    # Load the summary data from another file (e.g., CSV) and create a DataFrame
    summary_data = load_csv("summary_reviews_restaurants.csv")  # Replace with the correct file path
    summary_data['Restaurant'] = summary_data['Restaurant'].str.title()  # Convert restaurant names to title case

    # Merge the summary DataFrame with the main DataFrame on the 'Restaurant' column
    data = data.merge(summary_data[['Restaurant', 'Summary']], on='Restaurant', how='left')
    return data

data = load_data()

# Drop specified restaurants from the dataset
restaurants_to_drop = [
    'Www', 'Superchef', 'Qr-Restaurant', 'Ambiance-Restaurant', 'Daiyarestaurant',
    'Jyl-Software', 'Poulaillon', 'Cityvox', 'Lentourloupe', 'Eevad',
    'Limportant-Paris', 'Restopolitan', 'Contactless-Menu', 'Findfork',
    'Cottageinn', 'Zitti', 'Baianopizzeria', 'Bramlevbakker',
    'Pubdogcolorado', 'Bostoncoffeehouse', 'Menupages'
]

data = data[~data['Restaurant'].isin(restaurants_to_drop)]
# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering",  model="distilbert-base-cased-distilled-squad")

def get_concatenated_reviews(data):
    concatenated_reviews = {}
    for restaurant in data['Restaurant'].unique():
        concatenated_reviews[restaurant] = ' '.join(data[data['Restaurant'] == restaurant]['Review'].tolist())
    return concatenated_reviews

# Usage in your main app
concatenated_reviews = get_concatenated_reviews(data)

from PIL import Image

# Get the directory where the script is located
script_directory = Path(__file__).parent

# Construct the full path for the image file
image_file_path = script_directory / 'image_restaurant.jpg'

# Load the image into a PIL Image object
image = Image.open(image_file_path)

# Now display the image in Streamlit using the PIL Image object
st.image(image, use_column_width=True)

# Sidebar with terms 
selected_term = st.sidebar.radio("Choose a feature", ["Summary & Explanation","Prediction", "Information Retrieval", "QA"])

def display_review_with_stars(rating, num_reviews=None):
    # Round the rating to two decimal places
    rating_float = float(rating)
    rating_rounded = round(rating, 2)
    full_stars = math.floor(rating)
    half_star = '★' if rating % 1 >= 0.5 else ''
    empty_stars = 5 - full_stars - (1 if half_star else 0)
    
    # Generate the HTML for star ratings
    stars_html = '★' * full_stars + half_star + '☆' * empty_stars
    # Define the desired text color using inline CSS
    text_color = '#333'
    
    # Create the rating HTML with the specified text color
    rating_html = f'<div class="rating-number" style="color: {text_color};">{rating_rounded}</div>'
    
    # Combine the HTML for the numeric rating and stars
    combined_html = f'<div style="text-align: center;">{rating_html}<div class="star-rating">{stars_html}</div>'
    
    # Add number of reviews if provided
    if num_reviews is not None:
        reviews_html = f'<div class="reviews-number">({num_reviews} reviews)</div>'
        combined_html += reviews_html
    
    combined_html += '</div>'
    
    return combined_html

# Highlight function
def highlight_text(text, terms):
    highlighted_text = text
    for term in terms:
        highlighted_text = re.sub(f'({term})', r'<span style="background-color:#FFFF00;">\1</span>', highlighted_text, flags=re.IGNORECASE)
    return highlighted_text



# Function for "Summary & Explanation"
def summary_and_explanation():
    # Streamlit App
    st.title("Summary & Explanation")
    # Select Restaurant
    selected_restaurant = st.selectbox("Select a Restaurant", data['Restaurant'].unique())
    # Display Restaurant Data
    restaurant_data = data[data['Restaurant'] == selected_restaurant]
    
    if restaurant_data.empty:
        st.write("No data available for the selected restaurant.")
        return  # Exit the function if there's no data
    
    # Add more space after the select box
    st.write('\n\n')
    # Use columns to create a two-column layout
    col1, col2 = st.columns(2)
    with col1:
        # Display the star ratings in the first column
        avg_rating = restaurant_data['Rating'].mean()
        num_reviews = len(restaurant_data)  # Replace with actual number of reviews
        st.markdown(display_review_with_stars(avg_rating, num_reviews), unsafe_allow_html=True)
    
    with col2:
        # Display the summary in the second column with larger, italic font
        summary = restaurant_data.iloc[0]['Summary']
        if not pd.isna(summary):
            # Center the "Summary of all reviews" title and display the summary text
            st.markdown("<h5 style='text-align: center;'>Summary of all reviews</h2>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 16px; font-style: italic;'>{summary}</div>", unsafe_allow_html=True)

    
    # Add more space after the select box
    st.write('\n')
    # Display Topics as Buttons
    st.write("Read reviews that mention ")
    topics = restaurant_data.iloc[0]['Topics'].split('; ')
    if 'selected_topic' not in st.session_state:
        st.session_state.selected_topic = None

    # Create columns for the buttons
    columns = st.columns(4)

    selected_topics = set()  # To keep track of selected topics
    for i, topic in enumerate(topics, start=1):
        if topic not in selected_topics:  # Check if the topic hasn't been selected before
            reviews_with_topic = restaurant_data['Review'].str.contains('|'.join(topic.split()), case=False, na=False, regex=True).sum() >= 2
            if reviews_with_topic.any():
                selected_topics.add(topic)
                if columns[i % 4].button(topic, key=f'topic_{i}'):
                    st.session_state.selected_topic = topic

    # Display Reviews based on selected topic
    if st.session_state.selected_topic:
        st.write(f"Reviews mentioning : {st.session_state.selected_topic}")
        terms = st.session_state.selected_topic.split()

        # Custom function to check if a review contains at least two words from the selected topic
        def contains_topic_words(review):
            return sum(term in review for term in terms) >= 2

        filtered_reviews = restaurant_data[restaurant_data['Review'].apply(contains_topic_words)]

        for _, row in filtered_reviews.iterrows():
            review_html = display_review_with_stars(row['Rating'])
            st.markdown(review_html, unsafe_allow_html=True)
            st.markdown(f'<p style="text-align: center;">{highlight_text(row["Review"], terms)}</p><hr>', unsafe_allow_html=True)

      
            
    else:
        st.write("All Reviews :")
        for _, row in restaurant_data.iterrows():
            review_html = display_review_with_stars(row['Rating'])
            st.markdown(review_html, unsafe_allow_html=True)
            st.markdown(f'<p style="text-align: center;">{row["Review"]}</p><hr>', unsafe_allow_html=True)

def stars_html_prediction(rating):
    # Determine the number of full stars
    full_stars = int(rating)
    # Determine if there is a half star
    half_star = '★' if rating % 1 >= 0.5 else ''
    # Determine the number of empty stars
    empty_stars = '☆' * (5 - full_stars - (1 if half_star else 0))
    # Combine full stars, half star, and empty stars
    return '★' * full_stars + half_star + empty_stars

def Prediction():
    # Streamlit App
    st.title("Prediction using Sentiment Analysis")
    # Select Restaurant
    # Input text box for user reviews
    user_input = st.text_area("Enter a review :")

        # Add a "Submit" button
    if st.button("Analyze"):
        if user_input:
          
            processed_review = preprocess(user_input)
            vectorized_review = tfidf_vectorizer.transform([processed_review]).toarray()
            vectorized_review_rating = tfidf_vectorizer_rating.transform([user_input]).toarray()
            # Predict the sentiment
            sentiment = sentiment_model.predict(vectorized_review)[0]

            
            sentiment_label = 'Positive' if sentiment == 1 else 'Negative'

            # Printing the results
            st.write("Sentiment :",sentiment_label)

 
            # Predict the rating
            rating_prediction = rating_model.predict(vectorized_review_rating)
            predicted_rating = encoder.inverse_transform(rating_prediction)
            predicted_rating_value = int(predicted_rating[0])
            sentiment_label = 'Positive' if sentiment == 1 else 'Negative'
            
            # Printing the results with stars
            st.write("Sentiment:", sentiment_label)
            st.write(f"Predicted Rating : {stars_html(predicted_rating_value)}")

def semantic_search(query, documents, top_n=10):
    # Tokenize documents
    # Get the directory where the script is located
    script_directory = Path(__file__).parent
    
    # Construct the full path for the word2vec model file
    word2vec_model_path = script_directory / 'word2vec.model'
    
    # Load the model from the file using the full path
    model = Word2Vec.load(str(word2vec_model_path))
    tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
    flattened_docs = [word for doc in tokenized_documents for word in doc if word in model.wv]

    # Calculate TF-IDF scores
    tfidf = TfidfVectorizer(vocabulary=model.wv.key_to_index)
    tfidf.fit([" ".join(flattened_docs)])
    tfidf_scores = {word: tfidf.idf_[i] for word, i in tfidf.vocabulary_.items()}

    # Process query
    query_tokens = word_tokenize(query.lower())
    query_vectors = [model.wv[word] * tfidf_scores.get(word, 1) for word in query_tokens if word in model.wv]

    if not query_vectors:
        return []

    query_vector = mean(query_vectors, axis=0)

    scores = []
    for doc, doc_tokens in zip(documents, tokenized_documents):
        doc_vectors = [model.wv[word] * tfidf_scores.get(word, 1) for word in doc_tokens if word in model.wv]

        if doc_vectors:
            doc_vector = mean(doc_vectors, axis=0)
            similarity = 1 - cosine(query_vector, doc_vector)  # cosine returns the distance, so we subtract from 1
            scores.append((doc, similarity))

    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

def InformationRetrieval():
    # Streamlit App
    st.title("Information Retrieval")
    
    # Radio button to choose a specific restaurant or all restaurants
    search_option = st.radio("Choose Search Option", ["Specific Restaurant", "All Restaurants"])
    
    if search_option == "Specific Restaurant":
        # Select Restaurant
        selected_restaurant = st.selectbox("Select a Restaurant", data['Restaurant'].unique())
        # Display Restaurant Data
        restaurant_data = data[data['Restaurant'] == selected_restaurant]
    
        if restaurant_data.empty:
            st.write("No data available for the selected restaurant.")
            return  # Exit the function if there's no data
    else:
        # Use all restaurants for the search
        restaurant_data = data
    
    # Add more space after the select box
    st.write('\n\n')

    user_input = st.text_input("Enter specific terms you want to look for in all reviews :")

    # Button to trigger information retrieval
    if st.button("Search"):
        if user_input:
            # Get all the reviews for the selected restaurant(s)
            reviews = restaurant_data['Review'].tolist()

            # Perform semantic search
            search_results = semantic_search(user_input, reviews)

            # Display the search results
            st.markdown("<h2 style='font-size: 18px;'>Search Results :</h2>", unsafe_allow_html=True)
            for i, (review, similarity) in enumerate(search_results, start=1):
                if search_option == "All Restaurants":
                    # If searching for all restaurants, display the restaurant name for each review
                    st.markdown(f"Restaurant : {restaurant_data.iloc[i-1]['Restaurant']}")
                st.markdown(f"Review {i} (Similarity Score : {similarity:.2f}):")
                review_html = display_review_with_stars(restaurant_data.iloc[i-1]['Rating'])
                st.markdown(review_html, unsafe_allow_html=True)
                st.markdown(f'<p style="text-align: center;">{highlight_text(review, user_input.split())}</p><hr>', unsafe_allow_html=True)


# Function to display conversation
def display_conversation(conversation):
    for exchange in conversation:
        st.text_area("You", value=exchange['question'], height=40, disabled=True, key=f"q{exchange['id']}")
        st.text_area("Assistant", value=exchange['answer'], height=60, disabled=True, key=f"a{exchange['id']}")

def clear_input():
    # Clear the input box by setting the value to an empty string
    st.session_state.question = ""

def QASystem():
    st.title("Chat with Our Restaurant Assistant")

    # Chat bubble styling with labels on opposite sides
    chat_bubble_css = """
    <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            width: 100%;
            margin-bottom: 0.5rem;
        }
        .chat-bubble {
            padding: 10px 15px;
            border-radius: 18px;
            margin: 5px 10px;
            display: inline-block;
            max-width: 70%;
            word-wrap: break-word;
        }
        .user-bubble {
            background-color: #e6e6ea;
            align-self: flex-end;
        }
        .assistant-bubble {
            background-color: #0084ff;
            color: white;
            align-self: flex-start;
        }
        .label {
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 2px;
            color: #7c7c7c;
        }
        .user-label {
            align-self: flex-end;
            text-align: right;
            padding-right: 10px;
        }
        .assistant-label {
            align-self: flex-start;
            text-align: left;
            padding-left: 10px;
        }
    </style>
    """

    # Apply the chat bubble styling
    st.markdown(chat_bubble_css, unsafe_allow_html=True)

    # Initialize conversation state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Display conversation history with chat bubbles
    for exchange in st.session_state.conversation:
        st.markdown(
            f'<div class="chat-container">'
            f'    <div class="label user-label">You</div>'
            f'    <div class="chat-bubble user-bubble">{exchange["question"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="chat-container">'
            f'    <div class="label assistant-label">Assistant</div>'
            f'    <div class="chat-bubble assistant-bubble">{exchange["answer"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    

    # Static placeholder for the input field
    placeholder_text = (
            "Ask a question about any restaurant... (e.g., Give me the name of a good sushi restaurant, "
            "How is the food at Ayakosushi ?, Give me the name of a good French restaurant, "
            "What can I eat at Les Antiquaires ?...)"
        )


    # Input for new question with static placeholder
    question = st.text_input(placeholder_text, key="question_input")

    # Handling the question input
    if st.button("Send"):
        if question:
            # Create a dictionary to map reviews back to their restaurants
            review_to_restaurant = {row['Review']: row['Restaurant'] for _, row in data.iterrows()}

            # Get all reviews
            reviews = data['Review'].tolist()

            # Perform semantic search to find relevant reviews
            relevant_reviews = semantic_search(question, reviews, top_n=10)

            # Concatenate relevant reviews with their corresponding restaurant names
            relevant_reviews_text = ' '.join([f"{review_to_restaurant[review]}: {review}" for review, _ in relevant_reviews])

            # Get the answer from the QA pipeline
            answer = qa_pipeline({'question': question, 'context': relevant_reviews_text})

            # Add the exchange to the conversation history
            st.session_state.conversation.append({
                'question': question,
                'answer': answer['answer']  # The answer obtained from the QA pipeline
            })

            # Clear the input box after sending the message
            clear_input()

            # Rerun the app to update the conversation display
            st.experimental_rerun()
        else:
            st.write("Please ask a question.")

# Call the corresponding function based on the selected term
if selected_term == "Summary & Explanation":
    summary_and_explanation()
elif selected_term == "Prediction":
    Prediction()
elif selected_term == "Information Retrieval":
    InformationRetrieval()
elif selected_term == "QA":
    QASystem()


