import streamlit as st
import streamlit as st
import nltk

# Download necessary data
nltk.download("sentiwordnet")
nltk.download("punkt")
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import nltk


nltk.download('wordnet')
nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Function to calculate sentiment scores using SentiWordNet
def get_sentiment_scores(word):
    synsets = list(swn.senti_synsets(word))
    if synsets:
        sentiment_scores = [(synset.pos_score(), synset.neg_score(), synset.obj_score()) for synset in synsets]
        return sentiment_scores
    else:
        return [(0.0, 0.0, 0.0)]  # Return a single neutral score if no sentiment scores are available

# Function to identify adverbs and adjectives and their sentiment scores
def analyze_review(review_text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(review_text)
    tagged_words = pos_tag(words)
    adverbs_and_adjectives = [word for word, pos in tagged_words if (pos == 'RB' or pos == 'JJ') and word.lower() not in stop_words]
    
    adverb_and_adjective_sentiment_scores = []
    for word in adverbs_and_adjectives:
        sentiment_scores = get_sentiment_scores(word)
        adverb_and_adjective_sentiment_scores.append((word, sentiment_scores))
    
    return adverb_and_adjective_sentiment_scores

# Function to get overall sentiment

def get_overall_sentiment(sentiment_scores):
    # Initialize total scores
    total_positive_score = 0.0
    total_negative_score = 0.0
    total_objective_score = 0.0

    # Check if the sentiment scores iterable is empty
    if not sentiment_scores:
        return "Neutral"

    # Iterate over the sentiment scores and sum the positive, negative, and objective scores
    for score in sentiment_scores:
        try:
            positive_score, negative_score, objective_score = score
            total_positive_score += float(positive_score)
            total_negative_score += float(negative_score)
            total_objective_score += float(objective_score)
        except ValueError:
            # Ignore scores that cannot be converted to floats
            pass

    # Check if all scores are greater than or equal to zero
    if total_positive_score >= 0 and total_negative_score >= 0 and total_objective_score >= 0:
        # Determine overall sentiment polarity
        if total_positive_score > total_negative_score and total_positive_score > total_objective_score:
            return "Positive"
        elif total_negative_score > total_positive_score and total_negative_score > total_objective_score:
            return "Negative"
        elif total_objective_score > total_positive_score and total_objective_score > total_negative_score:
            return "Objective"
        else:
            return "Neutral"
    else:
        # Return neutral if any of the scores are less than zero
        return "Neutral"



# Streamlit UI
st.title("Movie Review Sentiment Analysis")

# Input for entering a movie review
user_input = st.text_area("Enter your movie review:")
if st.button("Analyze Sentiment"):
    if user_input:
        st.subheader("Review:")
        st.write(user_input)

        st.subheader("Adverbs and Adjectives and Their Sentiment Scores:")
        adverb_and_adjective_sentiment_scores = analyze_review(user_input)

        if not adverb_and_adjective_sentiment_scores:
            st.write("No adverbs or adjectives found in the review.")
        else:
            for word, scores in adverb_and_adjective_sentiment_scores:
                st.write(f"Word: {word}")
                for i, score in enumerate(scores):
                  st.write(f"Score {i + 1}: {score}")
                st.write("---")

        st.subheader("Overall Sentiment:")
        overall_sentiment = get_overall_sentiment(adverb_and_adjective_sentiment_scores)

        # Display the overall sentiment, even if the review does not contain any adverbs or adjectives
        if overall_sentiment is None:
            st.write("Neutral")
        else:
            st.write(overall_sentiment)

