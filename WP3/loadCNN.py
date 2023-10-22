import pickle
from pathlib import Path

def loadCNN():
    # Get the directory where the loadCNN.py script is located
    script_directory = Path(__file__).parent

    # Construct the full file paths
    articles_file = script_directory / "CNNArticles"
    abstracts_file = script_directory / "CNNGold"

    # Load articles and abstracts from the files
    with open(articles_file, 'rb') as article_file, open(abstracts_file, 'rb') as abstract_file:
        articles = pickle.load(article_file)
        abstracts = pickle.load(abstract_file)

    # Clean up the articles and abstracts as needed
    articles = [article.replace("”", "").rstrip("\n") for article in articles]
    abstracts = [article.replace("”", "").rstrip("\n") for article in abstracts]

    return articles, abstracts

articles, abstracts = loadCNN()
print("ARTICLE=",articles[0])
print("SUMMARY=",abstracts[0])
