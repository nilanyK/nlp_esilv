import pickle
from pathlib import Path
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import gensim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

# Load the NFCorpus dataset and relevant data
def loadNFCorpus():
    script_directory = Path(__file__).parent
    filename = script_directory /  "dev.docs"

    dicDoc = {}
    with open(filename) as file:
        lines = file.readlines()
    for line in lines:
        tabLine = line.split('\t')
        key = tabLine[0]
        value = tabLine[1]
        dicDoc[key] = value
    filename =  script_directory / "dev.all.queries"
    dicReq = {}
    with open(filename) as file:
        lines = file.readlines()
    for line in lines:
        tabLine = line.split('\t')
        key = tabLine[0]
        value = tabLine[1]
        dicReq[key] = value
    filename =   script_directory / "dev.2-1-0.qrel"
    dicReqDoc = defaultdict(dict)
    with open(filename) as file:
        lines = file.readlines()
    for line in lines:
        tabLine = line.strip().split('\t')
        req = tabLine[0]
        doc = tabLine[2]
        score = int(tabLine[3])
        dicReqDoc[req][doc] = score

    return dicDoc, dicReq, dicReqDoc