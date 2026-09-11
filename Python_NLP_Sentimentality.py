# DATA WRANGLING

# Import Packages
import pandas as pd
import numpy as np
import random
import spacy
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nlp = spacy.load('en_core_web_sm')

# Set Parameters
pd.set_option("display.precision", 3)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

# Import Data
cols = ["sentiment", "ID", "date", "flag", "user", "text"]
ds = pd.read_csv("x_tweets.csv", encoding = "latin1", header = None, names = cols)

# Basic Data Exploration
print(ds)
print(ds.info())

# PREPROCESSING

# Text Cleaning
def clean_text(text):
	text = text.lower()  # Convert all text to lowercase
	text = re.sub(r"http\s+|www\s+|https\s+", '', text, flags=re.MULTILINE)  # Remove URL
	text = re.sub(r'@\w+', '', text)  # Remove @mentions
	text = re.sub(r'&amp;', '&', text)  # Fix HTML entities
	text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters/spaces
	return text
# Apply to dataset
ds["clean_text"] = ds["text"].apply(clean_txt)
print(ds)


# NORMALISATION
def normalise_text(clean_text):
	tokens = word_tokenize(clean_text)  # Tokenise
	filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stopwords
	lemmatizer = WordNetLemmatizer()  # Lemmatise
	lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
	processed_text = " ".join(lemmatized_tokens)  # Join tokens back into string
	return processed_text

ds["normalise_text"] = ds["clean_text"].apply(normalise_text)
print(ds)
