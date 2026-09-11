# DATA WRANGLING

# Import Packages
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import revised_stopwords
from revised_stopwords import get_revised_stopwords

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

# Text Sampling
ds = ds.sample(n = 1000, random_state = 42).reset_index(drop=True)

# Text Cleaning
def clean_text(text):
	text = text.lower()  # Convert all text to lowercase
	text = re.sub(r"http\s+|www\s+|https\s+", '', text, flags=re.MULTILINE)  # Remove URL
	text = re.sub(r'@\w+', '', text)  # Remove @mentions
	text = re.sub(r'&amp;', '&', text)  # Fix HTML entities
	text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters/spaces
	return text
# Apply to dataset
ds["clean_text"] = ds["text"].apply(clean_text)
print(ds)


# NORMALISATION
def normalise_text(clean_text):
	tokens = word_tokenize(clean_text)  # Tokenise
	filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stopwords
	revised_filtered_tokens = [token for token in tokens if token not in get_revised_stopwords()]  # Get & apply revised stopwords for context
	lemmatizer = WordNetLemmatizer()  # Lemmatise
	lemmatized_tokens = [lemmatizer.lemmatize(token) for token in revised_filtered_tokens]
	processed_text = " ".join(lemmatized_tokens)  # Join tokens back into string
	return processed_text

ds["normalise_text"] = ds["clean_text"].apply(normalise_text)
print(ds)


# SENTIMENT ANALYSIS

analyser = SentimentIntensityAnalyzer()  # Initialise analyser

def get_sentiment(normalise_text):  # Create sentiment function
	scores = analyser.polarity_scores(normalise_text)  # Define score
	get_sentiment = scores["compound"]
	return get_sentiment  # Return new column

ds["get_sentiment"] = ds["normalise_text"].apply(get_sentiment)  # Apply get_sentiment function
print(ds)