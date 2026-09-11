# DATA WRANGLING

# Import Packages
import pandas as pd
import numpy as np
import random
import spacy
import re

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

# Test/Train Split
random.seed(123)  # Randomisation

# Split train/test sets
obs = len(ds)
test_idx = np.random.choice(
	obs,
	size = round(obs * 0.3),
	replace = False
)

test_ds = ds.iloc[test_idx]
train_ds = ds.drop(ds.index[test_idx])

print(train_ds.info())
print(test_ds.info())

# WORD VECTORS
docs = [nlp(text) for text in train_ds]