# DATA WRANGLING

# Import Packages
import pandas as pd
import numpy as np
import random
import spacy

nlp = spacy.load('en_core_web_sm')

# Set Parameters
pd.set_option("display.precision", 3)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

# Import Data
cols = ["Sentiment", "ID", "Date", "Flag", "User", "Text"]
ds = pd.read_csv("x_tweets.csv", encoding = "latin1", header = None, names = cols)

# Basic Data Exploration
print(ds)
print(ds.info())

# PREPROCESSING
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