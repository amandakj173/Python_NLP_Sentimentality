# DATA WRANGLING

# Import Packages
import pandas as pd
import numpy as np

# Set Parameters
pd.set_option("display.precision", 3)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

# Import Data
ds = pd.read_csv("x_tweets.csv", encoding = "latin1")

# Basic Data Exploration
print(ds)
ds.info()