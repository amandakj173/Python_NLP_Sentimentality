# Natural Language Processing - Tweet Sentiment Analysis
This project applied a Natural Language Processing (NLP) analysis to a random sample for 1,000 Tweets from a database containing 1.6 million X (formerly Twitter) Tweets in order to extract and analyse sentiment. The project combined text cleaning, contexual stopword filtering, and lemmatisation with the NLTK VADER Sentiment Intensity Analyser in order to extract sentiment scores from raw social media text

## Description
Social media presents a unique NLP challenge due to high noise, informal language, web artefacts, and colloquialisms. Traditional sentiment tools often misinterpret contexual nuances when standard stopword lists remove critical signal or leave in social medial-specific noise.

The project implements a structured NLP processing and analysis workflow:
1. Data wrangling & sampling: Processed the raw dataset containing 1.6 million utterances and extracted a reproducible sample of 1,000 utterances using fixed random seeding.
2. Text cleaning: Stripped URLs, @ mentions, HTML entities, and non-alphabetic characters, and standardised all text to lowercase.
3. Text normalisation: Tokenised cleaned text, applied a specialised set of revised stopwords (adjusted to account for contextual relevance) and lemmatised remaining tokens using WordNetLemmatizer.
4. Sentiment scoring: Computed continuous compound sentiment scores (between -1.0 and +1.0) across all normalised utterances using NLTKs VADER.

## Interpretation
### VADER
Analysis using VADER's compound polarity metric mapped normalised text strings into intervals. These intervals range from +1.0 to -1.0, wherein +1.0 indicates extreme positive sentiment, 0.0 represents neutral tone, and -1.0 indicates extreme negative sentiment.

## Next Steps
1. To add a benchmark comparison against original dataset lables by mapping and evaluating using a confusion matrix and classification report to identify where VADER analysis succeeds or fails.
2. To train a predictive model on new utterances to learn X-specific vocabulary patterns and predict sentiment in real time.
