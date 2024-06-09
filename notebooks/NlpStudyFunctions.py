import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize,sent_tokenize
from nltk.tokenize import word_tokenize
import nltk

from collections import Counter


# NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def CleanDataframeDF(df, text_column):
    df[text_column] = df[text_column].apply(SimplePreprocessTextDF)
    
    return df


def SimplePreprocessTextDF(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Replace words starting with 'http' or 'https' with 'link'
    words = ['link' if word.startswith('http') else word for word in words]
    
    # Remove specific unwanted terms and numbers
    unwanted_terms = {'û_', 'co', 'amp'}
    words = [word for word in words if word not in unwanted_terms and not word.isdigit()]
    
    # Replace words starting with '@' with '@user'
    words = ['@user' if word.startswith('@') else word for word in words]
    
    # Join the cleaned words back into a string
    cleaned_text = ' '.join(words)
    
    return cleaned_text

def SimplePreprocessText(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Replace words starting with 'http' or 'https' with 'link'
    words = ['link' if word.startswith('http') else word for word in words]
    
    # Remove specific unwanted terms and numbers
    unwanted_terms = {'û_', 'co', 'amp'}
    words = [word for word in words if word not in unwanted_terms and not word.isdigit()]
    
    # Replace words starting with '@' with '@user'
    words = ['@user' if word.startswith('@') else word for word in words]
    
    return words

def CleanDataframe(df, text_column):
    df[text_column] = df[text_column].apply(SimplePreprocessText)
    
    return df


def GetTopWords(df, top_number):
    # Initialize an empty list to store all words
    all_words = []

    # Iterate over each text in the 'text' column of the DataFrame
    for text in df['text']:
        # Clean and tokenize the text
        cleaned_text = SimplePreprocessTextDF(text)
        # Add the words to the list
        all_words.extend(cleaned_text.split())

    # Count word frequency
    word_freq = Counter(all_words)
    most_common_words = word_freq.most_common(top_number)
    most_common_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
    print(most_common_df)

    return most_common_df
