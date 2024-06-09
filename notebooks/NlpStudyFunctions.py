import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize,sent_tokenize
from nltk.tokenize import word_tokenize
import nltk

# NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


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

    #string with all the words
    all_words = [word for sublist in df['text'] for word in sublist]

    #count word frquency
    word_freq = Counter(all_words)
    most_common_words = word_freq.most_common(top_number)
    most_common_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
    print(most_common_df)

    return most_common_df