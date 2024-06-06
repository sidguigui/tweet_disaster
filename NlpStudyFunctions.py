import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')


def SimplePreprocessText(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Remove specific unwanted terms and numbers
    unwanted_terms = {'http', 'https', 'û_', 'co', 'amp'}
    words = [word for word in words if word not in unwanted_terms and not word.isdigit()]
    
    return words

# Example usage:
text = """
This is a test tweet! Visit http://example.com for more info. #exciting @user û_ https 2
"""
processed_text = SimplePreprocessText(text)
print(processed_text)



def GetTopWords(df, top_number):
    # Apply preprocessing to the column
    df['processed'] = df['text'].apply(SimplePreprocessText)

    # Flatten the list of lists into a single list of words
    all_words = [word for sublist in df['processed'] for word in sublist]

    # Count word frequencies
    word_freq = Counter(all_words)

    # Get the most common words
    most_common_words = word_freq.most_common(top_number)

    # Convert to DataFrame for easier analysis
    most_common_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

    # Print the most common words
    print(most_common_df)
    
    return most_common_df
