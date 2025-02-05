import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    """Cleans and tokenizes text messages."""
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize words
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)

def load_data(file_path):
    """Loads and processes the dataset."""
    df = pd.read_csv(file_path, names=['label', 'message'], sep='\t', header=None)
    df['message'] = df['message'].apply(preprocess_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary
    return df