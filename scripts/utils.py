# fintech_app_analytics/scripts/utils.py
import re
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import os # Add os for path manipulation
import sys # Add sys for path manipulation

# --- IMPORTANT: Ensure project root is on sys.path for internal imports ---
# This makes sure that 'scripts' is recognized as a package within the project structure
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- NLTK Downloads (More Robust) ---
# Define a central NLTK data path within your project's .venv
nltk_data_path = os.path.join(os.environ.get('VIRTUAL_ENV', current_script_dir), 'nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# List of NLTK resources to download
nltk_resources = [
    'vader_lexicon',
    'punkt',
    'stopwords',
    'wordnet',
    'omw-1.4' # Open Multilingual Wordnet, a dependency for WordNet
]

print(f"Ensuring NLTK data is available in {nltk_data_path}...")
for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}') # For corpora
        print(f"  {resource} found.")
    except LookupError:
        print(f"  {resource} not found. Attempting to download...")
        try:
            nltk.download(resource, download_dir=nltk_data_path)
            print(f"  {resource} downloaded successfully.")
        except Exception as e: # Catch a general Exception, not just DownloadError
            print(f"  Failed to download {resource}: {e}")
            print("  Please try running 'python -m nltk.downloader <resource_name>' in your terminal.")

# --- SpaCy Model Loading ---
# Ensure spacy model is loaded only once
try:
    nlp = spacy.load('en_core_web_sm')
    print("SpaCy model 'en_core_web_sm' loaded.")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Attempting to download...")
    try:
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
        print("SpaCy model 'en_core_web_sm' downloaded and loaded successfully.")
    except Exception as e:
        print(f"Failed to download or load spaCy model: {e}")
        print("Please try running 'python -m spacy download en_core_web_sm' in your terminal.")
        nlp = None # Set to None if loading fails

# Initialize NLTK components only if necessary resources are available
vader_analyzer = None
try:
    vader_analyzer = SentimentIntensityAnalyzer()
except LookupError:
    print("VADER lexicon not available. VADER sentiment analysis won't work.")

lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return []

    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization, lemmatization, and stop word removal using spaCy
    if nlp is not None:
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in english_stopwords and token.is_alpha]
    else:
        # Fallback to NLTK if spaCy failed to load (less robust)
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in english_stopwords and word.isalpha()]

    return tokens

def get_vader_sentiment(text):
    if vader_analyzer is None:
        return 'Neutral', 0.0 # Return neutral if VADER is not initialized
    
    score = vader_analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive', score
    elif score <= -0.05:
        return 'Negative', score
    else:
        return 'Neutral', score