#/scripts/utils.py

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import spacy

# --- NLTK and spaCy Downloads (ensure these are run once) ---
try:
    nltk.data.find('corpora/vader_lexicon')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except nltk.downloader.DownloadError:
    nltk.download('omw-1.4')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
vader_analyzer = SentimentIntensityAnalyzer()

def preprocess_text(text):
    """
    Cleans and preprocesses text for NLP analysis.
    Steps: lowercasing, punctuation removal, tokenization, stop word removal, lemmatization.
    """
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
    tokens = word_tokenize(text)
    
    # Remove stop words and lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word not in stop_words and len(word) > 2 # Remove short words
    ]
    return processed_tokens

def get_vader_sentiment(text):
    """
    Computes sentiment (Positive, Negative, Neutral) using VADER.
    Returns a tuple: (sentiment_label, compound_score).
    """
    if not isinstance(text, str):
        return 'Neutral', 0.0
    
    vs = vader_analyzer.polarity_scores(text)
    compound_score = vs['compound']
    
    if compound_score >= 0.05:
        return 'Positive', compound_score
    elif compound_score <= -0.05:
        return 'Negative', compound_score
    else:
        return 'Neutral', compound_score

def get_huggingface_sentiment(text, model, tokenizer):
    """
    Computes sentiment using a Hugging Face pre-trained model (e.g., DistilBERT).
    Requires 'transformers' and 'torch'.
    Returns a tuple: (sentiment_label, score).
    """
    if not isinstance(text, str) or not text.strip():
        return 'Neutral', 0.0

    # Hugging Face models are typically fine-tuned on specific max sequence lengths
    # DistilBERT has a max length of 512. Truncate if necessary.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # The output structure depends on the model; for sentiment, it's often logits.
    # For 'distilbert-base-uncased-finetuned-sst-2-english', the output is logits
    # for positive and negative classes.
    # The SST-2 dataset labels are usually 0 (negative), 1 (positive).
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get the predicted label index (0 or 1)
    predicted_class_id = predictions.argmax().item()
    
    # Get the confidence score for the predicted class
    score = predictions[0][predicted_class_id].item()

    # Map the class ID to a sentiment label
    # This might need adjustment based on the specific model's labels
    # For sst-2-english: 0=negative, 1=positive
    sentiment_label = "Positive" if predicted_class_id == 1 else "Negative"

    # You might want to define a 'Neutral' threshold if the score is very low for both.
    # For simplicity here, we'll assign positive/negative based on the higher score.
    # If the model gives probabilities for all 3 (pos/neg/neu), adjust this logic.
    
    # A simple threshold for neutral, if applicable:
    if score < 0.65: # Arbitrary threshold, tune this if needed
        return 'Neutral', score

    return sentiment_label, score