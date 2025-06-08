# fintech_app_analytics/scripts/analyze_reviews.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import os
import sys # <-- Add this import

# --- IMPORTANT: Adjust sys.path for direct script execution or notebook import ---
# This ensures that 'utils.py' can be found whether this script is run directly
# or imported from a notebook that has adjusted its own sys.path.
# We ensure the parent directory of 'scripts' is on the path if not already.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning to prioritize

# Now, import from 'scripts.utils' assuming 'project_root' is on sys.path
# This makes it an "absolute" import relative to the project root.
from scripts.utils import preprocess_text, get_vader_sentiment, nlp # Import utils functions

# For Hugging Face sentiment (if chosen)
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

def perform_sentiment_analysis(df, method='vader'):
    """
    Performs sentiment analysis on the 'Review Text' column.
    Method can be 'vader' or 'huggingface'.
    """
    print(f"\nPerforming sentiment analysis using {method}...")

    if 'Review Text' not in df.columns:
        print("Error: 'Review Text' column not found in DataFrame.")
        return df

    if method == 'vader':
        df[['Sentiment', 'Sentiment_Score']] = df['Review Text'].apply(lambda x: pd.Series(get_vader_sentiment(x)))
    elif method == 'huggingface':
        # Load Hugging Face model and tokenizer once
        print("Loading Hugging Face 'distilbert-base-uncased-finetuned-sst-2-english' model...")
        hf_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        hf_model = AutoModelForSequenceClassification.from_pretrained(hf_model_name)
        
        sentiment_pipeline = pipeline("sentiment-analysis", model=hf_model, tokenizer=hf_tokenizer)
        
        batch_size = 32
        sentiments = []
        scores = []
        for i in range(0, len(df), batch_size):
            batch_texts = df['Review Text'].iloc[i:i+batch_size].tolist()
            batch_texts = [text if isinstance(text, str) else "" for text in batch_texts]
            
            results = sentiment_pipeline(batch_texts)
            for res in results:
                sentiments.append(res['label'].capitalize())
                scores.append(res['score'])
        
        df['Sentiment'] = sentiments
        df['Sentiment_Score'] = scores
        df['Sentiment'] = df['Sentiment'].replace({'Positive': 'Positive', 'Negative': 'Negative'})
        
    else:
        raise ValueError("Invalid sentiment analysis method. Choose 'vader' or 'huggingface'.")

    print("Sentiment analysis complete.")
    print("\nSentiment Distribution:")
    print(df['Sentiment'].value_counts())
    return df

def perform_thematic_analysis(df):
    """
    Performs thematic analysis: keyword extraction (TF-IDF) and rule-based clustering.
    """
    print("\nStarting thematic analysis...")

    df['Processed_Reviews_Tokens'] = df['Review Text'].apply(lambda x: preprocess_text(x) if pd.notna(x) else [])
    
    df_filtered = df[df['Processed_Reviews_Tokens'].apply(len) > 0].copy()

    if df_filtered.empty:
        print("No valid reviews to perform thematic analysis.")
        df['Extracted_Keywords'] = [[]] * len(df)
        df['Identified_Theme'] = 'N/A'
        return df

    df_filtered['Processed_Reviews_Text'] = df_filtered['Processed_Reviews_Tokens'].apply(lambda x: ' '.join(x))

    print("Performing TF-IDF for keyword extraction...")
    tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_filtered['Processed_Reviews_Text'])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    def get_top_tfidf_keywords(text_index, top_n=5):
        row = tfidf_matrix[text_index].todense()
        sorted_indices = row.argsort()[0, ::-1]
        top_keywords = [feature_names[i] for i in sorted_indices.tolist()[0][:top_n]]
        return top_keywords

    df_filtered['Extracted_Keywords'] = [get_top_tfidf_keywords(i) for i in range(tfidf_matrix.shape[0])]
    print("TF-IDF keyword extraction complete.")

    print("Applying rule-based thematic clustering...")
    
    theme_keywords = {
        'Account Access Issues': ['login', 'log in', 'password', 'otp', 'access', 'fingerprint', 'faceid', 'pin', 'locked'],
        'Transaction Performance': ['transfer', 'send money', 'slow', 'fast', 'loading', 'delay', 'transaction', 'payment', 'speed'],
        'User Interface & Experience': ['ui', 'interface', 'design', 'easy', 'difficult', 'confusing', 'user friendly', 'layout', 'bug', 'crash', 'freeze', 'update'],
        'Customer Support': ['support', 'customer service', 'help', 'contact', 'call', 'response', 'agent'],
        'Feature Requests': ['feature', 'add', 'wish', 'want', 'suggest', 'improve', 'need', 'require', 'update']
    }

    def assign_theme(review_text_tokens):
        found_themes = []
        review_words = set(review_text_tokens)

        for theme, keywords in theme_keywords.items():
            if any(keyword in review_words for keyword in keywords):
                found_themes.append(theme)
        
        if found_themes:
            return ', '.join(found_themes)
        else:
            return 'Other/General Feedback'

    df_filtered['Identified_Theme'] = df_filtered['Processed_Reviews_Tokens'].apply(assign_theme)
    
    df_result = df.copy()
    df_result = df_result.merge(df_filtered[['reviewId', 'Processed_Reviews_Tokens', 'Extracted_Keywords', 'Identified_Theme']], 
                                how='left', on='reviewId', suffixes=('', '_new'))
    
    df_result['Processed_Reviews_Tokens'] = df_result['Processed_Reviews_Tokens_new'].fillna(df_result['Processed_Reviews_Tokens'])
    df_result['Extracted_Keywords'] = df_result['Extracted_Keywords'].fillna(value='[]')
    df_result['Identified_Theme'] = df_result['Identified_Theme'].fillna(value='Other/General Feedback')
    
    df_result.drop(columns=['Processed_Reviews_Tokens_new'], inplace=True)

    print("Thematic analysis complete.")
    print("\nTheme Distribution:")
    print(df_result['Identified_Theme'].value_counts())

    return df_result

if __name__ == "__main__":
    processed_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
    input_filepath = os.path.join(processed_data_dir, 'fintech_app_reviews_processed.csv')
    output_filepath = os.path.join(processed_data_dir, 'fintech_app_reviews_analyzed.csv')

    if not os.path.exists(input_filepath):
        print(f"Error: Processed data file not found at {input_filepath}. Please run scrape_reviews.py first.")
    else:
        df = pd.read_csv(input_filepath)
        df['reviewId'] = df['reviewId'].astype(str)
        df = perform_sentiment_analysis(df, method='huggingface')
        df = perform_thematic_analysis(df)
        df.to_csv(output_filepath, index=False)
        print(f"\nAnalyzed reviews saved to {output_filepath}")