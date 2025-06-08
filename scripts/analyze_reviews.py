# fintech_app_analytics/scripts/analyze_reviews.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import os
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
        
        # Use pipeline for simplicity, or directly call get_huggingface_sentiment
        # Note: The pipeline abstracts the output, for direct score, you might use the function in utils.
        sentiment_pipeline = pipeline("sentiment-analysis", model=hf_model, tokenizer=hf_tokenizer)
        
        # Process in batches for efficiency (optional, but good for large datasets)
        batch_size = 32
        sentiments = []
        scores = []
        for i in range(0, len(df), batch_size):
            batch_texts = df['Review Text'].iloc[i:i+batch_size].tolist()
            # Handle non-string types in batch_texts
            batch_texts = [text if isinstance(text, str) else "" for text in batch_texts]
            
            # The pipeline returns a list of dictionaries like [{'label': 'POSITIVE', 'score': 0.999}]
            results = sentiment_pipeline(batch_texts)
            for res in results:
                sentiments.append(res['label'].capitalize()) # POSITIVE -> Positive
                scores.append(res['score'])
        
        df['Sentiment'] = sentiments
        df['Sentiment_Score'] = scores

        # Map POSITIVE/NEGATIVE to Positive/Negative for consistency with VADER's output
        df['Sentiment'] = df['Sentiment'].replace({'Positive': 'Positive', 'Negative': 'Negative'})
        
        # Re-map "NEUTRAL" if your HF model supports it, or if you want to infer from low scores
        # For SST-2 finetuned models, they are binary (positive/negative). 
        # A 'Neutral' might need to be defined based on confidence scores being below a threshold for both.
        # Example for setting Neutral based on low confidence:
        # df.loc[df['Sentiment_Score'] < 0.65, 'Sentiment'] = 'Neutral' # Adjust threshold as needed
        
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

    # Apply text preprocessing
    # Ensure 'Review Text' is not None or NaN before applying preprocess_text
    df['Processed_Reviews_Tokens'] = df['Review Text'].apply(lambda x: preprocess_text(x) if pd.notna(x) else [])
    
    # Filter out empty lists from preprocessing
    df_filtered = df[df['Processed_Reviews_Tokens'].apply(len) > 0].copy()

    if df_filtered.empty:
        print("No valid reviews to perform thematic analysis.")
        df['Extracted_Keywords'] = [[]] * len(df)
        df['Identified_Theme'] = 'N/A'
        return df

    # Rejoin tokens for TF-IDF
    df_filtered['Processed_Reviews_Text'] = df_filtered['Processed_Reviews_Tokens'].apply(lambda x: ' '.join(x))

    # TF-IDF for keyword extraction
    print("Performing TF-IDF for keyword extraction...")
    tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2)) # Consider unigrams and bigrams
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_filtered['Processed_Reviews_Text'])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    def get_top_tfidf_keywords(text_index, top_n=5):
        """Extracts top N TF-IDF keywords for a given document."""
        row = tfidf_matrix[text_index].todense()
        sorted_indices = row.argsort()[0, ::-1]
        top_keywords = [feature_names[i] for i in sorted_indices.tolist()[0][:top_n]]
        return top_keywords

    df_filtered['Extracted_Keywords'] = [get_top_tfidf_keywords(i) for i in range(tfidf_matrix.shape[0])]
    print("TF-IDF keyword extraction complete.")

    # --- Rule-Based Thematic Clustering ---
    print("Applying rule-based thematic clustering...")
    
    # Define keywords for themes. These are examples; refine based on actual review data.
    theme_keywords = {
        'Account Access Issues': ['login', 'log in', 'password', 'otp', 'access', 'fingerprint', 'faceid', 'pin', 'locked'],
        'Transaction Performance': ['transfer', 'send money', 'slow', 'fast', 'loading', 'delay', 'transaction', 'payment', 'speed'],
        'User Interface & Experience': ['ui', 'interface', 'design', 'easy', 'difficult', 'confusing', 'user friendly', 'layout', 'bug', 'crash', 'freeze', 'update'],
        'Customer Support': ['support', 'customer service', 'help', 'contact', 'call', 'response', 'agent'],
        'Feature Requests': ['feature', 'add', 'wish', 'want', 'suggestion', 'budgeting', 'bill pay', 'qr code', 'dark mode']
    }

    def assign_theme(review_text_tokens):
        """Assigns an overarching theme based on keywords in processed review tokens."""
        found_themes = []
        review_words = set(review_text_tokens) # For faster lookup

        for theme, keywords in theme_keywords.items():
            if any(keyword in review_words for keyword in keywords):
                found_themes.append(theme)
        
        if found_themes:
            # Prioritize themes if a review matches multiple, or return all
            return ', '.join(found_themes) # Returns a comma-separated string if multiple themes found
        else:
            return 'Other/General Feedback' # Default for reviews not matching specific themes

    df_filtered['Identified_Theme'] = df_filtered['Processed_Reviews_Tokens'].apply(assign_theme)
    
    # Merge back to original DataFrame
    # Ensure all original rows exist, fill NaNs for those filtered out
    df_result = df.copy()
    df_result = df_result.merge(df_filtered[['reviewId', 'Processed_Reviews_Tokens', 'Extracted_Keywords', 'Identified_Theme']], 
                                how='left', on='reviewId', suffixes=('', '_new'))
    
    # Fill NaN values for new columns that were created by the merge (for rows filtered out)
    df_result['Processed_Reviews_Tokens'] = df_result['Processed_Reviews_Tokens_new'].fillna(df_result['Processed_Reviews_Tokens'])
    df_result['Extracted_Keywords'].fillna(value='[]', inplace=True) # Fill with empty list string
    df_result['Identified_Theme'].fillna(value='Other/General Feedback', inplace=True)
    
    # Clean up merged columns
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

        # Ensure 'reviewId' is treated as string to avoid merge issues later
        df['reviewId'] = df['reviewId'].astype(str)

        # Perform Sentiment Analysis
        # Choose 'vader' or 'huggingface'
        df = perform_sentiment_analysis(df, method='huggingface') # Or 'vader'

        # Perform Thematic Analysis
        df = perform_thematic_analysis(df)

        # Save the analyzed DataFrame
        df.to_csv(output_filepath, index=False)
        print(f"\nAnalyzed reviews saved to {output_filepath}")