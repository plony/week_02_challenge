import pandas as pd
import re
import os
import json
from nltk.corpus import stopwords
from datetime import datetime
import nltk

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # Remove non-alphanumeric
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text

def load_and_clean(bank_name):
    input_path = f"../data/raw/{bank_name}_reviews_raw.json"
    output_path = f"../data/processed/{bank_name}_reviews_cleaned.csv"

    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Clean and enrich data
    df['cleaned_review'] = df['content'].apply(clean_text)
    df['rating'] = df['score']
    df['date'] = pd.to_datetime(df['at']).dt.strftime('%Y-%m-%d')
    df['sentiment'] = df['rating'].apply(
        lambda x: 'positive' if x >= 4 else ('neutral' if x == 3 else 'negative')
    )

    # Select final columns
    df = df[['userName', 'rating', 'date', 'content', 'cleaned_review', 'sentiment']]

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved for {bank_name}")
    return df

if __name__ == "__main__":
    banks = ['cbe', 'boa', 'dashen']
    for bank in banks:
        load_and_clean(bank)