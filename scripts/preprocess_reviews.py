import pandas as pd
import re
from datetime import datetime
import os

# Load raw data
df = pd.read_csv('data/raw/reviews_raw.csv')

# Remove duplicates
df.drop_duplicates(subset=['review'], keep='first', inplace=True)

# Handle missing values
df.dropna(subset=['review'], inplace=True)

# Normalize date format
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

# Clean review text
def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'[^a-zA-Z0-9\s.!?]', '', text).strip()
    return ''

df['review'] = df['review'].apply(clean_text)

# Save cleaned data
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/reviews_cleaned.csv', index=False)
print("Cleaned reviews saved.")