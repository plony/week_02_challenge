from textblob import TextBlob
import pandas as pd

def analyze_sentiment(df):
    df['sentiment_polarity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['predicted_sentiment'] = df['sentiment_polarity'].apply(
        lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
    )
    return df

if __name__ == "__main__":
    banks = ['cbe', 'boa', 'dashen']
    for bank in banks:
        input_path = f"../../data/processed/{bank}_reviews_cleaned.csv"
        df = pd.read_csv(input_path)
        df = analyze_sentiment(df)
        df.to_csv(f"../../data/processed/{bank}_reviews_with_sentiment.csv", index=False)
        print(f"📊 Sentiment analysis completed for {bank}")