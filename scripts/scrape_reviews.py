# fintech_app_analytics/scripts/scrape_reviews.py

import pandas as pd
from google_play_scraper import Sort, reviews_all
import os
import datetime

def scrape_and_preprocess_reviews(app_ids_map, min_reviews_per_bank=400):
    """
    Scrapes reviews from Google Play Store for given app IDs,
    performs initial preprocessing, and returns a DataFrame.
    """
    all_reviews = []
    
    # Path to save raw data
    raw_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)

    for bank_name, app_id in app_ids_map.items():
        print(f"Scraping reviews for {bank_name} ({app_id})...")
        scraped_reviews_count = 0
        try:
            # Using reviews_all to get as many reviews as possible.
            # google-play-scraper automatically handles pagination up to its limits.
            # If rate limiting occurs or you need precise control, you can use `reviews`
            # function with `count` and `continuation_token`.
            
            # For the challenge, aiming for 400+ per bank, reviews_all is usually sufficient.
            # If it fetches too many or hits limits, consider using the 'reviews' function
            # with `count=min_reviews_per_bank` and then manually fetching more if needed.
            
            current_reviews = reviews_all(
                app_id,
                lang='en', # English reviews
                country='et', # Ethiopia
                sort=Sort.NEWEST # Prioritize newer reviews
            )
            scraped_reviews_count = len(current_reviews)

            if scraped_reviews_count < min_reviews_per_bank:
                print(f"Warning: Only scraped {scraped_reviews_count} reviews for {bank_name}. "
                      f"Target was {min_reviews_per_bank}. Consider re-running or checking app_id.")
            
            # Add bank and source info, then extend the main list
            for r in current_reviews:
                r['bank_name'] = bank_name
                r['source'] = 'Google Play'
                all_reviews.append(r)
            
            print(f"Finished scraping {scraped_reviews_count} reviews for {bank_name}.")

            # Save raw data for each bank
            raw_df_bank = pd.DataFrame(current_reviews)
            raw_filename = os.path.join(raw_data_dir, f"{bank_name.lower()}_reviews_raw.csv")
            raw_df_bank.to_csv(raw_filename, index=False)
            print(f"Raw reviews for {bank_name} saved to {raw_filename}")

        except Exception as e:
            print(f"Error scraping {bank_name} ({app_id}): {e}")

    if not all_reviews:
        print("No reviews were scraped. Exiting.")
        return pd.DataFrame()

    df = pd.DataFrame(all_reviews)

    # --- Preprocessing ---
    print("\nStarting data preprocessing...")

    # Select and rename necessary columns
    df = df[['reviewId', 'userName', 'score', 'at', 'content', 'bank_name', 'source']]
    df.rename(columns={
        'score': 'Rating',
        'at': 'Date',
        'content': 'Review Text',
        'bank_name': 'Bank/App Name',
        'source': 'Source',
        'userName': 'User Name' # Standardize column name
    }, inplace=True)

    # Convert 'Date' to datetime objects and normalize
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize() # Removes time component

    # Handle missing data (e.g., drop rows with missing review text or rating)
    initial_rows = len(df)
    df.dropna(subset=['Review Text', 'Rating'], inplace=True)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} rows with missing 'Review Text' or 'Rating'.")

    # Remove duplicates based on reviewId (unique identifier)
    initial_rows = len(df)
    df.drop_duplicates(subset=['reviewId'], inplace=True)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate reviews.")

    print(f"Preprocessing complete. Total unique reviews after cleaning: {len(df)}")
    print("\nSample of cleaned data:")
    print(df.head())
    print("\nReview counts per bank after cleaning:")
    print(df['Bank/App Name'].value_counts())

    return df

if __name__ == "__main__":
    # Actual Google Play Store App IDs (You need to find these by searching Google Play)
    # Example: search "CBE Mobile Banking" on Google Play, find the URL
    # e.g., https://play.google.com/store/apps/details?id=com.cbe.mobilebanking
    app_ids = {
        "CBE": "com.combanketh.mobilebanking", # Replace with actual CBE App ID
        "BOA": "com.boa.boaMobileBanking", # Replace with actual BOA App ID
        "Dashen": "com.dashen.dashensuperapp" # Replace with actual Dashen Bank App ID
    }

    # Define path for processed data
    processed_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
    os.makedirs(processed_data_dir, exist_ok=True)
    output_filepath = os.path.join(processed_data_dir, 'fintech_app_reviews_processed.csv')

    # Scrape and preprocess
    processed_df = scrape_and_preprocess_reviews(app_ids, min_reviews_per_bank=400)

    if not processed_df.empty:
        # Save the processed DataFrame
        processed_df.to_csv(output_filepath, index=False)
        print(f"\nProcessed reviews saved to {output_filepath}")
    else:
        print("No processed data to save.")