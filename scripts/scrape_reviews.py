from google_play_scraper import reviews, Sort
import pandas as pd
import time

# App IDs (you need to find these from Google Play URL)
app_ids = {
    "Commercial Bank of Ethiopia": "com.cbe.mobilebanking",
    "Bank of Abyssinia": "com.boa.mobile",
    "Dashen Bank": "com.dashen.mobile"
}

all_reviews = []

for bank_name, app_id in app_ids.items():
    result, continuation_token = reviews(
        app_id,
        lang='en',
        country='ET',
        sort=Sort.NEWEST,
        count=400
    )
    for r in result:
        all_reviews.append({
            'review': r['content'],
            'rating': r['score'],
            'date': r['at'].date(),
            'bank': bank_name,
            'source': 'Google Play'
        })
    print(f"Scraped {len(result)} reviews from {bank_name}")
    time.sleep(5)  # Avoid rate limits

df = pd.DataFrame(all_reviews)
df.to_csv('data/raw/reviews_raw.csv', index=False)
print("Raw reviews saved.")