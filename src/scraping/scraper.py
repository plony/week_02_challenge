from google_play_scraper import reviews, Sort
import json
import os
import time
from datetime import datetime

def convert_dates_to_strings(data):
    """Recursively convert datetime objects to ISO format strings."""
    if isinstance(data, dict):
        return {key: convert_dates_to_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_dates_to_strings(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()  # Convert datetime to string (e.g., "2024-05-10T12:30:00")
    else:
        return data

def scrape_app_reviews(app_id, bank_name, count=400):
    print(f"🔍 Scraping {bank_name} ({app_id})...")
    try:
        result, continuation_token = reviews(
            app_id,
            lang='any',
            country='ET',
            sort=Sort.NEWEST,
            count=count
        )
        print(f"📄 Retrieved {len(result)} reviews")
        if result:
            print(f"📝 Sample review: {result[0]['content'][:100]}...")

        # Convert datetime objects to strings
        cleaned_result = convert_dates_to_strings(result)

        output_path = f"../../data/raw/{bank_name}_reviews_raw.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_result, f, ensure_ascii=False, indent=2)

        print(f"✅ Successfully saved reviews for {bank_name}")
        return cleaned_result

    except Exception as e:
        print(f"❌ Error scraping {bank_name}: {str(e)}")
        return []

if __name__ == "__main__":
    apps = {
        "cbe": "com.combanketh.mobilebanking",
        "boa": "com.boa.boaMobileBanking",
        "dashen": "com.dashen.dashensuperapp"
    }
    for name, app_id in apps.items():
        scrape_app_reviews(app_id, name)
        time.sleep(5)