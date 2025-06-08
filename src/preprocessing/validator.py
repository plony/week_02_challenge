import pandas as pd

def validate_data(bank_name):
    input_path = f"../../data/processed/{bank_name}_reviews_cleaned.csv"
    df = pd.read_csv(input_path)

    missing = df.isnull().sum().to_dict()
    duplicates = df.duplicated().sum()

    print(f"🔍 Validation for {bank_name}:")
    print(f"Missing values: {missing}")
    print(f"Duplicates: {duplicates}")
    assert duplicates == 0, "❌ Duplicates found!"
    assert df['review'].notnull().all(), "❌ Missing reviews found!"

if __name__ == "__main__":
    banks = ['cbe', 'boa', 'dashen']
    for bank in banks:
        validate_data(bank)