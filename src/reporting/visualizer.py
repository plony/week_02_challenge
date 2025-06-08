import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_sentiment_distribution(df, bank_name):
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='sentiment')
    plt.title(f"{bank_name} Sentiment Distribution")
    plt.savefig(f"../../assets/images/{bank_name}_sentiment.png")
    plt.close()
    print(f"📈 Saved sentiment chart for {bank_name}")

if __name__ == "__main__":
    banks = ['cbe', 'boa', 'dashen']
    for bank in banks:
        df = pd.read_csv(f"../../data/processed/{bank}_reviews_with_sentiment.csv")
        plot_sentiment_distribution(df, bank)