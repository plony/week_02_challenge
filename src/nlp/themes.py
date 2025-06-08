from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

def extract_topics(df, n_topics=5):
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_review'])

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)

    features = tfidf.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [features[i] for i in topic.argsort()[-10:]]
        topics.append(f"Topic {idx+1}: {', '.join(top_words)}")
    return topics

if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/boa_reviews_cleaned.csv")
    topics = extract_topics(df)
    print("🧠 Extracted Topics:")
    for t in topics:
        print(t)