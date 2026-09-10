import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Machine learning algorithms analyze structured data effectively",
    "Deep learning and neural networks excel at processing unstructured data",
    "Natural language processing helps computers understand human language",
    "Python is widely used for machine learning and data science"
]

query = ["machine learning algorithms for data"]

vectorizer = CountVectorizer(stop_words='english')
doc_vectors = vectorizer.fit_transform(documents)

query_vector = vectorizer.transform(query)

similarity_scores = cosine_similarity(query_vector, doc_vectors)
print(similarity_scores)

scores_flat = similarity_scores.flatten()

ranking_df = pd.DataFrame({
    "Document": [f"Document {i+1}" for i in range(len(documents))],
    "Text": documents,
    "Cosine Similarity": scores_flat
})

ranking_df = ranking_df.sort_values(by="Cosine Similarity", ascending=False).reset_index(drop=True)
print(ranking_df)