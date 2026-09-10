import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
 
corpus = [
 "The product performance is amazing and fast",
 "The service was fast and performance was great",
 "Terrible customer service and bad performance"
]

vectorizer = CountVectorizer(stop_words='english')

bow_matrix = vectorizer.fit_transform(corpus)

vocabulary = vectorizer.get_feature_names_out()
print(vocabulary)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=[f"Document {i+1}" for i in range(len(corpus))]
)
print(bow_df)