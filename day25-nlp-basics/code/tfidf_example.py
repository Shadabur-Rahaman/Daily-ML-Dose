from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "Natural Language Processing is fun",
    "Language is a part of artificial intelligence",
    "Deep learning improves NLP tasks"
]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
X = vectorizer.fit_transform(documents)

# Display the feature names
print("Vocabulary:", vectorizer.get_feature_names_out())

# Display TF-IDF scores
print("TF-IDF Matrix:\n", X.toarray())
