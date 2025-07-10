from gensim.models import Word2Vec

# Sample tokenized sentences
sentences = [
    ["natural", "language", "processing", "is", "fun"],
    ["deep", "learning", "boosts", "NLP", "performance"],
    ["word2vec", "creates", "semantic", "embeddings"]
]

# Train Word2Vec model using skip-gram
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=1)

# Access vector for a word
print("Vector for 'NLP':", model.wv["NLP"][:5])
