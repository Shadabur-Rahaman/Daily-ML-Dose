# Zero-Shot Classification using Hugging Face pipeline

from transformers import pipeline

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Test input
text = "India won the cricket match by 6 wickets."
labels = ["sports", "politics", "entertainment", "technology"]

# Perform classification
result = classifier(text, candidate_labels=labels)
print("Labels:", result["labels"])
print("Scores:", result["scores"])
