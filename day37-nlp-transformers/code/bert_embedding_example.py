# BERT Embedding Extraction

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode text
inputs = tokenizer("Transformers are the backbone of modern NLP.", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Extract embeddings
last_hidden_state = outputs.last_hidden_state
print("Last hidden state shape:", last_hidden_state.shape)
print("Embedding of [CLS] token:", last_hidden_state[0][0])
