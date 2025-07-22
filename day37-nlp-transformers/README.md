# 🌐 Day 37 – Advanced NLP Applications  
🧠 #DailyMLDose | Transformers, BERT & Zero-Shot Learning

Welcome to **Day 37** of #DailyMLDose!  
Today’s topic: Cutting-edge NLP powered by **Transformers**, **BERT**, and **Zero-shot Learning**.

---

## 📁 Folder Structure
```css
day37-advanced-nlp/
├── code/
│ ├── transformer_basics.py
│ ├── zero_shot_classification_hf.py
│ └── bert_embedding_example.py
│
├── images/
│ ├── transformer_architecture.png
│ ├── attention_is_all_you_need.png
│ ├── bert_visualization.png
│ ├── zero_shot_diagram.png
│ └── transfer_learning_nlp.png
└── README.md
```

---

## 🧠 From RNNs to Transformers

Traditional NLP models like **RNNs** struggled with long dependencies.  
The **Transformer architecture** (2017) solved this with **self-attention** and parallelization.

![Transformer Architecture](images/transformer_architecture.png)  
![Attention Paper](images/attention_is_all_you_need.png)

---

## 🔍 Transformer Architecture Key Points

- **Self-Attention** → Relates each word to all others
- **Multi-Head Attention** → Multiple attention heads learn diverse aspects
- **Feedforward Layers** → Nonlinear processing of each token
- **Positional Encoding** → Adds order to sequences

---

## 🧪 BERT and Friends

**BERT (Bidirectional Encoder Representations from Transformers)** understands text in context — both left and right!

It’s pre-trained on large corpora and fine-tuned for:
- Sentiment analysis
- Question answering
- Named entity recognition

![BERT Visualization](images/bert_visualization.png)

### 🔤 Code: Extract Embeddings with BERT
```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Transformers are powerful", return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```
🚀 Zero-Shot Learning (ZSL)
Zero-shot models predict without task-specific training — they match text to hypotheses.

Use case: Classify sentences into categories like finance, tech, sports without training on them!


📌 Hugging Face Pipeline Example
```python

from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    "The launch of the new iPhone is tomorrow.",
    candidate_labels=["technology", "sports", "politics"]
)
print(result["labels"])
```
🔄 Transfer Learning in NLP
Modern NLP → pretrained base model + task-specific fine-tuning.
This drastically reduces compute and improves results!


📊 Summary Table
Model / Technique	Use Case	Tools/Libraries
BERT	Text classification	🤗 Transformers
GPT	Text generation	OpenAI, LangChain
ZSL	Label new categories	facebook/bart-large-mnli
Transformers	Universal NLP tasks	PyTorch, TensorFlow

🔁 Previous Day
![📌 Day 36 – Ensemble Learning Techniques (Bagging, Stacking, Blending)](https://github.com/Shadabur-Rahaman/Daily-ML-Dose/tree/main/day36-ensemble-learning)

🙌 Let’s Connect!
📎 Connect With Me
- 🔗 [Follow Shadabur Rahaman on LinkedIn](https://www.linkedin.com/in/shadabur-rahaman-1b5703249)
⭐ Star the GitHub Repo
🔁 Share this if it helped!

“Language is the bridge between humans and machines. And Transformers are the architects.”
