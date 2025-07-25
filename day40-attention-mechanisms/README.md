# 🧠 Day 40 – Attention Mechanisms  
> Attention in NLP and Vision; Self-Attention and Transformers  
📅 #DailyMLDose

---

## 📌 Overview

Attention mechanisms revolutionized how models learn long-range dependencies in data.  
From neural machine translation to image captioning and transformers, attention helps models **focus** on the most relevant parts of input sequences.

In this session, we explore:
- The intuition behind attention
- Scaled Dot-Product & Multi-Head Attention
- Self-attention in Transformers
- Applications in NLP and Vision (ViT, BERT, GPT, etc.)

---

## 🎯 Key Concepts

| Concept                     | Description |
|-----------------------------|-------------|
| **Basic Attention**         | Assigning importance to input parts during prediction |
| **Self-Attention**          | Attention applied within the same sequence |
| **Multi-Head Attention**    | Captures information from multiple representation subspaces |
| **Positional Encoding**     | Preserves token order in transformers |
| **Transformers**            | Model architecture based solely on attention |
| **Cross-Attention**         | Source-target attention in encoder-decoder models |
| **Vision Transformers**     | Applies transformer architecture to image patches |

---

## 🧠 Visual Explanations

### 🎯 1. What is Attention?
![Basic Attention](../assets/day40/basic_attention.png)

---

### 🔁 2. Self-Attention Flow (as in Transformers)
![Self Attention](../assets/day40/self_attention.png)

---

### 🧠 3. Multi-Head Attention Structure
![Multi-Head Attention](../assets/day40/multihead_attention.png)

---

### 🧬 4. Positional Encoding
![Positional Encoding](../assets/day40/positional_encoding.png)

---

### 🖼️ 5. Vision Transformer Patch Encoding
![ViT Attention](../assets/day40/vision_transformer.png)

---

## 📁 Folder Stucture
```css
 `day40-attention-mechanisms/`  
├── basic_attention_numpy.py
├── self_attention_scratch.py
├── multihead_attention_demo.py
├── positional_encoding.py
├── transformer_nlp_pipeline.py
├── vit_image_classification.py
```
File	Description
basic_attention_numpy.py	Simulates simple attention using NumPy
self_attention_scratch.py	Raw self-attention implementation
multihead_attention_demo.py	Multi-head in PyTorch
positional_encoding.py	Sinusoidal + learnable PE
transformer_nlp_pipeline.py	Text classification using transformers
vit_image_classification.py	Image classification using Vision Transformer
```
🧪 Sample Snippet
```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn
```
🔗 Related Posts
🔙 Day 37: Advanced NLP Applications

🔜 Day 41: Transfer Learning in Vision (Coming Soon)

🔍 References
Vaswani et al. (2017) Attention is All You Need

Annotated Transformer: http://nlp.seas.harvard.edu/2018/04/03/attention.html

Jay Alammar Visualizations: https://jalammar.github.io/illustrated-transformer/

🔖 Hashtags
#AttentionMechanism #Transformers #VisionTransformer #DeepLearning #NLP #DailyMLDose #100DaysOfML
