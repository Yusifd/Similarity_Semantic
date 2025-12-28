# Similarity_Semantic
# 📖 Semantic Similarity Evaluation: Old vs Modern Sentences

# 🎯 Objective
This project evaluates how **XLM-RoBERTa** and **BERT** perform in measuring semantic similarity between **old literary sentences** and their **modern equivalents**.  
The aim is to understand how well each model bridges linguistic gaps across **time-evolving language styles**.

# ⚙️ Workflow

# 1️⃣ Install Required Libraries
```bash
!pip install transformers datasets sentencepiece scikit-learn

# 2️⃣ Import Libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# 3️⃣ Dataset
- Old Sentences → Classical poetic and archaic expressions
- Modern Sentences → Simplified, contemporary equivalents
# 4️⃣ Sentiment Analysis
Models used:
- 🟣 XLM-RoBERTa (xlm-roberta-base)
- 🟢 BERT (bert-base-multilingual-cased)
# 5️⃣ Metrics
Evaluated Precision, Recall, and F1-score for both models.
# 6️⃣ Sentence Similarity
- Embeddings generated using XLM-RoBERTa and BERT
- Cosine similarity applied to compare old vs modern sentence pairs
# 📊 Results
✅ XLM-RoBERTa
- Accuracy in similarity detection: 99.11% – 99.8%
- Superior at bridging linguistic gaps between old and modern structures
⚠️ BERT
- Accuracy range: 75% – 95%
- Effective in general semantic tasks, but less consistent with nuanced, time-evolving language
🔑 Key Insights
- XLM-RoBERTa shows robust performance in semantic similarity across historical and modern texts
- BERT demonstrates variability, highlighting limitations in capturing nuanced linguistic evolution
- For applications requiring high precision, XLM-RoBERTa is the recommended choice
# 🚀 How to Run
Clone the repository
git clone https://github.com/Yusifd/Similarity_Semantic.git
cd Similarity_Semantic

Install dependencies
pip install -r requirements.txt


Run the notebook or script
python similarity_eval.py


# 📌 Applications
- 📝 Digital Humanities → Comparing classical literature with modern translations
- 🌍 Cross-lingual NLP → Bridging linguistic gaps across time and culture
- 🤖 AI-powered text analysis → Enhancing semantic search and historical text understanding
3 ✨ Example Output
Similarity between old and modern sentence 1: 0.9911
Similarity between old and modern sentence 2: 0.9980
...


# 🏆 Conclusion
This project highlights the strength of XLM-RoBERTa in semantic similarity tasks involving historical vs modern language.
It provides a reproducible workflow for evaluating NLP models in linguistic evolution studies.
# 📚 References
- HuggingFace Transformers → https://huggingface.co/transformers
- Scikit-learn Metrics → https://scikit-learn.org/stable/modules/model_evaluation.html
# 👨‍💻 Author
Developed by Yusif
Passionate about applied data science, reproducibility, and bridging NLP with cultural contexts
