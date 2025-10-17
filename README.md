# Machine Learning in Network Intrusion Detection

## 📘 Overview
This project implements a Machine Learning–based Intrusion Detection System (IDS) using the **KDDCup’99** dataset.  
The goal is to classify network connections as either *normal* or *attack* using supervised learning models.

---

## 🧩 Algorithms Implemented
1. Gaussian Naive Bayes  
2. Decision Tree (max depth = 9)  
3. Random Forest (n_estimators = 16, max depth = 9)

---

## ⚙️ How to Run
```bash
# Clone and setup
git clone https://github.com/<username>/ml-intrusion-detection.git
cd ml-intrusion-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Train models
python -m scripts.train_models

# Evaluate and generate confusion matrices
python -m scripts.evaluate
