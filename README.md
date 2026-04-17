# 📰 Fake News Detection Using NLP Classifiers

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## 🎓 Course Details

| Field | Details |
|---|---|
| **Course** | Predictive Analytics |
| **Institution** | Digital University Kerala |
| **Instructor** | Aswin S |
| **Year** | 2025–2027 |

---

## 👥 Team Members

| Name | GitHub |
|---|---|
| Anagha Suresh | 
| Angel Treesa John |
| Yadhu |

---

## 📌 Problem Statement

Fake news has become a critical issue in the digital era, spreading misinformation
rapidly through social media and online platforms. This project builds an automated
fake news detection system using NLP techniques and machine learning classifiers
to distinguish between **Real** and **Fake** news articles with high accuracy.

---

## 💡 Motivation

- Prevent the spread of misinformation in digital media
- Assist users in verifying news authenticity in real time
- Apply and compare NLP techniques (TF-IDF, BERT) on a real-world problem
- Understand the strengths and limitations of different ML classifiers

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Name** | WELFake Dataset |
| **Source** | [Kaggle – WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| **Size** | ~75,000+ samples |
| **Features** | `title`, `text`, `label` |
| **Label** | 0 = Fake, 1 = Real |
| **Class Distribution** | Fake: ~52% · Real: ~48% |

---

## 🔬 Methodology — Data Science Life Cycle

### Stage 1: Problem Definition & Literature Review
- Defined the task as binary text classification (Fake vs Real)
- Reviewed prior work on misinformation detection using TF-IDF baselines,
  BERT-based models, and linguistic feature engineering

### Stage 2: Data Collection & Data Understanding
- Loaded WELFake dataset from CSV
- Examined dataset structure, column types, label distribution, and missing values

### Stage 3: Data Preprocessing & Cleaning
- Lowercased all text
- Removed punctuation, numbers, and special characters
- Tokenization using NLTK
- Stopword removal
- Lemmatization for text normalization

### Stage 4: Exploratory Data Analysis (EDA)
- Visualized class distribution (Fake vs Real)
- Analysed word frequency using word clouds
- Compared text length distributions across both classes
- Identified linguistic patterns unique to fake news articles

### Stage 5: Feature Engineering & Selection
- **TF-IDF Vectorization** — for Logistic Regression and Gradient Boosting
- **BERT Embeddings** — using DistilBERT (HuggingFace Transformers)
- Compared feature representation effectiveness across models

### Stage 6: Model Building & Training
Three models trained on the processed dataset:

| Model | Feature Input |
|---|---|
| Logistic Regression | TF-IDF |
| Gradient Boosting | TF-IDF |
| DistilBERT Transformer | BERT Embeddings |

### Stage 7: Model Evaluation & Comparison
Evaluated all models using:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curve
- Cross-domain generalization analysis

### Stage 8: Model Interpretation & Explainability
- Compared performance of all 3 models side by side
- Error analysis — examined misclassified samples
- Identified key linguistic markers that influenced predictions

### Stage 9: Deployment
- Built an interactive web app using **Streamlit**
- Users can input news text and receive a prediction (Fake / Real)
- Displays confidence score and model explanation
- Models saved and loaded using:
  - `logistic_model.pkl`
  - `GradientBoosting_model.pkl`
  - `DistilBert.pkl`

### Stage 10: Documentation
- README (this file), PPT presentation, GitHub activity profiles

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Gradient Boosting | — | — | — | — |
| DistilBERT Transformer | — | — | — | — |

> ✅ **Best Model:** *(to be updated after evaluation)*

---

## 🖥️ Application Screenshots

### Home Page
![Home Page](images/front_pic_.jpeg)

### Prediction Output
![Prediction Output](images/prediction_pic.jpeg)
