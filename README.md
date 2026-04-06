Fake News Detection using NLP classifiers


Fake News Detection using logistic regression, gradientBoosting and transformer model

Team Members
Anagha Suresh, Angel Treesa John, Yadhu 
🎓 Course Details
Course: Predictive analytics
Institution: Digital University Kerala
Instructor: Aswin.S
Year: 2025–2027

Problem Statement

Fake news has become a major issue in the digital era, spreading misinformation rapidly through social media and online platforms. The goal of this project is to build a machine learning model that can automatically classify news articles as Real or Fake based on textual content.

Motivation
Prevent misinformation spread
Assist users in verifying news authenticity
Apply NLP techniques to real-world problems
Understand and compare different text representation methods
Dataset Description
Source: WELFake dataset
Size: ~75,000+ samples 
Features:
title
text/content
label (0 = Fake, 1 = Real)
Class Distribution:
Fake: ~52%
Real: ~48%
⚙️ Methodology
Stage 1: Data Collection
Dataset loaded from CSV files

Stage 2: Data Understanding
Checked class distribution
Explored dataset structure

Stage 3: Data Preprocessing
Lowercasing text
Removing punctuation & numbers
Tokenization 
Stopword removal 

Stage 4: Feature Engineering
TF-IDF Vectorization
BERT embeddings
 
Stage 5: Model Building
Model used: Logistic Regression, GradientBoost, Transformer
Trained on processed dataset

Stage 6: Evaluation
Metrics used:
Accuracy
Precision
Recall
F1-score
Stage 7: Error Analysis
Compared performance of all 3 models 

Stage 8: Model Saving
Model saved using:
pickle.dump(model, open('model.pkl', 'wb'))
logistic_model (4).pkl
GradientBoosting_model.pkl
DistilBert.pkl

Stage 9: Deployment
Built an interactive web app using Streamlit
Users can input news text and get prediction (Fake/Real)

Results Summary
Model	Accuracy
Logistic Regression:
GradientBoosting: 
DistilBert: 


Best model: (Your best model name)

Application Screenshots

(Add screenshots after deployment)

Home Page
Prediction Output
🚀 How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the App
streamlit run app.py
🌐 Live Deployment

🔗 Streamlit App:
(Add your deployed link here)

🛠️ Technologies Used
Python
Scikit-learn
Pandas, NumPy
Matplotlib / Seaborn
Streamlit
NLP (TF-IDF / BERT)

Conclusion
This project successfully demonstrates how NLP techniques can be used to detect fake news. The deployed application allows real-time predictions, making it useful for practical scenarios.
