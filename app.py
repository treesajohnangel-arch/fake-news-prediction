
import gdown, os, zipfile

BERT_PATH = '/tmp/bert_model'

if not os.path.exists(BERT_PATH):
    print("Downloading BERT model...")
    gdown.download(
        'https://drive.google.com/file/d/1X5C5yhOgc8v...',  # your actual link
        '/tmp/bert_model.zip',
        fuzzy=True
    )
    with zipfile.ZipFile('/tmp/bert_model.zip', 'r') as z:
        z.extractall('/tmp/bert_model')
    print("✅ BERT ready!")


import streamlit as st
import pickle, torch, os
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

st.set_page_config(page_title="FakeGuard AI", page_icon="🛡️", layout="wide")

st.markdown("""<style>
.stApp {
  background: #0b0f19 !important;
}

/* Container */
.block-container {
  padding-top: 2rem !important;
  max-width: 820px !important;
}

/* Hero Section */
.hero {
  background: #111827;
  border-radius: 20px;
  border: 1.5px solid #d4af37;
  padding: 36px 32px 28px;
  text-align: center;
  margin-bottom: 20px;
  box-shadow: 0 0 25px rgba(212, 175, 55, 0.15);
}

/* Title */
.hero h1 {
  color: #ffffff !important;
}

/* Subtitle */
.hero p {
  color: #cbd5e1 !important;
}

/* Tags */
.hero span {
  border: 1px solid #d4af37 !important;
  background: transparent !important;
  color: #d4af37 !important;
}

/* Text Area */
.stTextArea textarea {
  background: #111827 !important;
  border: 1.5px solid #d4af37 !important;
  border-radius: 12px !important;
  color: #ffffff !important;
  caret-color: #d4af37 !important;
  font-size: 14px !important;
}

/* Placeholder */
.stTextArea textarea::placeholder {
  color: #94a3b8 !important;
}

/* Label */
.stTextArea label {
  color: #ffffff !important;
  font-weight: 600;
}

/* Button */
.stButton>button {
  background: linear-gradient(135deg, #d4af37, #f5d76e);
  color: #000 !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  width: 100%;
  padding: 0.7rem 2rem !important;
  transition: 0.3s;
}

/* Button Hover */
.stButton>button:hover {
  background: linear-gradient(135deg, #c5a028, #e6c55a);
  transform: scale(1.03);
}

/* Cards */
<div class="card">
.card {
  background: #111827 !important;
  border-radius: 16px;
  border: 1.5px solid #d4af37;
  color: #ffffff;
  box-shadow: 0 0 20px rgba(212, 175, 55, 0.1);
  padding: 16px;
}

/* Text fixes */
div, p, span {
  color: #ffffff !important;
}

/* Footer remove */
footer {
  visibility: hidden;
}

</style>""", unsafe_allow_html=True)

            

st.markdown("""
<div class="hero">
  <h1 style="font-size:30px;font-weight:700;color:#0f172a;margin-bottom:8px">🛡️ Fake<span style="color:#4f8ef7">Guard</span> AI</h1>
  <p style="color:#000000;font-size:14px;max-width:480px;margin:0 auto 18px">Three AI models analyze your text and vote on whether it's real or fake.</p>
  <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap">
    <span style="background:#eff6ff;color:#3b82f6;border:1.5px solid #d4af37;  /* gold */;border-radius:20px;padding:4px 12px;font-size:12px;font-weight:600">📈 Logistic Regression</span>
    <span style="background:#f0fdf4;color:#22c55e;border:1.5px solid #d4af37;  /* gold */;border-radius:20px;padding:4px 12px;font-size:12px;font-weight:600">🌲 Gradient Boosting</span>
    <span style="background:#faf5ff;color:#a855f7;border:1.5px solid #d4af37;  /* gold */;border-radius:20px;padding:4px 12px;font-size:12px;font-weight:600">🧠 DistilBERT</span>
    <span style="background:#fffbeb;color:#f59e0b;border:1.5px solid #d4af37;  /* gold */;border-radius:20px;padding:4px 12px;font-size:12px;font-weight:600">📊 LIAR Dataset</span>
  </div>
</div>""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    with open('models/lr_model.pkl','rb') as f: lr=pickle.load(f)
    with open('models/gb_model.pkl','rb') as f: tfidf,gb=pickle.load(f)
    tok=DistilBertTokenizer.from_pretrained('/tmp/bert_model')
    bert=DistilBertForSequenceClassification.from_pretrained('/tmp/bert_model'); bert.eval()
    return lr,tfidf,gb,tok,bert

with st.spinner("Loading models..."): lr_model,tfidf,gb_model,tokenizer,bert_model=load_models()

text=st.text_area("📰 Paste news text",placeholder="Enter any news headline or article text...",height=150)
analyze=st.button("🔍 Analyze Now")

if analyze and text.strip():
    with st.spinner("Analyzing..."):
        lr_pred=lr_model.predict([text])[0]; lr_prob=lr_model.predict_proba([text])[0][1]
        X=tfidf.transform([text]); gb_pred=gb_model.predict(X)[0]; gb_prob=gb_model.predict_proba(X)[0][1]
        inp=tokenizer(text,return_tensors='pt',truncation=True,max_length=128)
        with torch.no_grad(): out=bert_model(**inp)
        probs=torch.softmax(out.logits,dim=1).numpy()[0]; bert_pred=int(np.argmax(probs)); bert_prob=float(probs[1])

    labels={0:"✅ Real",1:"❌ Fake"}
    colors={0:"#000000",1:"#dc2626"}
    borders={0:"#bbf7d0",1:"#fecaca"}
    tops={0:"#22c55e",1:"#ef4444"}

    c1,c2,c3=st.columns(3)
    for col,name,pred,prob,top_c,icon in [
        (c1,"Logistic Regression",lr_pred,lr_prob,"#4f8ef7","📈"),
        (c2,"Gradient Boosting",gb_pred,gb_prob,"#22c55e","🌲"),
        (c3,"DistilBERT",bert_pred,bert_prob,"#a855f7","🧠")]:
        with col:
            st.markdown(f"""
            <div class="card">
              <div style="height:4px;background:{top_c}"></div>
              <div style="padding:18px 14px">
                <div style="font-size:24px;margin-bottom:6px">{icon}</div>
                <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">{name}</div>
                <div style="font-size:18px;font-weight:700;color:{colors[pred]};margin-bottom:10px">{labels[pred]}</div>
                <div style="background:#f1f5f9;border-radius:8px;height:7px;margin-bottom:6px">
                  <div style="background:{'#22c55e' if pred==0 else '#ef4444'};height:7px;border-radius:8px;width:{prob*100:.0f}%"></div>
                </div>
                <div style="font-size:11px;color:#000000">Fake probability: {prob:.0%}</div>
              </div>
            </div>""",unsafe_allow_html=True)

    votes=[lr_pred,gb_pred,bert_pred]; final=1 if sum(votes)>=2 else 0
    avg=(lr_prob+gb_prob+bert_prob)/3
    agree=sum(v==final for v in votes)
    
    st.markdown(f"""
<div class="card">

  <div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">
    🗳️ Ensemble Verdict
  </div>

  <div style="font-size:30px;font-weight:800;color:{colors[final]};margin-bottom:6px">
    {'✅ Real News' if final==0 else '❌ Fake News'}
  </div>

  <div style="font-size:13px;color:#cbd5e1">
    {agree} of 3 models agree · Average fake probability: {avg:.0%}
  </div>

  <div style="display:flex;gap:10px;justify-content:center;margin-top:14px;flex-wrap:wrap">
    <span class="tag">📈 {'Real' if lr_pred==0 else 'Fake'}</span>
    <span class="tag">🌲 {'Real' if gb_pred==0 else 'Fake'}</span>
    <span class="tag">🧠 {'Real' if bert_pred==0 else 'Fake'}</span>
  </div>

</div>
""", unsafe_allow_html=True)

    col_a,col_b=st.columns(2)
    with col_a:
        st.markdown("""<div class="card">""",unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class="card">""",unsafe_allow_html=True)
