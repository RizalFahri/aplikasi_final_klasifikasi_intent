import streamlit as st
import pandas as pd
from datetime import date, timedelta
import torch
import numpy as np
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

# Import fungsi dari util.py
try:
    from util import (
        get_reviews_by_date_range,
        clean_text,
        normalize_text,
        load_kbba_dict
    )
except ImportError:
    st.error("⚠️ File 'util.py' tidak ditemukan di direktori.")

# =========================================================
# 1. KONFIGURASI MODEL & PATH (DEPLOY READY)
# =========================================================
# JANGAN gunakan path C:\Users\... gunakan repo Hugging Face
MODEL_NAME = "ree28/klasifikasiulasankai-indobert"
KBBA_FILE = "kbba.txt" 

LABEL_MAP = {
    0: "Pujian",
    1: "Keluhan",
    2: "Saran",
    3: "Laporan Kesalahan"
}

# =========================================================
# 2. FUNGSI LOAD DATA & MODEL (DENGAN PROTEKSI)
# =========================================================


@st.cache_resource
def load_essentials():
    try:
        # Kita paksa menggunakan BertTokenizer agar tidak "None" atau "Rusak"
        # use_fast=False lebih stabil untuk deployment
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        
        # Load Model
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error saat download model: {e}")
        return None, None
@st.cache_data
def load_dictionary():
    try:
        return load_kbba_dict(KBBA_FILE)
    except:
        return {}

# Inisialisasi
tokenizer, model = load_essentials()
KBBA_MAP = load_dictionary()

# =========================================================
# 3. LOGIKA PREPROCESS & KLASIFIKASI
# =========================================================

def classify_review(text, _tokenizer, _model):
    # Validasi objek sebelum digunakan
    if _tokenizer is None or not hasattr(_tokenizer, 'encode_plus'):
        return "Error: Tokenizer Rusak", 0.0
    
    try:
        # 1. Preprocessing
        text = clean_text(text)
        text = normalize_text(text, KBBA_MAP)
        
        # 2. Encoding
        encoded = _tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128, # Sesuaikan dengan saat training skripsi
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        # 3. Predict
        with torch.no_grad():
            output = _model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"]
            )
            probs = torch.softmax(output.logits, dim=1)[0].numpy()
        
        idx = np.argmax(probs)
        return LABEL_MAP[idx], probs[idx]
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# =========================================================
# 4. ANTARMUKA STREAMLIT
# =========================================================
st.set_page_config(page_title="Klasifikasi Intent KAI", layout="wide")
st.title("🚂 Klasifikasi Intent Ulasan Access by KAI")

# Pastikan aplikasi berhenti jika model tidak ada
if tokenizer is None or model is None:
    st.warning("⚠️ Aplikasi tidak dapat berjalan karena model gagal dimuat.")
    st.stop()

mode = st.sidebar.radio("Menu", ["Ulasan Tunggal", "Analisis Batch"])

if mode == "Ulasan Tunggal":
    review_input = st.text_area("Input Ulasan")
    if st.button("Analisis"):
        res, score = classify_review(review_input, tokenizer, model)
        st.write(f"**Hasil:** {res} ({score:.2%})")

else:
    # --- BAGIAN BATCH ---
    st.info("Gunakan tombol 'Proses' untuk mengambil data dari Play Store")
    start_date = st.date_input("Mulai", date.today() - timedelta(days=7))
    end_date = st.date_input("Selesai", date.today())
    
    if st.button("🚀 Ambil & Proses Data"):
        with st.spinner("Scraping & Classifying..."):
            days = (end_date - start_date).days + 1
            df = get_reviews_by_date_range(end_date, days)
            
            if not df.empty:
                results = [classify_review(t, tokenizer, model) for t in df["Ulasan"]]
                df["Intent Prediksi"] = [r[0] for r in results]
                df["Confidence Score"] = [r[1] for r in results]
                st.session_state['df_kai'] = df
            else:
                st.error("Data tidak ditemukan.")

    if 'df_kai' in st.session_state:
        st.dataframe(st.session_state['df_kai'])