import streamlit as st
import pandas as pd
from datetime import date, timedelta
import torch
import numpy as np
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

# Import fungsi dari util.py Anda di GitHub
try:
    from util import (
        get_reviews_by_date_range,
        clean_text,
        normalize_text,
        load_kbba_dict
    )
except ImportError:
    st.error("File util.py tidak ditemukan. Pastikan util.py ada di repository GitHub Anda.")

# =========================================================
# 1. KONFIGURASI MODEL & PATH
# =========================================================
# Merujuk ke repository Hugging Face Anda
MODEL_NAME = "ree28/klasifikasiulasankai-indobert"
# Merujuk ke file di root repository GitHub Anda
KBBA_FILE = "kbba.txt" 

LABEL_MAP = {
    0: "Pujian",
    1: "Keluhan",
    2: "Saran",
    3: "Laporan Kesalahan"
}

# =========================================================
# 2. KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Klasifikasi Intent Access by KAI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚂 SISTEM KLASIFIKASI INTENT ULASAN APLIKASI ACCESS BY KAI")
st.caption("Analisis Otomatis Berbasis IndoBERT - Implementasi Tugas Akhir")
st.divider()

# =========================================================
# 3. FUNGSI LOAD DATA & MODEL (DENGAN PROTEKSI)
# =========================================================

@st.cache_resource
def load_essentials():
    try:
        # Gunakan use_fast=False untuk stabilitas model IndoBERT
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        
        # Tambahkan use_safetensors=True sesuai format file di HF Anda
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            use_safetensors=True
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Gagal memuat model/tokenizer: {e}")
        return None, None

@st.cache_data
def load_dictionary():
    try:
        return load_kbba_dict(KBBA_FILE)
    except Exception:
        return {}

# Inisialisasi Objek
tokenizer, model = load_essentials()
KBBA_MAP = load_dictionary()

# =========================================================
# 4. FUNGSI PREDIKSI & VISUALISASI
# =========================================================

def classify_review(text, _tokenizer, _model):
    # Proteksi kritis agar tidak terjadi AttributeError
    if _tokenizer is None or not hasattr(_tokenizer, 'encode_plus'):
        return "Error: Tokenizer Tidak Siap", 0.0
    
    try:
        # Preprocessing menggunakan util.py
        text = clean_text(str(text))
        text = normalize_text(text, KBBA_MAP)
        
        # Encoding
        encoded = _tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        # Predict
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

def generate_wordcloud(text_series):
    text = " ".join(text_series.astype(str))
    wc = WordCloud(
        width=800, height=450,
        background_color="white",
        colormap='tab10'
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def to_excel(df):
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Laporan")
        return output.getvalue()
    except Exception:
        return None

# =========================================================
# 5. ANTARMUKA PENGGUNA (UI)
# =========================================================
st.sidebar.title("⚙️ Panel Kontrol")
mode = st.sidebar.radio("Pilih Mode Analisis", ["📝 Ulasan Tunggal", "📊 Analisis Batch"])

if tokenizer is None or model is None:
    st.warning("⚠️ Aplikasi tidak dapat dijalankan karena model gagal dimuat.")
    st.stop()

# --- MODE 1: ULASAN TUNGGAL ---
if mode == "📝 Ulasan Tunggal":
    st.subheader("📝 Klasifikasi Ulasan Tunggal")
    review_text = st.text_area("Masukkan teks ulasan", placeholder="Contoh: Aplikasinya lemot saat bayar tiket.")

    if st.button("🔍 Analisis", type="primary"):
        if review_text.strip():
            intent, score = classify_review(review_text, tokenizer, model)
            c1, c2 = st.columns(2)
            c1.metric("Intent Terdeteksi", intent)
            c2.metric("Confidence Score", f"{score:.4f}")
        else:
            st.warning("Silakan isi teks ulasan.")

# --- MODE 2: ANALISIS BATCH ---
else:
    st.subheader("📊 Analisis Batch dari Google Play")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Tanggal Mulai", date.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("Tanggal Akhir", date.today())

    if st.button("🚀 Proses Data", type="primary"):
        if start_date >= end_date:
            st.error("Rentang tanggal tidak valid.")
        else:
            days = (end_date - start_date).days + 1
            with st.spinner("Scraping & Klasifikasi sedang berjalan..."):
                df_raw = get_reviews_by_date_range(end_date, days)

            if not df_raw.empty:
                # Proses Klasifikasi
                intents, scores = [], []
                pbar = st.progress(0)
                for i, row in enumerate(df_raw["Ulasan"]):
                    it, sc = classify_review(row, tokenizer, model)
                    intents.append(it)
                    scores.append(sc)
                    pbar.progress((i + 1) / len(df_raw))
                
                df_raw["Intent Prediksi"] = intents
                df_raw["Confidence Score"] = scores
                st.session_state['df_kai'] = df_raw
            else:
                st.warning("Ulasan tidak ditemukan.")

    if 'df_kai' in st.session_state:
        df_final = st.session_state['df_kai']
        
        # Visualisasi
        st.divider()
        v1, v2 = st.columns(2)
        with v1:
            st.pyplot(generate_wordcloud(df_final["Ulasan"]))
        with v2:
            intent_count = df_final["Intent Prediksi"].value_counts().reset_index()
            fig = px.pie(intent_count, values="count", names="Intent Prediksi", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        # Tabel & Download
        st.divider()
        search = st.text_input("Cari kata kunci dalam tabel:")
        df_filtered = df_final[df_final["Ulasan"].str.contains(search, case=False)] if search else df_final
        
        st.dataframe(df_filtered, use_container_width=True)
        
        excel_data = to_excel(df_filtered)
        if excel_data:
            st.download_button("📥 Unduh Excel", data=excel_data, file_name="hasil_analisis_kai.xlsx")