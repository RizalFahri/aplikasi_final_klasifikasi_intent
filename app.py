import streamlit as st
import pandas as pd
from datetime import date, timedelta
import torch
import numpy as np
import plotly.express as px
import os
import subprocess
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

from util import (
    get_reviews_by_date_range,
    clean_text,
    normalize_text,
    load_kbba_dict
)

# =========================================================
# 1. KONFIGURASI PATH (Path Relatif untuk GitHub)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_kai")
KBBA_PATH = os.path.join(BASE_DIR, "kbba.txt")

# Fungsi untuk menarik file asli dari LFS (Hanya berjalan di server Cloud)
def ensure_lfs_files():
    if os.path.exists(MODEL_PATH):
        file_check = os.path.join(MODEL_PATH, "model.safetensors")
        # Jika file cuma 1KB, berarti itu cuma pointer LFS, perlu di-pull
        if os.path.exists(file_check) and os.path.getsize(file_check) < 10000:
            try:
                subprocess.run(["git", "lfs", "pull"], check=True)
            except:
                pass

ensure_lfs_files()

# =========================================================
# 3. FUNGSI UTILITY & NLP
# =========================================================
# Gunakan try-except agar tidak crash jika file kbba.txt hilang
try:
    KBBA_MAP = load_kbba_dict(KBBA_PATH)
except:
    KBBA_MAP = {}

@st.cache_resource
def load_model_and_tokenizer(path):
    try:
        # Cek apakah folder ada
        if not os.path.exists(path):
            return None, None
            
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        return tokenizer, model
    except Exception as e:
        # Menampilkan error asli di Streamlit untuk debug
        st.sidebar.error(f"Error Loading Model: {e}")
        return None, None

# Pemanggilan Model
tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

# PROTEKSI: Jika model/tokenizer gagal muat, hentikan aplikasi di sini
if tokenizer is None or model is None:
    st.error("❌ Gagal memuat Model atau Tokenizer.")
    st.warning("Pastikan folder 'model_kai' sudah terupload lengkap ke GitHub dan file model.safetensors bukan merupakan pointer LFS.")
    st.info("Saran: Gunakan file 'packages.txt' berisi 'git-lfs' di repository Anda.")
    st.stop() # Menghentikan eksekusi kode di bawahnya agar tidak muncul AttributeError

def preprocess_text(text, tokenizer, max_length=32):
    text = clean_text(text)
    text = normalize_text(text, KBBA_MAP)
    return tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

def classify_review(text, tokenizer, model):
    encoded = preprocess_text(text, tokenizer)
    with torch.no_grad():
        output = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )
        probs = torch.softmax(output.logits, dim=1)[0].numpy()
    idx = np.argmax(probs)
    return LABEL_MAP[idx], probs[idx]

def generate_wordcloud(text_series):
    text = " ".join(text_series)
    wc = WordCloud(
        width=800, height=450,
        background_color="white",
        collocations=False,
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
            df.to_excel(writer, index=False, sheet_name="Laporan_Analisis")
        return output.getvalue()
    except ImportError:
        return None

# =========================================================
# 4. SIDEBAR PANEL
# =========================================================
st.sidebar.title("⚙️ Panel Kontrol")
mode = st.sidebar.radio("Pilih Mode Analisis", ["📝 Ulasan Tunggal", "📊 Analisis Batch"])
st.sidebar.divider()
st.sidebar.info("Aplikasi ini mengklasifikasikan ulasan ke dalam 4 kategori: Pujian, Keluhan, Saran, dan Laporan Kesalahan.")

# =========================================================
# MODE 1: ULASAN TUNGGAL
# =========================================================
if mode == "📝 Ulasan Tunggal":
    st.subheader("📝 Klasifikasi Ulasan Tunggal")
    review_text = st.text_area("Masukkan teks ulasan", placeholder="Contoh: Aplikasinya sangat membantu buat beli tiket kereta!")

    if st.button("🔍 Klasifikasikan", type="primary"):
        if review_text.strip():
            intent, score = classify_review(review_text, tokenizer, model)
            c1, c2 = st.columns(2)
            c1.metric("Intent Terdeteksi", intent)
            c2.metric("Confidence Score", f"{score:.4f}")
        else:
            st.warning("⚠️ Masukkan teks ulasan terlebih dahulu.")

# =========================================================
# MODE 2: ANALISIS BATCH (REVISI PENCARIAN)
# =========================================================
else:
    st.subheader("📊 Analisis Batch Ulasan")
    
    with st.expander("📅 Filter Rentang Waktu", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Tanggal Mulai", date.today() - timedelta(days=7))
        with col2:
            end_date = st.date_input("Tanggal Akhir", date.today())

    if start_date >= end_date:
        st.error("Tanggal mulai harus lebih kecil dari tanggal akhir.")
        st.stop()

    # Tombol untuk memicu proses scraping & klasifikasi
    if st.button("🚀 Ambil & Proses Data", type="primary"):
        days = (end_date - start_date).days + 1
        
        with st.spinner("🔄 Scraping data dari Google Play..."):
            df_raw = get_reviews_by_date_range(end_date, days)

        if df_raw.empty:
            st.warning("Tidak ada data ulasan ditemukan.")
            st.stop()

        # --- PROSES KLASIFIKASI ---
        pbar = st.progress(0)
        intents, scores = [], []
        
        for i, review in enumerate(df_raw["Ulasan"]):
            intent, score = classify_review(review, tokenizer, model)
            intents.append(intent)
            scores.append(score)
            pbar.progress((i + 1) / len(df_raw))
        
        df_raw["Intent Prediksi"] = intents
        df_raw["Confidence Score"] = scores
        
        # --- SIMPAN KE SESSION STATE ---
        # Ini kunci agar data tidak hilang saat Anda mengetik di kolom pencarian
        st.session_state['df_hasil_kai'] = df_raw
        st.success("✅ Klasifikasi Selesai!")

    # --- TAMPILKAN HASIL JIKA DATA SUDAH TERSEDIA DI MEMORI (SESSION STATE) ---
    if 'df_hasil_kai' in st.session_state:
        df_final = st.session_state['df_hasil_kai']

        # --- VISUALISASI BERDAMPINGAN ---
        st.divider()
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            st.markdown("#### ☁️ Word Cloud")
            st.pyplot(generate_wordcloud(df_final["Ulasan"]))

        with v_col2:
            st.markdown("#### 🥧 Distribusi Intent")
            intent_count = df_final["Intent Prediksi"].value_counts().reset_index()
            intent_count.columns = ["Intent", "Jumlah"]

            fig = px.pie(
                intent_count, values="Jumlah", names="Intent",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig.update_traces(textinfo="label+value+percent", textposition="outside")
            fig.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="batch_pie_kai")

        # --- FITUR PENCARIAN & FILTER ---
        st.divider()
        st.subheader("🔍 Detail Data & Laporan")
        
        # Field pencarian yang sekarang akan bekerja secara real-time
        search_term = st.text_input("Cari ulasan spesifik (contoh: 'login' atau 'bayar'):")
        
        # Logika pencarian yang aman dari nilai kosong (NaN)
        if search_term:
            df_filtered = df_final[df_final["Ulasan"].str.contains(search_term, case=False, na=False)]
        else:
            df_filtered = df_final

        st.write(f"Menampilkan **{len(df_filtered)}** dari **{len(df_final)}** ulasan.")
        st.dataframe(df_filtered, use_container_width=True)

        # --- DOWNLOAD EXCEL ---
        excel_data = to_excel(df_filtered)
        if excel_data:
            st.download_button(
                label="📥 Unduh Laporan Excel Terfilter",
                data=excel_data,
                file_name=f"Laporan_KAI_Filter_{date.today()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_btn_kai"
            )
