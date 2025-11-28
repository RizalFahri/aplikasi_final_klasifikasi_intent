# app.py

import streamlit as st
import pandas as pd
from datetime import date, timedelta
import torch
import numpy as np
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from util import get_reviews_by_date_range # Import fungsi scraping riil

# --- KONFIGURASI PATH DAN LABEL ---
# GANTI DENGAN PATH ASLI MODEL ANDA (Gunakan r"" untuk path Windows)
MODEL_PATH = "model_kai"

# Label mapping sesuai dengan 4 kategori intent penelitian
LABEL_MAP = {0: 'Pujian', 1: 'Keluhan', 2: 'Saran', 3: 'Laporan Kesalahan'} 

# --- FUNGSI MEMUAT MODEL DAN TOKENIZER (DENGAN CACHING) ---
@st.cache_resource 
def load_model_and_tokenizer(model_path):
    """Memuat konfigurasi, model, dan tokenizer IndoBERT yang telah dilatih."""
    try:
        # Memuat Tokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # MEMUAT MODEL SECARA EKSPLISIT DENGAN parameter 'local_files_only=True' 
        # dan memastikan ia mencari di folder lokal.
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True 
        )
        model.eval() 
        return tokenizer, model
    except Exception as e:
        # PENTING: Jika error tetap terjadi setelah menambahkan safetensors, 
        # kemungkinan besar ada masalah pada file 'model.safetensors' itu sendiri.
        st.error(f"❌ Gagal memuat model/tokenizer. Pastikan path '{model_path}' benar dan semua file ada. Error: {e}")
        return None, None

tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

# --- FUNGSI PREPROCESSING DAN KLASIFIKASI ULASAN TUNGGAL (Inferensi) ---
def preprocess_text(text, tokenizer, max_length=32):
    """Melakukan Preprocessing (Tokenizing, Padding, Masking) untuk IndoBERT."""
    text = text.lower() # Case Folding
    # Note: Text Cleaning dan Normalisasi (jika ada) dilakukan sebelum ini
    
    # Tokenizing, Konversi ke ID Numerik, Padding, dan Attention Mask
    encoded = tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt' # Return PyTorch tensors
    )
    return encoded

def classify_review(text, tokenizer, model):
    """Melakukan prediksi intent menggunakan model IndoBERT."""
    encoded_input = preprocess_text(text, tokenizer)
    
    # Inferensi (Prediksi)
    with torch.no_grad():
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        probabilities = torch.softmax(logits, dim=1)[0].numpy()
        predicted_class_id = np.argmax(probabilities)
        
    intent = LABEL_MAP[predicted_class_id]
    confidence_score = probabilities[predicted_class_id]
    
    return intent, confidence_score

# ====================================================================
# --- APLIKASI STREAMLIT (UI/UX) ---
# ====================================================================

st.set_page_config(page_title="Klasifikasi Intent Ulasan Access by KAI", layout="wide")
st.title("🚂 Sistem Klasifikasi Intent Ulasan Access by KAI")
st.markdown("Aplikasi berbasis **Streamlit** untuk klasifikasi otomatis ulasan pengguna menggunakan model **IndoBERT**.")
st.markdown("---")

if model and tokenizer:
    
    # Sidebar untuk pemilihan mode
    st.sidebar.header("Mode Analisis")
    selection_type = st.sidebar.radio(
        "Pilih Sumber Ulasan:",
        ('Input Ulasan Tunggal', 'Analisis Batch (Rentang Waktu)')
    )

    if selection_type == 'Input Ulasan Tunggal':
        
        st.header("1. Klasifikasi Ulasan Tunggal (Real-Time)")
        st.markdown("Masukkan satu ulasan untuk diproses secara langsung oleh model IndoBERT.")
        
        review_input = st.text_area("Masukkan teks ulasan di sini:", "")
        
        if st.button("Klasifikasikan Intent", key='single_classify_btn', type="primary"):
            if review_input:
                intent, score = classify_review(review_input, tokenizer, model)
                
                st.success(f"✅ **Intent Terdeteksi:** **{intent}**")
                st.info(f"**Tingkat Kepercayaan (Confidence Score):** {score:.4f}")
            else:
                st.warning("⚠️ Masukkan ulasan terlebih dahulu.")
                
    else: # Mode Batch (Analisis Rentang Waktu)
        
        st.header("1. Analisis Batch Ulasan Berdasarkan Rentang Waktu")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date_manual = st.date_input("Tanggal Mulai Pengambilan Data:", date.today() - timedelta(days=7))
        
        with col2:
            end_date_manual = st.date_input("Tanggal Akhir Pengambilan Data:", date.today())
        
        if start_date_manual >= end_date_manual:
            st.error("Tanggal mulai harus sebelum tanggal akhir.")
            st.stop()
        
        days_to_retrieve = (end_date_manual - start_date_manual).days + 1

        if st.button("Tampilkan & Proses Klasifikasi Batch", key='batch_classify_btn', type="primary"):
            
            # --- PENGAMBILAN DATA RIIL ---
            with st.spinner(f"⏳ Mengambil ulasan riil dari Google Play (Maks 500 ulasan terbaru) dan memproses..."):
                df_reviews = get_reviews_by_date_range(end_date_manual, days_to_retrieve)
            
            if not df_reviews.empty:
                st.success(f"🎉 Berhasil mengambil **{len(df_reviews)}** ulasan yang sesuai rentang waktu.")

                # --- PROSES KLASIFIKASI DATA BATCH ---
                st.subheader("2. Menjalankan Klasifikasi IndoBERT pada Data Ulasan")
                
                # Tambahkan kolom prediksi dan confidence score
                # Menggunakan progress bar untuk proses klasifikasi yang mungkin lama
                progress_bar = st.progress(0, text="Mengklasifikasikan ulasan...")
                
                results = []
                for i, review in enumerate(df_reviews['Ulasan']):
                    intent, score = classify_review(review, tokenizer, model)
                    results.append((intent, score))
                    progress_bar.progress((i + 1) / len(df_reviews), text=f"Mengklasifikasikan ulasan... ({i+1}/{len(df_reviews)})")
                
                progress_bar.empty()
                
                df_reviews['Intent Prediksi'] = [r[0] for r in results]
                df_reviews['Confidence Score'] = [r[1] for r in results]
                
                # --- TAMPILAN HASIL ---
                st.markdown("### 3. Data Hasil Ulasan + Label")
                st.dataframe(df_reviews, use_container_width=True)
                
                st.markdown("### 4. Distribusi Intent (Visualisasi)")
                
                intent_counts = df_reviews['Intent Prediksi'].value_counts().reset_index()
                intent_counts.columns = ['Intent', 'Jumlah']
                
                fig = px.pie(
                    intent_counts, 
                    values='Jumlah', 
                    names='Intent', 
                    title=f'Distribusi Intent Ulasan Periode {start_date_manual} hingga {end_date_manual}',
                    color_discrete_sequence=px.colors.qualitative.T10
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("Tidak ada ulasan yang ditemukan atau gagal melakukan scraping. Coba rentang waktu yang lebih pendek atau cek koneksi.")
            
else:
    st.warning("⚠️ Model klasifikasi belum siap. Silakan periksa pesan error di atas dan pastikan folder model sudah diatur dengan benar.")
