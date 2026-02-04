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

# Import fungsi dari util.py Anda
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
# 1. KONFIGURASI MODEL HUGGING FACE
# =========================================================
# Ganti dengan repo Hugging Face Anda
MODEL_NAME = "ree28/klasifikasiulasankai-indobert"
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
st.caption("Analisis Otomatis Berbasis IndoBERT - Versi Deploy Cloud")
st.divider()

# =========================================================
# 3. FUNGSI UTILITY & NLP
# =========================================================

@st.cache_data
def get_kbba():
    try:
        # Mencari file kbba.txt di folder yang sama dengan app.py
        return load_kbba_dict(KBBA_FILE)
    except Exception as e:
        st.warning(f"File {KBBA_FILE} tidak ditemukan, menggunakan kamus kosong.")
        return {}

KBBA_MAP = get_kbba()

@st.cache_resource
def load_model_and_tokenizer(model_path):
    try:
        # Load langsung dari Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Gagal memuat model dari Hugging Face: {e}")
        return None, None

tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

def preprocess_text(text, _tokenizer, max_length=128):
    # Membersihkan dan normalisasi teks menggunakan fungsi dari util.py
    text = clean_text(text)
    text = normalize_text(text, KBBA_MAP)
    
    return _tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

def classify_review(text, _tokenizer, _model):
    if _tokenizer is None or _model is None:
        return "Model Error", 0.0
        
    encoded = preprocess_text(text, _tokenizer)
    with torch.no_grad():
        output = _model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )
        # Ambil probabilitas menggunakan softmax
        probs = torch.softmax(output.logits, dim=1)[0].numpy()
        
    idx = np.argmax(probs)
    return LABEL_MAP[idx], probs[idx]

def generate_wordcloud(text_series):
    text = " ".join(text_series.astype(str))
    wc = WordCloud(
        width=800, height=400,
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
# 4. SIDEBAR & MENU
# =========================================================
st.sidebar.title("⚙️ Panel Kontrol")
mode = st.sidebar.radio("Pilih Mode Analisis", ["📝 Ulasan Tunggal", "📊 Analisis Batch"])

# =========================================================
# MODE 1: ULASAN TUNGGAL
# =========================================================
if mode == "📝 Ulasan Tunggal":
    st.subheader("📝 Klasifikasi Ulasan Tunggal")
    review_input = st.text_area("Masukkan ulasan pelanggan:", placeholder="Tulis di sini...")

    if st.button("🔍 Analisis Intent", type="primary"):
        if review_input.strip():
            intent, score = classify_review(review_input, tokenizer, model)
            col_a, col_b = st.columns(2)
            col_a.metric("Kategori Intent", intent)
            col_b.metric("Confidence Score", f"{score:.4f}")
        else:
            st.warning("Silakan masukkan teks ulasan.")

# =========================================================
# MODE 2: ANALISIS BATCH
# =========================================================
else:
    st.subheader("📊 Analisis Batch (Play Store Scraping)")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Mulai", date.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("Selesai", date.today())

    if st.button("🚀 Proses Data", type="primary"):
        if start_date >= end_date:
            st.error("Rentang tanggal tidak valid.")
        else:
            days = (end_date - start_date).days + 1
            with st.spinner("Mengambil ulasan dari Google Play..."):
                df_raw = get_reviews_by_date_range(end_date, days)

            if not df_raw.empty:
                # Proses Klasifikasi
                intents, scores = [], []
                pbar = st.progress(0)
                
                for i, txt in enumerate(df_raw["Ulasan"]):
                    it, sc = classify_review(txt, tokenizer, model)
                    intents.append(it)
                    scores.append(sc)
                    pbar.progress((i + 1) / len(df_raw))
                
                df_raw["Intent Prediksi"] = intents
                df_raw["Confidence Score"] = scores
                st.session_state['data_kai'] = df_raw
                st.success(f"Berhasil memproses {len(df_raw)} ulasan!")
            else:
                st.warning("Tidak ada ulasan ditemukan pada rentang tersebut.")

    # Tampilkan Hasil Jika Ada
    if 'data_kai' in st.session_state:
        df_res = st.session_state['data_kai']
        
        st.divider()
        c1, c2 = st.columns([1, 1])
        with c1:
            st.pyplot(generate_wordcloud(df_res["Ulasan"]))
        with c2:
            counts = df_res["Intent Prediksi"].value_counts().reset_index()
            fig_pie = px.pie(counts, values="count", names="Intent Prediksi", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()
        search = st.text_input("Cari kata kunci dalam ulasan:")
        df_show = df_res[df_res["Ulasan"].str.contains(search, case=False)] if search else df_res
        
        st.dataframe(df_show, use_container_width=True)
        
        excel_file = to_excel(df_show)
        if excel_file:
            st.download_button("📥 Download Excel", data=excel_file, file_name="hasil_analisis.xlsx")