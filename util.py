import pandas as pd
from datetime import datetime, timedelta, date
import re
from google_play_scraper import Sort, reviews
import streamlit as st 

# --- KONFIGURASI APLIKASI ---
APP_ID = 'com.kai.kaiticketing' 

# ====================================================================
# --- 1. FUNGSI TEXT PREPROCESSING ---
# ====================================================================

def load_kbba_dict(file_path):
    """Memuat kamus KBBA (Kata Baku Bukan Baku) dari file teks."""
    kbba_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Memecah berdasarkan karakter TAB (\t)
                    parts = line.split('\t', 1) 
                    if len(parts) == 2:
                        slang = parts[0].strip().lower()
                        baku = parts[1].strip().lower()
                        kbba_dict[slang] = baku
        return kbba_dict
    except FileNotFoundError:
        st.error(f"❌ File KBBA tidak ditemukan di: {file_path}")
        return None
    except Exception as e:
        st.error(f"❌ Gagal memuat KBBA: {e}")
        return None

def clean_text(text):
    """Membersihkan teks dari URL, karakter khusus, dan spasi ganda."""
    if not isinstance(text, str):
        return ""
    # 1. Menghapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 2. Menghapus karakter khusus (hanya biarkan huruf, angka, dan spasi)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # 3. Menghapus spasi ganda
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text, kbba_map):
    """Melakukan Case Folding dan Slang Replacement."""
    if kbba_map is None:
        return text.lower()
    
    text = text.lower() 
    words = text.split()
    # Mengganti kata jika ada di kamus, jika tidak tetap kata asli
    normalized_words = [kbba_map.get(word, word) for word in words]
    return " ".join(normalized_words)


# ====================================================================
# --- 2. FUNGSI PENGAMBILAN DATA (SCRAPING AKURAT) ---
# ====================================================================

def get_reviews_by_date_range(end_date, days):
    """
    Mengambil ulasan menggunakan teknik Looping & Token agar akurat 
    seperti di Play Store, lalu difilter berdasarkan tanggal.
    """
    all_reviews = []
    continuation_token = None
    
    # Hitung tanggal mulai
    start_date = end_date - timedelta(days=days - 1)
    
    # Konversi ke datetime untuk pembandingan yang akurat
    start_datetime = datetime.combine(start_date, datetime.min.time())
    
    # Batas maksimal pengambilan (biar tidak looping selamanya)
    MAX_LIMIT = 5000 
    
    try:
        while len(all_reviews) < MAX_LIMIT:
            # Ambil ulasan per batch (200 ulasan)
            result, continuation_token = reviews(
                APP_ID,
                lang='id', 
                country='id',
                sort=Sort.NEWEST, # Sesuai dengan urutan terbaru di Play Store
                count=200, 
                continuation_token=continuation_token
            )
            
            if not result:
                break
                
            all_reviews.extend(result)
            
            # Cek tanggal ulasan terakhir di batch ini
            last_review_in_batch = result[-1]['at']
            
            # Jika ulasan terakhir dalam batch sudah lebih lama dari start_date, berhenti scraping
            if last_review_in_batch.date() < start_date:
                break
                
            # Jika tidak ada token lagi, berhenti
            if continuation_token is None:
                break
        
        if not all_reviews:
            return pd.DataFrame()

        # Konversi ke DataFrame
        df = pd.DataFrame(all_reviews)
        
        # Rename kolom agar sesuai dengan app.py
        df = df.rename(columns={'content': 'Ulasan', 'at': 'Tanggal'})
        
        # Konversi kolom Tanggal ke format date (tanpa jam) untuk filter
        df['Tanggal_Filter'] = pd.to_datetime(df['Tanggal']).dt.date
        
        # Filter berdasarkan rentang tanggal yang dipilih user
        mask = (df['Tanggal_Filter'] >= start_date) & (df['Tanggal_Filter'] <= end_date)
        df_filtered = df.loc[mask].copy()
        
        # Urutkan berdasarkan yang terbaru
        df_filtered = df_filtered.sort_values(by='Tanggal', ascending=False)
        
        # Kembalikan hanya kolom yang diperlukan untuk UI
        return df_filtered[['Tanggal', 'Ulasan']].reset_index(drop=True)

    except Exception as e:
        st.error(f"❌ Gagal melakukan scraping data: {e}")
        return pd.DataFrame()