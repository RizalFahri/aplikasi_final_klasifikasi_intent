# util.py (Versi Final dengan APP_ID yang Benar)

import pandas as pd
from datetime import datetime, timedelta, date
from google_play_scraper import Sort, reviews
import streamlit as st 

# ID Aplikasi Access by KAI yang BENAR
APP_ID = 'com.kai.kaiticketing' 

# --- FUNGSI PENGAMBILAN DATA ULASAN RIIL DARI GOOGLE PLAY STORE ---
def get_reviews_by_date_range(end_date, days):
    """
    Mengambil ulasan riil dari Google Play Store dan memfilternya berdasarkan rentang tanggal.
    """
    MAX_RESULTS = 5000  # Batasi jumlah ulasan yang di-scrape
    
    # Hitung tanggal mulai
    start_date = end_date - timedelta(days=days - 1)
    
    try:
        # Panggil fungsi reviews
        # Mengambil ulasan terbaru sebanyak MAX_RESULTS
        result, continuation_token = reviews(
            APP_ID,
            lang='id', 
            country='id',
            sort=Sort.NEWEST,
            count=MAX_RESULTS, 
            filter_score_with=None 
        )
        
        # Cek jika hasil scraping kosong
        if not result:
            st.warning("⚠️ Hasil scraping kosong. Mungkin tidak ada ulasan dalam bahasa Indonesia terbaru, atau APP_ID salah.")
            return pd.DataFrame()
            
        # Konversi ke DataFrame
        df = pd.DataFrame(result)
        
        # --- RENAME KOLOM ---
        # Kolom harus ada setelah scraping berhasil
        if 'content' not in df.columns or 'at' not in df.columns:
            st.error("❌ Terjadi masalah pada penamaan kolom data. (content/at tidak ditemukan)")
            return pd.DataFrame()
            
        df = df.rename(columns={'content': 'Ulasan', 'at': 'Tanggal'})
        
        # --- FILTERING TANGGAL ---
        
        # Konversi kolom tanggal ke format date (datetime.date)
        df['Tanggal'] = pd.to_datetime(df['Tanggal']).dt.date
        
        # Filter berdasarkan rentang tanggal yang diminta dari Streamlit
        # start_date dan end_date adalah objek date
        df_filtered = df[(df['Tanggal'] >= start_date) & (df['Tanggal'] <= end_date)].reset_index(drop=True)
        
        return df_filtered[['Tanggal', 'Ulasan']] # Kembalikan hanya kolom yang diperlukan
    
    except Exception as e:
        # Menampilkan pesan error scraping di Streamlit
        st.error(f"❌ Gagal melakukan scraping data. Error: {e}")
        return pd.DataFrame()
