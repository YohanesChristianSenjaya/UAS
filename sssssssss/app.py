import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns from sklearn.preprocessing 
import StandardScaler, LabelEncoder 
from sklearn.cluster 
import KMeans
import os


# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Netflix-Style Algo",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- JUDUL & HEADER ---
st.title("🎬 TV Shows Recommendation Engine")
st.markdown("""
**Konsep Algoritma:**
Aplikasi ini mendeteksi status acara TV untuk strategi promosi:
* **💎 Underrated:** Rating Tinggi, Popularitas Rendah $\\rightarrow$ **BOOST PROMO (Hidden Gem)**
* **🔥 Well Rated:** Rating Tinggi, Popularitas Tinggi $\\rightarrow$ **MAINTAIN (Fan Favorite)**
* **⚠️ Overrated/Low:** Rating Rendah $\\rightarrow$ **REDUCE PROMO (Fix Quality)**
""")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    # Pastikan file csv ada di folder yang sama
    df = pd.read_csv('sssssssss/10k_Poplar_Tv_Shows.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File '10k_Poplar_Tv_Shows.csv' tidak ditemukan. Pastikan file ada di GitHub repository.")
    st.stop()

# --- 2. PREPROCESSING & MODELING ---
# Cleaning
df_clean = df.drop_duplicates().dropna(subset=['first_air_date']).copy()

# Feature Engineering
df_clean['popularity_log'] = np.log1p(df_clean['popularity'])
df_clean['vote_count_log'] = np.log1p(df_clean['vote_count'])

# Scaling
X = df_clean[['popularity_log', 'vote_average', 'vote_count_log']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering (K=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_clean['Cluster'] = clusters

# --- 3. AUTO-MAPPING LOGIC (LOGIKA CERDAS) ---
# Kita harus mencari tahu Cluster 0, 1, 2 itu yang mana secara otomatis
# Caranya: Hitung rata-rata rating & popularitas tiap cluster

cluster_stats = df_clean.groupby('Cluster')[['vote_average', 'popularity_log']].mean()

# 1. Cari Cluster dengan Rating TERENDAH -> Overrated/Low Quality
cluster_overrated = cluster_stats['vote_average'].idxmin()

# 2. Sisa 2 cluster adalah yang ratingnya bagus. Bandingkan Popularitasnya.
remaining_clusters = cluster_stats.drop(cluster_overrated)
cluster_well_rated = remaining_clusters['popularity_log'].idxmax() # Populer & Bagus
cluster_underrated = remaining_clusters['popularity_log'].idxmin() # Kurang Populer & Bagus

# Buat Dictionary Mapping
label_mapping = {
    cluster_well_rated: "Well Rated (Popular)",
    cluster_underrated: "Underrated (Hidden Gem)",
    cluster_overrated: "Overrated / Low Perf"
}

action_mapping = {
    cluster_well_rated: "✅ Pertahankan Promosi (Cash Cow)",
    cluster_underrated: "🚀 BOOST PROMO! (Potensi Viral)",
    cluster_overrated: "🔻 Kurangi Budget Iklan (Evaluasi)"
}

df_clean['Category'] = df_clean['Cluster'].map(label_mapping)
df_clean['Action'] = df_clean['Cluster'].map(action_mapping)

# --- 4. SEARCH ENGINE INTERFACE ---
st.divider()
st.subheader("🔍 Cari Judul Film / TV Show")

# Input Search
search_query = st.text_input("Masukkan kata kunci judul (Contoh: Game of Thrones, Naruto, dll):")

if search_query:
    # Filter Data (Case Insensitive)
    results = df_clean[df_clean['original_name'].str.contains(search_query, case=False, na=False)]
    
    if len(results) > 0:
        st.success(f"Ditemukan {len(results)} hasil pencarian:")
        
        # Loop untuk menampilkan hasil seperti kartu
        for index, row in results.head(5).items(): # Tampilkan max 5 hasil teratas
             # Ambil data baris (karena iterrows agak lambat, kita pakai iloc di loop terpisah jika data besar, tapi untuk 5 ok)
             pass 
        
        # Tampilan Tabel Interaktif
        for i, row in results.iterrows():
            with st.container():
                c1, c2, c3 = st.columns([2, 1, 1])
                
                with c1:
                    st.subheader(f"📺 {row['original_name']}")
                    st.write(f"**Rilis:** {row['first_air_date']} | **Bahasa:** {row['original_language']}")
                    st.write(f"**Overview:** {str(row['overview'])[:150]}...")
                
                with c2:
                    st.metric("Rating", f"{row['vote_average']}/10")
                    st.metric("Popularitas", f"{row['popularity']:.0f}")
                
                with c3:
                    # Logika Warna Badge
                    if row['Cluster'] == cluster_underrated:
                        st.info(f"💎 {row['Category']}")
                        st.markdown(f"**Saran:**\n\n:rocket: {row['Action']}")
                    elif row['Cluster'] == cluster_well_rated:
                        st.success(f"🔥 {row['Category']}")
                        st.markdown(f"**Saran:**\n\n:white_check_mark: {row['Action']}")
                    else:
                        st.warning(f"⚠️ {row['Category']}")
                        st.markdown(f"**Saran:**\n\n:small_red_triangle_down: {row['Action']}")
                
                st.divider()
    else:
        st.warning("Tidak ditemukan judul yang cocok. Coba kata kunci lain.")

# --- 5. STATISTIK GLOBAL ---
st.write("---")
with st.expander("Lihat Analisis Keseluruhan (Dashboard)"):
    st.write("Profil Rata-rata per Kategori:")
    summary = df_clean.groupby('Category')[['vote_average', 'popularity', 'vote_count']].mean()
    st.dataframe(summary)
    
    # Visualisasi
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df_clean, x='popularity_log', y='vote_average', hue='Category', palette='Set1', s=30, ax=ax)
    ax.set_title("Peta Persebaran Kategori")
    st.pyplot(fig)
