import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# ... (import library lainnya di atas)
import os  # <--- Tambahkan ini untuk deteksi lokasi file

# ... (kode konfigurasi halaman)

# --- 1. LOAD DATA (ANTI-NYASAR) ---
@st.cache_data
def load_data():
    try:
        # 1. Cari tahu lokasi file app.py ini ada di mana
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Gabungkan lokasi tersebut dengan nama file CSV
        # Hasilnya akan jadi path lengkap: /mount/src/uas/sssssssss/10k_Poplar_Tv_Shows.csv
        file_path = os.path.join(current_dir, '10k_Poplar_Tv_Shows.csv')
        
        # 3. Baca file dari path lengkap
        df = pd.read_csv(file_path)
        return df
        
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    # Tampilkan error debugging biar ketahuan dia nyari di mana
    import os
    st.error(f"❌ File tidak ketemu! Sistem mencari di: {os.path.join(os.path.dirname(os.path.abspath(__file__)), '10k_Poplar_Tv_Shows.csv')}")
    st.write("Cek apakah nama file di GitHub huruf besar/kecilnya SAMA PERSIS?")
    st.stop()

# ... (lanjutan kode lainnya ke bawah)


# --- 1. LOAD DATA (METODE RAW URL) ---
@st.cache_data
def load_data():
    try:
        # GANTI URL DI BAWAH INI dengan URL Raw yang Anda copy dari GitHub
        url_csv = "https://raw.githubusercontent.com/YohanesChristianSenjaya/UAS/refs/heads/main/sssssssss/10k_Poplar_Tv_Shows.csv"
        
        df = pd.read_csv(url_csv)
        return df
        
    except Exception as e:
        st.error(f"Gagal memuat data dari URL. Error: {e}")
        return None

# Konfigurasi Halaman
st.set_page_config(page_title="TV Shows Segmentation", layout="wide")

st.title("📺 Analisis Segmentasi TV Shows")
st.write("Aplikasi ini mengelompokkan acara TV berdasarkan Popularitas dan Rating menggunakan K-Means Clustering.")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    # Pastikan file csv ada di folder yang sama
    df = pd.read_csv('10k_Poplar_Tv_Shows.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File '10k_Poplar_Tv_Shows.csv' tidak ditemukan. Pastikan file ada di GitHub repository.")
    st.stop()

# --- 2. DATA PREPROCESSING ---
# Hapus Duplikat & Missing Values
df_clean = df.drop_duplicates().dropna(subset=['first_air_date']).copy()

# Transformasi Log
df_clean['popularity_log'] = np.log1p(df_clean['popularity'])
df_clean['vote_count_log'] = np.log1p(df_clean['vote_count'])

# Encoding (Persiapan Fitur)
le = LabelEncoder()
df_analysis = df_clean.copy()
# Kita hanya butuh fitur numerik untuk clustering ini, tapi encoding tetap kita siapkan
df_analysis['lang_code'] = le.fit_transform(df_analysis['original_language'])
df_analysis['country_code'] = le.fit_transform(df_analysis['origin_country'].astype(str))
df_analysis['genre_code'] = le.fit_transform(df_analysis['genre_ids'].astype(str))

# Scaling
cluster_features = ['popularity_log', 'vote_average', 'vote_count_log']
X = df_analysis[cluster_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. MODELING (K-MEANS) ---
# Menggunakan K=3 sesuai hasil Elbow Method di notebook
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Gabungkan hasil ke dataframe utama
df_clean['Cluster'] = clusters

# Mapping nama cluster agar lebih mudah dibaca
cluster_mapping = {
    0: "Mainstream (Standard)",
    1: "Blockbuster (Hits)",
    2: "Niche (Low Perform)"
}
df_clean['Cluster Name'] = df_clean['Cluster'].map(cluster_mapping)

# --- 4. DASHBOARD TAMPILAN ---

# Tampilkan Statistik Cluster
st.subheader("📊 Profil Setiap Cluster")
col1, col2 = st.columns([1, 2])

with col1:
    st.write("Rata-rata Metrik per Cluster:")
    # Grouping untuk melihat karakteristik
    cluster_stats = df_clean.groupby('Cluster Name')[['popularity', 'vote_average', 'vote_count']].mean().reset_index()
    st.dataframe(cluster_stats)

    st.write("Jumlah Data per Cluster:")
    st.bar_chart(df_clean['Cluster Name'].value_counts())

with col2:
    # Visualisasi Scatter Plot
    st.write("Visualisasi Sebaran: Popularitas vs Rating")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_clean, 
        x='popularity_log', 
        y='vote_average', 
        hue='Cluster Name', 
        palette='viridis', 
        s=20, 
        alpha=0.6,
        ax=ax
    )
    ax.set_title("Segmentasi: Log Popularity vs Vote Average")
    ax.set_xlabel("Log Popularity")
    ax.set_ylabel("Rating (Vote Average)")
    st.pyplot(fig)

# --- 5. DATA EXPLORER ---
st.subheader("🔍 Cari Judul Acara TV")
selected_cluster = st.selectbox("Pilih Kategori Cluster:", df_clean['Cluster Name'].unique())

filtered_data = df_clean[df_clean['Cluster Name'] == selected_cluster]
st.write(f"Menampilkan 10 data teratas dari kategori **{selected_cluster}**:")
st.dataframe(filtered_data[['original_name', 'popularity', 'vote_average', 'vote_count', 'first_air_date']].head(10))

# Tampilkan Raw Data (Opsional)
if st.checkbox("Tampilkan semua data mentah"):
    st.dataframe(df_clean)
