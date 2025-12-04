import streamlit as st
import pandas as pd
import numpy as np

# --- PENANGKAL ERROR MATPLOTLIB (WAJIB DI ATAS) ---
import matplotlib
matplotlib.use('Agg') # Memaksa backend server (non-interaktif)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="TV Shows Segmentation",
    layout="wide"
)

# --- JUDUL ---
st.title("📺 TV Shows Segmentation Analytics")
st.write("Aplikasi Clustering K-Means untuk Segmentasi Penonton.")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Pastikan nama file CSV sesuai dengan yang ada di folder Anda
        df = pd.read_csv('10k_Poplar_Tv_Shows.csv')
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("❌ File '10k_Poplar_Tv_Shows.csv' tidak ditemukan di GitHub/Folder.")
    st.stop()

# --- 2. DATA PREPARATION ---
# Cleaning
df_clean = df.drop_duplicates().dropna(subset=['first_air_date']).copy()

# Transformasi Log
df_clean['popularity_log'] = np.log1p(df_clean['popularity'])
df_clean['vote_count_log'] = np.log1p(df_clean['vote_count'])

# Scaling
cluster_features = ['popularity_log', 'vote_average', 'vote_count_log']
X = df_clean[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. MODELING ---
# Kita set K=3 default
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_clean['Cluster'] = clusters

# Mapping Nama (Opsional, sesuaikan dengan hasil notebook Anda)
# Cluster 0, 1, 2 mungkin bertukar posisi tiap run, ini mapping generik:
df_clean['Cluster Label'] = df_clean['Cluster'].astype(str)

# --- 4. VISUALISASI (Fixed Matplotlib) ---
st.divider()
st.subheader("📊 Dashboard Analisis")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("##### 1. Peta Sebaran (Scatter Plot)")
    
    # Membuat Figure & Axes secara eksplisit
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plotting Seaborn ke dalam 'ax1'
    sns.scatterplot(
        data=df_clean,
        x='popularity_log',
        y='vote_average',
        hue='Cluster Label',
        palette='viridis',
        s=50,
        alpha=0.6,
        ax=ax1
    )
    ax1.set_title("Popularitas vs Rating")
    ax1.set_xlabel("Log Popularity")
    ax1.set_ylabel("Vote Average")
    
    # Tampilkan di Streamlit
    st.pyplot(fig1)

with col2:
    st.markdown("##### 2. Distribusi Rating (Box Plot)")
    
    # Membuat Figure & Axes secara eksplisit
    fig2, ax2 = plt.subplots(figsize=(6, 8))
    
    # Plotting ke dalam 'ax2'
    sns.boxplot(
        data=df_clean,
        x='Cluster Label',
        y='vote_average',
        palette='viridis',
        ax=ax2
    )
    ax2.set_title("Sebaran Rating per Cluster")
    
    # Tampilkan di Streamlit
    st.pyplot(fig2)

# --- 5. DATA VIEW ---
st.divider()
st.subheader("📂 Lihat Data")
pilihan = st.selectbox("Pilih Cluster:", sorted(df_clean['Cluster Label'].unique()))

st.write(f"Menampilkan 5 data teratas dari **Cluster {pilihan}**:")
tampil = df_clean[df_clean['Cluster Label'] == pilihan][['original_name', 'popularity', 'vote_average', 'vote_count']]
st.dataframe(tampil.head(5), use_container_width=True)