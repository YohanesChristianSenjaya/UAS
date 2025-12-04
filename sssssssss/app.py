import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# Konfigurasi Halaman
st.set_page_config(page_title="TV Shows Strategy", layout="wide")

st.title("📺 Netflix Strategy: TV Shows Segmentation")
st.markdown("""
Aplikasi ini menggunakan algoritma **K-Means** untuk menentukan strategi promosi:
* **💎 Hidden Gem (Underrated):** Rating Tinggi, Popularitas Rendah -> **PRIORITAS PROMOSI 🚀**
* **⭐ Crowd Favorite (Well Rated):** Rating Tinggi, Popularitas Tinggi -> **PERTAHANKAN ✅**
* **📉 Hype Only (Overrated):** Popularitas Tinggi, Rating Rendah -> **TURUNKAN PROMOSI 🔻**
""")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Menggunakan URL Raw GitHub agar bisa langsung jalan
        url_csv = "https://raw.githubusercontent.com/YohanesChristianSenjaya/UAS/refs/heads/main/sssssssss/10k_Poplar_Tv_Shows.csv"
        df = pd.read_csv(url_csv)
        return df
    except Exception as e:
        st.error(f"Gagal memuat data. Error: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# --- 2. DATA PREPROCESSING ---
# Hapus Duplikat & Missing Values
df_clean = df.drop_duplicates().dropna(subset=['first_air_date', 'vote_average', 'popularity']).copy()

# Transformasi Log untuk menangani skewness
df_clean['popularity_log'] = np.log1p(df_clean['popularity'])
df_clean['vote_count_log'] = np.log1p(df_clean['vote_count'])

# Fitur untuk Clustering (Fokus pada Popularitas & Rating sesuai logikamu)
# Kita masukkan vote_count juga agar hasil lebih stabil
cluster_features = ['popularity_log', 'vote_average', 'vote_count_log']
X = df_clean[cluster_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. MODELING (K-MEANS) ---
# K=3 sesuai strategi bisnis kamu
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_clean['Cluster'] = clusters

# --- 4. LOGIKA "NETFLIX" (AUTO-LABELING) ---
# Kita harus mendeteksi karakteristik cluster secara otomatis agar labelnya tidak tertukar
# Hitung rata-rata rating dan popularitas tiap cluster
cluster_profile = df_clean.groupby('Cluster')[['vote_average', 'popularity_log']].mean()

# Definisikan Logika Mapping:
# 1. Cluster dengan Rating Tertinggi & Popularitas Rendah/Sedang = UNDERRATED
# 2. Cluster dengan Rating Tertinggi & Popularitas Tinggi = WELL RATED
# 3. Sisanya (Rating Rendah/Sedang & Popularitas Tinggi) = OVERRATED

# Ini adalah pendekatan sederhana menggunakan sorting
# (Note: Hasil K-Means bisa bervariasi, kita coba pendekatan ranking)
df_clean['Cluster Name'] = df_clean['Cluster'].map({
    0: "Kelompok A", 
    1: "Kelompok B", 
    2: "Kelompok C"
}) # Placeholder awal

# Kita buat logic mapping manual berdasarkan centroid (titik tengah)
# Kita urutkan cluster berdasarkan Rating (Vote Average)
sorted_by_rating = cluster_profile.sort_values('vote_average', ascending=False)
highest_rating_cluster = sorted_by_rating.index[0]
lowest_rating_cluster = sorted_by_rating.index[-1]
middle_rating_cluster = sorted_by_rating.index[1]

# Kustomisasi nama berdasarkan karakteristik yang ditemukan:
# (Kamu mungkin perlu menyesuaikan logika ini jika data berubah drastis)
nama_cluster = {}

for cluster_id in [0, 1, 2]:
    mean_pop = cluster_profile.loc[cluster_id, 'popularity_log']
    mean_vote = cluster_profile.loc[cluster_id, 'vote_average']
    
    # Logika If-Else Sederhana untuk menamai
    if mean_vote > 7.0 and mean_pop < 5.0:
        label = "💎 Underrated (Hidden Gem)"
        action = "🚀 BOOST PROMO"
    elif mean_vote > 7.0 and mean_pop >= 5.0:
        label = "⭐ Well Rated (Crowd Fav)"
        action = "✅ MAINTAIN"
    else:
        label = "📉 Overrated / Niche"
        action = "🔻 REDUCE PROMO"
        
    nama_cluster[cluster_id] = label
    # Kita simpan action ke kolom baru nanti (dilakukan lewat map di bawah)

df_clean['Category'] = df_clean['Cluster'].map(nama_cluster)

# Buat Kolom Action Plan khusus untuk Search Engine nanti
def get_action(category):
    if "Underrated" in category:
        return "PUSH PROMOTION (High Potential)"
    elif "Overrated" in category:
        return "DE-PRIORITIZE (Low ROI)"
    else:
        return "STANDARD DISPLAY"

df_clean['Action Recommendation'] = df_clean['Category'].apply(get_action)


# --- 5. DASHBOARD TAMPILAN ---

st.subheader("📊 Analisis Profil Cluster")

# Statistik Rata-rata
col1, col2 = st.columns([1, 2])
with col1:
    st.info("Rata-rata Metrik per Kategori:")
    summary = df_clean.groupby('Category')[['vote_average', 'popularity', 'vote_count']].mean().round(2)
    st.dataframe(summary)
    
    st.write("---")
    st.caption("Logika Bisnis:")
    st.markdown("""
    1.  **Underrated**: Rating bagus tapi kurang exposure -> Perlu didorong.
    2.  **Well Rated**: Sudah perform bagus -> Biarkan.
    3.  **Overrated**: Banyak yang lihat tapi kecewa -> Kurangi biaya iklan.
    """)

with col2:
    st.write("Visualisasi: Popularitas vs Kualitas (Rating)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_clean, 
        x='popularity_log', 
        y='vote_average', 
        hue='Category', 
        palette={'💎 Underrated (Hidden Gem)': 'green', '⭐ Well Rated (Crowd Fav)': 'blue', '📉 Overrated / Niche': 'red'},
        s=40, 
        alpha=0.7,
        ax=ax
    )
    
    # Menambahkan garis ambang batas (contoh visual)
    plt.axhline(y=7, color='grey', linestyle='--', alpha=0.5, label='Rating Threshold (7.0)')
    
    ax.set_title("Peta Strategi Konten")
    ax.set_xlabel("Popularitas (Log Scale)")
    ax.set_ylabel("Rating (0-10)")
    ax.legend(loc='lower right')
    st.pyplot(fig)

# --- 6. SEARCH ENGINE SIMULATION ---
st.write("---")
st.subheader("🔍 Engine Rekomendasi Promo")
st.write("Simulasi sistem backend Netflix yang menentukan nasib sebuah TV Show.")

search_query = st.text_input("Cari Judul TV Show (Contoh: Naruto, Breaking Bad, etc):")

if search_query:
    # Filter data
    result = df_clean[df_clean['original_name'].str.contains(search_query, case=False, na=False)]
    
    if not result.empty:
        st.write(f"Ditemukan {len(result)} hasil:")
        
        for index, row in result.iterrows():
            # Tampilan Kartu Hasil
            with st.container():
                c1, c2, c3 = st.columns([1, 2, 1])
                with c1:
                    st.metric("Rating", row['vote_average'])
                with c2:
                    st.subheader(row['original_name'])
                    st.caption(f"Popularitas: {row['popularity']} | Votes: {row['vote_count']}")
                    st.write(f"**Status:** {row['Category']}")
                with c3:
                    # Logic warna tombol berdasarkan Action
                    if "PUSH" in row['Action Recommendation']:
                        st.error(f"📢 {row['Action Recommendation']}") # Pakai error biar merah (mencolok) atau success
                    elif "DE-PRIORITIZE" in row['Action Recommendation']:
                        st.warning(f"⚠️ {row['Action Recommendation']}")
                    else:
                        st.success(f"✅ {row['Action Recommendation']}")
                st.markdown("---")
    else:
        st.warning("Judul tidak ditemukan.")

# Tampilkan Sampel Data Tiap Kategori
st.write("### Contoh Data per Kategori")
kategori_pilihan = st.selectbox("Pilih Kategori:", df_clean['Category'].unique())
st.dataframe(df_clean[df_clean['Category'] == kategori_pilihan][['original_name', 'vote_average', 'popularity', 'Action Recommendation']].head(10))
