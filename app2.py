import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

# Memuat model dan encoder yang telah disimpan
model = joblib.load('models/random_forest_model.pkl')
onehot_encoder = joblib.load('models/onehot_encoder.pkl')
ordinal_encoder = joblib.load('models/ordinal_encoder.pkl')

# Mapping untuk input kode ke deskripsi (Bahasa Indonesia)
cap_shape_mapping = {
    'b': 'Bel',
    'c': 'Kerucut',
    'x': 'Cembung',
    'f': 'Datar',
    'k': 'Berkonde',
    's': 'Cekung'
}
cap_surface_mapping = {
    'f': 'Berserat',
    'g': 'Beralur',
    'y': 'Bersisik',
    's': 'Halus'
}
cap_color_mapping = {
    'n': 'Cokelat',
    'b': 'Buff',
    'c': 'Kayu Manis',
    'g': 'Abu-abu',
    'r': 'Hijau',
    'p': 'Merah Muda',
    'u': 'Ungu',
    'e': 'Merah',
    'w': 'Putih',
    'y': 'Kuning'
}
odor_mapping = {
    'a': 'Almond',
    'l': 'Adas Manis',
    'c': 'Kreosot',
    'y': 'Amis',
    'f': 'Busuk',
    'm': 'Pengap',
    'n': 'Tidak Ada',
    'p': 'Menyengat',
    's': 'Pedas'
}
gill_attachment_mapping = {
    'a': 'Terlampir',
    'd': 'Menurun',
    'f': 'Bebas',
    'n': 'Takik'
}
gill_spacing_mapping = {
    'c': 'Dekat',
    'w': 'Padat',
    'd': 'Jarak'
}
gill_size_mapping = {
    'b': 'Lebar',
    'n': 'Sempit'
}
gill_color_mapping = {
    'k': 'Hitam',
    'n': 'Cokelat',
    'b': 'Buff',
    'h': 'Cokelat Tua',
    'g': 'Abu-abu',
    'r': 'Hijau',
    'o': 'Oranye',
    'p': 'Merah Muda',
    'u': 'Ungu',
    'e': 'Merah',
    'w': 'Putih',
    'y': 'Kuning'
}
stalk_shape_mapping = {
    'e': 'Membesar',
    't': 'Meruncing'
}
stalk_root_mapping = {
    'b': 'Bulbous',
    'c': 'Klub',
    'u': 'Cangkir',
    'e': 'Seimbang',
    'z': 'Rhizomorf',
    'r': 'Berakar',
    '?': 'Tidak Diketahui'
}
stalk_surface_above_ring_mapping = {
    'f': 'Berserat',
    'y': 'Bersisik',
    'k': 'Halus',
    's': 'Halus'
}
stalk_surface_below_ring_mapping = {
    'f': 'Berserat',
    'y': 'Bersisik',
    'k': 'Halus',
    's': 'Halus'
}
stalk_color_above_ring_mapping = {
    'n': 'Cokelat',
    'b': 'Buff',
    'c': 'Kayu Manis',
    'g': 'Abu-abu',
    'o': 'Oranye',
    'p': 'Merah Muda',
    'e': 'Merah',
    'w': 'Putih',
    'y': 'Kuning'
}
stalk_color_below_ring_mapping = {
    'n': 'Cokelat',
    'b': 'Buff',
    'c': 'Kayu Manis',
    'g': 'Abu-abu',
    'o': 'Oranye',
    'p': 'Merah Muda',
    'e': 'Merah',
    'w': 'Putih',
    'y': 'Kuning'
}
veil_type_mapping = {
    'p': 'Parsial',
    'u': 'Universal'
}
veil_color_mapping = {
    'n': 'Cokelat',
    'o': 'Oranye',
    'w': 'Putih',
    'y': 'Kuning'
}
ring_number_mapping = {
    'n': 'Tidak Ada',
    'o': 'Satu',
    't': 'Dua'
}
ring_type_mapping = {
    'c': 'Jaring Laba-laba',
    'e': 'Cepat Hilang',
    'f': 'Melebar',
    'l': 'Besar',
    'n': 'Tidak Ada',
    'p': 'Gantung',
    's': 'Sarung',
    'z': 'Zona'
}
spore_print_color_mapping = {
    'k': 'Hitam',
    'n': 'Cokelat',
    'b': 'Buff',
    'h': 'Cokelat Tua',
    'r': 'Hijau',
    'o': 'Oranye',
    'u': 'Ungu',
    'w': 'Putih',
    'y': 'Kuning'
}
population_mapping = {
    'a': 'Melimpah',
    'c': 'Berkelompok',
    'n': 'Banyak',
    's': 'Bersebaran',
    'v': 'Beberapa',
    'y': 'Sendiri'
}
habitat_mapping = {
    'g': 'Rumput',
    'l': 'Daun',
    'm': 'Padang Rumput',
    'p': 'Jalur',
    'u': 'Perkotaan',
    'w': 'Limbah',
    'd': 'Hutan'
}

# Fungsi untuk memproses input pengguna
def preprocess_user_input(user_input):
    # Kolom-kolom yang membutuhkan encoding
    nominal_columns = ['cap-shape', 'cap-surface', 'cap-color', 'odor', 
                       'gill-attachment', 'gill-color', 'stalk-root', 
                       'veil-color', 'ring-type', 'spore-print-color', 
                       'population', 'habitat']
    ordinal_columns = ['bruises', 'gill-spacing', 'gill-size', 'stalk-shape', 
                       'stalk-surface-above-ring', 'stalk-surface-below-ring',
                       'stalk-color-above-ring', 'stalk-color-below-ring', 'ring-number']

    # Gabungkan kolom nominal dan ordinal
    all_columns = nominal_columns + ordinal_columns

    # Nilai default untuk kolom yang hilang
    default_values = {
        'cap-shape': 'x',  # Default convex
        'cap-surface': 's',  # Default smooth
        'cap-color': 'n',  # Default brown
        'odor': 'n',  # Default no odor
        'gill-attachment': 'f',  # Default free
        'gill-color': 'n',  # Default brown
        'stalk-root': 'e',  # Default equal
        'veil-color': 'w',  # Default white
        'ring-type': 'p',  # Default pendant
        'spore-print-color': 'n',  # Default brown
        'population': 's',  # Default scattered
        'habitat': 'g',  # Default grasses
        'bruises': 't',  # Default true (bruised)
        'gill-spacing': 'c',  # Default close
        'gill-size': 'n',  # Default narrow
        'stalk-shape': 't',  # Default tapering
        'stalk-surface-above-ring': 's',  # Default smooth
        'stalk-surface-below-ring': 's',  # Default smooth
        'stalk-color-above-ring': 'w',  # Default white
        'stalk-color-below-ring': 'w',  # Default white
        'ring-number': 'o'  # Default one
    }

    # Tambahkan nilai default untuk kolom yang hilang
    for col in all_columns:
        if col not in user_input:
            user_input[col] = default_values[col]

    # Ubah user_input menjadi DataFrame
    X = pd.DataFrame([user_input])

    # OneHotEncoding untuk kategori nominal
    X_nominal = onehot_encoder.transform(X[nominal_columns])

    # OrdinalEncoding untuk kategori ordinal
    X_ordinal = ordinal_encoder.transform(X[ordinal_columns])

    # Gabungkan hasil encoding
    X_processed = np.hstack([X_nominal, X_ordinal])
    return X_processed

# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    # Ganti dengan path ke dataset Anda
    df = pd.read_csv('https://raw.githubusercontent.com/MuhammadKhoirulMustaqim010/Kuliah/refs/heads/main/mushrooms.csv')  # Misalnya 'dataset.csv'
    return df

# Memuat data
df = load_data()

# Judul aplikasi
st.title("Prediksi Keamanan Jamur: Beracun atau Aman")


# Form untuk input data jamur yang lebih sedikit
st.sidebar.header("Masukkan Data Jamur")

# Input pengguna berdasarkan fitur yang lebih penting
cap_shape = st.sidebar.selectbox("Bentuk Topi", list(cap_shape_mapping.values()))
cap_color = st.sidebar.selectbox("Warna Topi", list(cap_color_mapping.values()))
odor = st.sidebar.selectbox("Aroma Jamur", list(odor_mapping.values()))
stalk_root = st.sidebar.selectbox("Akar Batang Jamur", list(stalk_root_mapping.values()))
ring_type = st.sidebar.selectbox("Jenis Cincin", list(ring_type_mapping.values()))
spore_print_color = st.sidebar.selectbox("Warna Cetakan Spora", list(spore_print_color_mapping.values()))
population = st.sidebar.selectbox("Populasi Jamur", list(population_mapping.values()))
# Menyiapkan input pengguna
user_input = {
    'cap-shape': [key for key, value in cap_shape_mapping.items() if value == cap_shape][0],
     'cap-color': [key for key, value in cap_color_mapping.items() if value == cap_color][0],
    'odor': [key for key, value in odor_mapping.items() if value == odor][0],
    'stalk-root': [key for key, value in stalk_root_mapping.items() if value == stalk_root][0],
    'ring-type': [key for key, value in ring_type_mapping.items() if value == ring_type][0],
     'spore-print-color': [key for key, value in spore_print_color_mapping.items() if value == spore_print_color][0],
    'population': [key for key, value in population_mapping.items() if value == population][0],
}

# Proses input dan prediksi
if st.sidebar.button("Tentukan"):
    try:
        # Mengolah input untuk prediksi
        X_user = preprocess_user_input(user_input)

        # Prediksi dengan model
        prediction = model.predict(X_user)

        # Tampilkan hasil prediksi
        if prediction[0] == 'e':
            st.success("Jamur ini aman untuk dimakan.")
        else:
            st.error("Jamur ini beracun dan harus dihindari.")
    except ValueError as e:
        st.error(f"Terjadi kesalahan: {e}")
