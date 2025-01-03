import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import joblib

# Preprocessing dataset
def preprocess_data(df):
    # Pisahkan target dan fitur
    X = df.drop('class', axis=1)
    y = df['class']

    # Pilih kolom untuk encoding
    nominal_columns = ['cap-shape', 'cap-surface', 'cap-color', 'odor', 
                        'gill-attachment', 'gill-color', 'stalk-root', 
                        'veil-color', 'ring-type', 'spore-print-color', 
                        'population', 'habitat']
    ordinal_columns = ['bruises', 'gill-spacing', 'gill-size', 'stalk-shape', 
                       'stalk-surface-above-ring', 'stalk-surface-below-ring',
                       'stalk-color-above-ring', 'stalk-color-below-ring', 'ring-number']

    # OneHotEncoding untuk kategori nominal
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Ubah sparse menjadi sparse_output
    X_nominal = onehot_encoder.fit_transform(X[nominal_columns])

    # OrdinalEncoding untuk kategori ordinal
    ordinal_encoder = OrdinalEncoder()
    X_ordinal = ordinal_encoder.fit_transform(X[ordinal_columns])

    # Gabungkan kembali data
    X_processed = np.hstack([X_nominal, X_ordinal])
    return X_processed, y, onehot_encoder, ordinal_encoder


# Load data
df = pd.read_csv('https://raw.githubusercontent.com/MuhammadKhoirulMustaqim010/Kuliah/refs/heads/main/mushrooms.csv')
X, y, onehot_encoder, ordinal_encoder = preprocess_data(df)

# Membagi data menjadi data latih dan data uji (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Membuat model Random Forest
model = RandomForestClassifier(
    n_estimators=100,  # Meningkatkan jumlah estimator
    max_depth=5,  # Memperdalam pohon keputusan
    random_state=42
)
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, 'models/random_forest_model.pkl')  # Simpan model yang benar
joblib.dump(onehot_encoder, 'models/onehot_encoder.pkl')
joblib.dump(ordinal_encoder, 'models/ordinal_encoder.pkl')
