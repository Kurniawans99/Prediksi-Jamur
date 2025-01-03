from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load model dan encoder
model = joblib.load('random_forest_model.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    data = request.json
    cap_shape = data['cap_shape']
    cap_surface = data['cap_surface']
    cap_color = data['cap_color']
    gill_color = data['gill_color']
    odor = data['odor']

    # Transformasi data sesuai model
    nominal_features = np.array([[cap_shape, cap_surface, cap_color, odor]])
    ordinal_features = np.array([[gill_color]])  # Sesuaikan dengan fitur ordinal
    
    nominal_encoded = onehot_encoder.transform(nominal_features)
    ordinal_encoded = ordinal_encoder.transform(ordinal_features)
    
    # Gabungkan semua fitur
    X_input = np.hstack([nominal_encoded, ordinal_encoded])

    # Prediksi menggunakan model
    prediction = model.predict(X_input)

    # Konversi hasil prediksi ke teks
    result = "Aman dimakan" if prediction[0] == 0 else "Beracun"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
