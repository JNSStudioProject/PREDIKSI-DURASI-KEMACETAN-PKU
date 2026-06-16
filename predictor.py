import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model dan scaler
model = load_model("models/lstm_traffic_v1.keras")
scaler = joblib.load("models/scaler_traffic_v1.pkl")

weather_freq_map = {
    "Clouds": 0.31,
    "Clear": 0.28,
    "Rain": 0.11,
    "Snow": 0.02
}

def inverse_target(scaled_val, scaler, target_idx=0, n_features=15):
    dummy = np.zeros((1, n_features))
    dummy[:, target_idx] = scaled_val
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, target_idx][0]

def get_category(volume):
    vc = volume / 6000
    if vc < 0.35:
        return "Lancar"
    elif vc < 0.65:
        return "Agak Padat"
    elif vc < 0.85:
        return "Padat"
    elif vc < 1.0:
        return "Macet"
    return "Macet Total"

def bpr_delay(volume):
    C = 6000
    t0 = 1
    alpha = 0.15
    beta = 4
    return t0 * (1 + alpha * (volume/C)**beta)

# Estimasi volume awal berdasarkan jam agar prediksi tidak stagnan
def get_base_volume_by_hour(jam_str):
    hour = int(jam_str.split(":")[0])
    
    if 0 <= hour < 5:
        return 800   # Tengah malam - Subuh (Sangat Sepi)
    elif 5 <= hour < 7:
        return 2200  # Pagi awal
    elif 7 <= hour < 9:
        return 5200  # Jam Berangkat Kerja (Sibuk)
    elif 9 <= hour < 12:
        return 3800  # Menjelang Siang
    elif 12 <= hour < 14:
        return 4500  # Jam Istirahat Siang
    elif 14 <= hour < 16:
        return 3900  # Sore awal
    elif 16 <= hour < 19:
        return 5500  # Jam Pulang Kerja (Sangat Sibuk)
    elif 19 <= hour < 22:
        return 3200  # Malam
    else:
        return 1500  # Larut malam

def predict_traffic(temp, rain, snow, clouds, weather, jam_pilihan):
    weather_freq = weather_freq_map.get(weather, 0.25)

    # Membuat input sequensial 24 jam untuk LSTM
    X = np.zeros((24, 15))

    base_volume = get_base_volume_by_hour(jam_pilihan)
    X[:, 0] = base_volume  
    X[:, 1] = temp
    X[:, 2] = rain
    X[:, 3] = snow
    X[:, 4] = clouds
    X[:, 5] = weather_freq

    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape(1, 24, 15)

    pred = model.predict(X_scaled, verbose=0)

    volume = inverse_target(pred[0][0], scaler)
    delay = bpr_delay(volume)
    category = get_category(volume)

    return volume, delay, category