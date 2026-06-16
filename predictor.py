import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tensorflow.keras.layers as layers

_original_dense_from_config = layers.Dense.from_config

def _patched_dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return _original_dense_from_config(config)

layers.Dense.from_config = classmethod(_patched_dense_from_config)

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

def get_category(volume, accident=False, roadwork=False):
    C = 6000
    if accident: C = C * 0.4
    elif roadwork: C = C * 0.7
    vc = volume / C
    if vc < 0.35:
        return "Lancar"
    elif vc < 0.65:
        return "Agak Padat"
    elif vc < 0.85:
        return "Padat"
    elif vc < 1.0:
        return "Macet"
    return "Macet Total"

def bpr_delay(volume, accident=False, roadwork=False):
    C = 6000
    if accident: C = C * 0.4
    elif roadwork: C = C * 0.7
    t0 = 1
    alpha = 0.15
    beta = 4
    return t0 * (1 + alpha * (volume/C)**beta)

# Estimasi volume awal berdasarkan jam agar prediksi lebih dinamis dan realistis (membentuk kurva)
def get_base_volume_by_hour(jam_str):
    hour = int(jam_str.split(":")[0])
    
    # Profil lalu lintas yang mulus (smooth) untuk 24 jam
    hourly_profiles = {
        0: 900, 1: 700, 2: 500, 3: 400, 4: 800,        # Dini hari sangat sepi
        5: 1500, 6: 3200,                              # Pagi (mulai aktivitas)
        7: 5200, 8: 5500,                              # Puncak Berangkat Kerja (Sangat Sibuk)
        9: 4500, 10: 3800, 11: 3900,                   # Menjelang Siang (Melonggar)
        12: 4500, 13: 4300,                            # Jam Istirahat Siang (Sedikit Naik)
        14: 3900, 15: 4100,                            # Sore Awal
        16: 5100, 17: 5800, 18: 5400,                  # Puncak Pulang Kerja (Paling Macet)
        19: 4200, 20: 3200, 21: 2500,                  # Malam Hari (Lancar)
        22: 1800, 23: 1200                             # Larut Malam
    }
    
    return hourly_profiles.get(hour, 1500)

def predict_traffic(temp, rain, snow, clouds, weather, jam_pilihan, accident=False, roadwork=False):
    weather_freq = weather_freq_map.get(weather, 0.25)

    # Membuat input sequensial 24 jam untuk LSTM (Data Science Best Practice)
    X = np.zeros((24, 15))
    target_hour = int(jam_pilihan.split(":")[0])

    for i in range(24):
        # Simulasi urutan waktu: 24 jam sebelum (dan termasuk) jam target
        # i=0 adalah 23 jam yang lalu, i=23 adalah jam target saat ini
        h = (target_hour - 23 + i) % 24
        
        # 1. Volume historis berfluktuasi sesuai jam sebenarnya
        X[i, 0] = get_base_volume_by_hour(f"{h:02d}:00")  
        
        # 2. Suhu sintetis (Suhu puncak di siang hari, terdingin di dini hari)
        # Asumsi 'temp' dari input adalah suhu referensi hari itu
        hour_diff = abs(h - 14) # Jarak dari jam 14:00 (siang terpanas)
        temp_variation = -5.0 * (hour_diff / 12.0) # Suhu bisa turun hingga 5 derajat di malam hari
        X[i, 1] = temp + temp_variation
        
        # 3. Faktor cuaca lainnya
        X[i, 2] = rain
        X[i, 3] = snow
        X[i, 4] = clouds
        X[i, 5] = weather_freq

    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape(1, 24, 15)

    pred = model.predict(X_scaled, verbose=0)

    volume = inverse_target(pred[0][0], scaler)
    delay = bpr_delay(volume, accident, roadwork)
    category = get_category(volume, accident, roadwork)

    return volume, delay, category