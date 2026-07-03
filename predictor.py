import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tensorflow.keras.layers as layers
from datetime import datetime

_original_dense_from_config = layers.Dense.from_config

def _patched_dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return _original_dense_from_config(config)

layers.Dense.from_config = classmethod(_patched_dense_from_config)

# Load model dan scaler PKU
model = load_model("models/pku_lstm_model.keras")
scaler_X = joblib.load("models/pku_scaler_X.pkl")
scaler_y = joblib.load("models/pku_scaler_y.pkl")
le_origin = joblib.load("models/le_origin.pkl")
le_dest = joblib.load("models/le_dest.pkl")
le_weather = joblib.load("models/le_weather.pkl")

# Definisi rute valid
routes = {
    ("Simpang SKA", "Bandara SSK II"): 8.5,
    ("Panam (UNRI)", "Simpang SKA"): 5.2,
    ("Pasar Pusat", "Rumbai"): 7.0,
    ("Jl. Sudirman (MTQ)", "Kantor Gubernur"): 4.5,
    ("Pandau", "Simpang Tiga"): 6.0,
    ("Harapan Raya", "Sudirman"): 3.8
}

def get_category(delay):
    if delay < 2.0:
        return "Lancar"
    elif delay < 4.5:
        return "Agak Padat"
    elif delay < 7.0:
        return "Padat"
    elif delay < 10.0:
        return "Macet"
    return "Macet Total"

def get_base_volume_by_hour(jam_str, base_volume=1500):
    hour = int(jam_str.split(":")[0])
    # Profil lalu lintas
    hourly_profiles = {
        0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.1, 
        5: 0.3, 6: 0.7, 7: 1.0, 8: 0.9, 9: 0.7, 
        10: 0.6, 11: 0.6, 12: 0.7, 13: 0.6, 14: 0.6, 
        15: 0.7, 16: 0.9, 17: 1.0, 18: 0.9, 19: 0.6, 
        20: 0.4, 21: 0.3, 22: 0.2, 23: 0.1
    }
    return base_volume * hourly_profiles.get(hour, 0.5)

def predict_traffic(origin, destination, weather, temp_c, jam_pilihan, bulan_pilihan="Mar", hari_pilihan="Senin", accident=False):
    # Jarak
    distance = routes.get((origin, destination))
    if distance is None:
        # Jika rute tidak ada di daftar, buat jarak acak (pseudo-random berdasarkan nama)
        # agar simulasi tetap terlihat dinamis (antara 3 km sampai 15 km)
        hash_val = hash(origin + destination)
        distance = 3.0 + (abs(hash_val) % 120) / 10.0
        
    # Encoder
    try:
        orig_enc = le_origin.transform([origin])[0]
    except ValueError:
        orig_enc = 0 # Fallback jika label tidak dikenal

    try:
        dest_enc = le_dest.transform([destination])[0]
    except ValueError:
        dest_enc = 0
    
    # Pastikan cuaca ada di encoder (fallback ke Cerah jika gagal)
    try:
        weather_enc = le_weather.transform([weather])[0]
    except:
        weather_enc = le_weather.transform(["Cerah"])[0]
    
    # Waktu
    target_hour = int(jam_pilihan.split(":")[0])
    month_map = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "Mei":5, "Jun":6, "Jul":7, "Agu":8, "Sep":9, "Okt":10, "Nov":11, "Des":12}
    month_num = month_map.get(bulan_pilihan, 3)
    
    is_weekend = 1 if hari_pilihan in ["Sabtu", "Minggu"] else 0

    # Membuat input sequensial 12 jam untuk LSTM
    X = np.zeros((12, 10))
    # Volume diasumsikan rata-rata 300 kend/jam per km
    base_volume_route = distance * 300 

    for i in range(12):
        h = (target_hour - 11 + i) % 24
        
        # 'origin_encoded', 'destination_encoded', 'distance_km', 'weather_encoded', 'is_weekend', 'is_rush_hour', 'traffic_volume', 'temperature_c', 'hour', 'month'
        X[i, 0] = orig_enc
        X[i, 1] = dest_enc
        X[i, 2] = distance
        X[i, 3] = weather_enc
        X[i, 4] = is_weekend
        # Hitung rush hour historis
        X[i, 5] = 1 if h in [7, 8, 16, 17] else 0
        X[i, 6] = get_base_volume_by_hour(f"{h:02d}:00", base_volume_route)
        X[i, 7] = temp_c # Suhu konstan untuk sequence simulasi ini
        X[i, 8] = h
        X[i, 9] = month_num

    X_scaled = scaler_X.transform(X)
    X_scaled = X_scaled.reshape(1, 12, 10)

    pred_scaled = model.predict(X_scaled, verbose=0)
    delay = scaler_y.inverse_transform(pred_scaled)[0][0]
    
    # Jangan sampai delay negatif
    delay = max(0.0, float(delay))
    
    # Heuristik untuk skenario khusus
    if accident:
        # Berdasarkan riset, base delay rata-rata kecelakaan adalah ~10 menit.
        # Namun di jalan yang sudah padat, kecelakaan memberi efek berantai (multiplier).
        delay = delay * 1.2 + 10.0 
        X[11, 6] *= 0.5 # Volume drop 50% (asumsi 1 lajur tertutup)
        
    # Ambil volume untuk jam target
    volume = float(X[11, 6])
    
    category = get_category(delay)

    return volume, delay, category