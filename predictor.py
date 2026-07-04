import numpy as np
import math
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

# Definisi rute valid: (distance_km, speed_free_flow, speed_min_macet, base_volume, tipe)
routes = {
    ("Pandau", "Simpang Tiga"): (12.0, 42, 10, 2800, "komuter"),
    ("Simpang SKA", "Bandara SSK II"): (8.5, 40, 15, 2200, "bandara"),
    ("Panam (UNRI)", "Simpang SKA"): (6.0, 38, 10, 2500, "kampus"),
    ("Pasar Pusat", "Rumbai"): (10.0, 35, 12, 2000, "pasar_industri"),
    ("Jl. Sudirman (MTQ)", "Kantor Gubernur"): (5.0, 40, 12, 2400, "perkantoran"),
    ("Harapan Raya", "Sudirman"): (7.5, 35, 10, 2600, "komersial"),
}

def get_category(delay, distance):
    delay_per_km = delay / distance
    if delay_per_km < 0.5:
        return "Lancar"
    elif delay_per_km < 1.2:
        return "Agak Padat"
    elif delay_per_km < 2.5:
        return "Padat"
    elif delay_per_km < 4.0:
        return "Macet"
    return "Macet Total"


def gaussian_peak(hour, center, sigma, amplitude):
    return amplitude * math.exp(-0.5 * ((hour - center) / sigma) ** 2)


def get_congestion_factor(tipe, hour, is_wknd, hari_pilihan):
    """Menghitung congestion_factor berdasarkan tipe rute dan jam."""
    cf = 0.0

    if tipe == "komuter":
        cf = gaussian_peak(hour, 7.0, 1.5, 1.15) + gaussian_peak(hour, 17.5, 2.0, 1.25) + gaussian_peak(hour, 12.5, 1.5, 0.35)
        if is_wknd:
            cf *= 0.30
    elif tipe == "bandara":
        cf = gaussian_peak(hour, 6.5, 1.8, 0.90) + gaussian_peak(hour, 17.0, 1.8, 0.85)
        if is_wknd:
            cf *= 0.85
    elif tipe == "kampus":
        cf = gaussian_peak(hour, 7.5, 1.0, 1.10) + gaussian_peak(hour, 12.5, 1.0, 0.65) + gaussian_peak(hour, 17.0, 1.5, 0.90)
        if is_wknd:
            cf *= 0.20
    elif tipe == "pasar_industri":
        cf = gaussian_peak(hour, 6.5, 2.0, 1.00) + gaussian_peak(hour, 15.5, 1.8, 0.60)
        if is_wknd:
            cf *= 0.55
    elif tipe == "perkantoran":
        cf = gaussian_peak(hour, 7.5, 0.9, 1.05) + gaussian_peak(hour, 12.5, 0.9, 0.45) + gaussian_peak(hour, 16.5, 1.0, 0.90)
        if is_wknd:
            cf *= 0.10
    elif tipe == "komersial":
        cf = gaussian_peak(hour, 11.0, 3.0, 0.55) + gaussian_peak(hour, 18.0, 2.5, 0.75) + gaussian_peak(hour, 14.0, 2.0, 0.45)
        if is_wknd:
            cf *= 1.25

    cf += 0.03  # baseline malam
    return max(0.0, cf)


def get_volume_from_cf(cf, base_vol):
    return int(base_vol * (0.3 + 0.9 * cf))


def predict_traffic(origin, destination, weather, temp_c, jam_pilihan, bulan_pilihan="Mar", hari_pilihan="Senin", accident=False):
    # Ambil info rute
    route_info = routes.get((origin, destination))
    if route_info is None:
        hash_val = hash(origin + destination)
        distance = 3.0 + (abs(hash_val) % 120) / 10.0
        speed_ff = 40
        speed_min = 12
        base_vol = 2000
        tipe = "komuter"
    else:
        distance, speed_ff, speed_min, base_vol, tipe = route_info
        
    # Encoder
    try:
        orig_enc = le_origin.transform([origin])[0]
    except ValueError:
        orig_enc = 0

    try:
        dest_enc = le_dest.transform([destination])[0]
    except ValueError:
        dest_enc = 0
    
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

    for i in range(12):
        h = (target_hour - 12 + i) % 24
        
        cf = get_congestion_factor(tipe, h, is_weekend, hari_pilihan)
        is_rush = 1 if cf >= 0.45 else 0
        volume = get_volume_from_cf(cf, base_vol)
        
        X[i, 0] = orig_enc
        X[i, 1] = dest_enc
        X[i, 2] = distance
        X[i, 3] = weather_enc
        X[i, 4] = is_weekend
        X[i, 5] = is_rush
        X[i, 6] = volume
        X[i, 7] = temp_c
        X[i, 8] = h
        X[i, 9] = month_num

    X_scaled = scaler_X.transform(X)
    X_scaled = X_scaled.reshape(1, 12, 10)

    pred_scaled = model.predict(X_scaled, verbose=0)
    delay = scaler_y.inverse_transform(pred_scaled)[0][0]
    
    delay = max(0.0, float(delay))
    
    # Heuristik untuk kecelakaan
    if accident:
        delay = delay * 1.2 + 10.0

    # Volume untuk jam target
    cf_target = get_congestion_factor(tipe, target_hour, is_weekend, hari_pilihan)
    volume = get_volume_from_cf(cf_target, base_vol)
    if accident:
        volume = int(volume * 0.5)
        
    category = get_category(delay, distance)

    return volume, delay, category