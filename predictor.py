import numpy as np
import math
import joblib
from tensorflow.keras.models import load_model
import tensorflow.keras.layers as layers
from datetime import datetime, date
import os

_original_dense_from_config = layers.Dense.from_config

def _patched_dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return _original_dense_from_config(config)

layers.Dense.from_config = classmethod(_patched_dense_from_config)

# Load model dan scaler PKU
model = load_model("models/pku_lstm_model.keras")
scaler_X = joblib.load("models/pku_scaler_X.pkl")
scaler_y = joblib.load("models/pku_scaler_y.pkl")

# Load daftar kolom dari proses training terbaru
feature_cols = joblib.load("models/feature_columns.pkl")
onehot_cols = joblib.load("models/onehot_columns.pkl")

# Definisi rute valid: (distance_km, speed_free_flow, speed_min_macet, base_volume, tipe)
routes = {
    ("Pandau", "Simpang Tiga"): (12.0, 42, 10, 2800, "komuter"),
    ("Simpang SKA", "Bandara SSK II"): (8.5, 40, 15, 2200, "bandara"),
    ("Panam (UNRI)", "Simpang SKA"): (6.0, 38, 10, 2500, "kampus"),
    ("Pasar Pusat", "Rumbai"): (10.0, 35, 12, 2000, "pasar_industri"),
    ("Jl. Sudirman (MTQ)", "Kantor Gubernur"): (5.0, 40, 12, 2400, "perkantoran"),
    ("Harapan Raya", "Sudirman"): (7.5, 35, 10, 2600, "komersial"),
}

HOLIDAYS_2025 = {
    date(2025, 1, 1), date(2025, 1, 27), date(2025, 1, 29),
    date(2025, 3, 29), date(2025, 3, 31), date(2025, 4, 1),
    date(2025, 4, 18), date(2025, 5, 1), date(2025, 5, 12),
    date(2025, 5, 29), date(2025, 6, 1), date(2025, 6, 6),
    date(2025, 6, 27), date(2025, 8, 17), date(2025, 9, 5),
    date(2025, 12, 25),
}
RAMADAN_START, RAMADAN_END = date(2025, 3, 1), date(2025, 3, 29)
MUDIK_START, MUDIK_END = date(2025, 3, 24), date(2025, 4, 8)

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

def get_congestion_factor(tipe, hour, is_wknd, is_hol, is_ram, is_mud):
    """Menghitung congestion_factor berdasarkan tipe rute dan jam."""
    cf = 0.0

    if tipe == "komuter":
        cf = gaussian_peak(hour, 7.0, 1.5, 1.15) + gaussian_peak(hour, 17.5, 2.0, 1.25) + gaussian_peak(hour, 12.5, 1.5, 0.35)
        if is_wknd:
            cf *= 0.30
        if is_hol:
            cf *= 0.25
        if is_mud:
            cf *= 0.50
        if is_ram:
            cf += gaussian_peak(hour, 16.5, 1.2, 0.40)
            
    elif tipe == "bandara":
        cf = gaussian_peak(hour, 6.5, 1.8, 0.90) + gaussian_peak(hour, 17.0, 1.8, 0.85)
        if is_wknd:
            cf *= 0.85
        if is_hol:
            cf *= 1.10
        if is_mud:
            cf *= 1.20
            
    elif tipe == "kampus":
        cf = gaussian_peak(hour, 7.5, 1.0, 1.10) + gaussian_peak(hour, 12.5, 1.0, 0.65) + gaussian_peak(hour, 17.0, 1.5, 0.90)
        if is_wknd:
            cf *= 0.20
        if is_hol:
            cf *= 0.15
        if is_mud:
            cf *= 0.30
            
    elif tipe == "pasar_industri":
        cf = gaussian_peak(hour, 6.5, 2.0, 1.00) + gaussian_peak(hour, 15.5, 1.8, 0.60)
        if is_wknd:
            cf *= 0.55
        if is_hol:
            cf *= 0.40
        if is_ram:
            cf += gaussian_peak(hour, 16.5, 1.0, 0.30)
            
    elif tipe == "perkantoran":
        cf = gaussian_peak(hour, 7.5, 0.9, 1.05) + gaussian_peak(hour, 12.5, 0.9, 0.45) + gaussian_peak(hour, 16.5, 1.0, 0.90)
        if is_wknd:
            cf *= 0.10
        if is_hol:
            cf *= 0.08
        if is_mud:
            cf *= 0.40
        if is_ram:
            cf += gaussian_peak(hour, 16.0, 0.8, 0.30)
            
    elif tipe == "komersial":
        cf = gaussian_peak(hour, 11.0, 3.0, 0.55) + gaussian_peak(hour, 18.0, 2.5, 0.75) + gaussian_peak(hour, 14.0, 2.0, 0.45)
        if is_wknd:
            cf *= 1.25
        if is_hol:
            cf *= 1.15
        if is_ram:
            cf += gaussian_peak(hour, 17.0, 1.0, 0.40)

    cf += 0.03  # baseline malam
    return max(0.0, cf)

def get_volume_from_cf(cf, base_vol):
    return int(base_vol * (0.3 + 0.9 * cf))

def get_onehot_vector(weather, origin, destination):
    # Buat dictionary sesuai nama kolom one-hot, inisialisasi 0
    onehot_dict = {col: 0 for col in onehot_cols}
    
    # Set ke 1 jika fiturnya ada
    if f"weather_{weather}" in onehot_dict:
        onehot_dict[f"weather_{weather}"] = 1
    if f"origin_{origin}" in onehot_dict:
        onehot_dict[f"origin_{origin}"] = 1
    if f"dest_{destination}" in onehot_dict:
        onehot_dict[f"dest_{destination}"] = 1
        
    return [onehot_dict[col] for col in onehot_cols]

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
        
    # Ekstraksi Waktu
    target_hour = int(jam_pilihan.split(":")[0])
    month_map = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "Mei":5, "Jun":6, "Jul":7, "Agu":8, "Sep":9, "Okt":10, "Nov":11, "Des":12}
    month_num = month_map.get(bulan_pilihan, 3)
    
    # Asumsikan tgl 15 untuk representasi bulan berjalan
    try:
        simulated_date = date(2025, month_num, 15)
    except:
        simulated_date = date(2025, 3, 15)
        
    hari_map = {"Senin":0, "Selasa":1, "Rabu":2, "Kamis":3, "Jumat":4, "Sabtu":5, "Minggu":6}
    dow_num = hari_map.get(hari_pilihan, 0)
    
    is_weekend = 1 if dow_num >= 5 else 0
    is_hol = 1 if simulated_date in HOLIDAYS_2025 else 0
    is_ram = 1 if RAMADAN_START <= simulated_date <= RAMADAN_END else 0
    is_mud = 1 if MUDIK_START <= simulated_date <= MUDIK_END else 0
    
    # Dapatkan vektor one-hot
    onehot_vals = get_onehot_vector(weather, origin, destination)
    
    # Siapkan array input (12 jam berurutan)
    X = []
    
    for i in range(12):
        h = (target_hour - 12 + i) % 24
        
        # Hitung congestion factor historis
        cf = get_congestion_factor(tipe, h, is_weekend, is_hol, is_ram, is_mud)
        is_rush = 1 if cf >= 0.45 else 0
        volume = get_volume_from_cf(cf, base_vol)
        
        # 1. Fitur Continuous (Akan di-scale)
        cont_features = [distance, volume, temp_c]
        
        # 2. Fitur Binary
        bin_features = [is_weekend, is_rush, is_hol, is_ram, is_mud]
        
        # 3. Fitur Cyclical (sin & cos)
        cyc_features = [
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            np.sin(2 * np.pi * month_num / 12), np.cos(2 * np.pi * month_num / 12),
            np.sin(2 * np.pi * dow_num / 7), np.cos(2 * np.pi * dow_num / 7)
        ]
        
        # Gabungkan semua fitur menjadi 1 baris sequence
        row = cont_features + bin_features + cyc_features + onehot_vals
        X.append(row)

    X = np.array(X, dtype=np.float32)
    
    # Scale HANYA fitur continuous (3 kolom pertama)
    X[:, 0:3] = scaler_X.transform(X[:, 0:3])
    
    # Reshape ke (batch_size, time_steps, features)
    X_scaled = X.reshape(1, 12, len(feature_cols))

    # Prediksi menggunakan LSTM
    pred_scaled = model.predict(X_scaled, verbose=0)
    
    # Inverse transform (MinMax -> log1p)
    delay_log = scaler_y.inverse_transform(pred_scaled)[0][0]
    
    # Kembalikan ke menit aktual dengan eksponensial (expm1)
    delay = np.expm1(delay_log)
    delay = max(0.0, float(delay))
    
    # Tambahkan heuristik khusus jika ada kecelakaan
    if accident:
        delay = delay * 1.2 + 10.0

    # Kalkulasi volume aktual pada jam tersebut untuk ditampilkan
    cf_target = get_congestion_factor(tipe, target_hour, is_weekend, is_hol, is_ram, is_mud)
    volume = get_volume_from_cf(cf_target, base_vol)
    if accident:
        volume = int(volume * 0.5)
        
    category = get_category(delay, distance)

    return volume, delay, category