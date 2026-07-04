import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import math

random.seed(42)
np.random.seed(42)

# =============================================================================
# 1. DEFINISI RUTE (baseline tetap)
# =============================================================================
ROUTES = [
    {
        "origin": "Pandau", "destination": "Simpang Tiga",
        "tipe": "komuter", "distance_km": 12.0,
        "speed_free_flow": 42, "speed_min_macet": 10, "base_volume": 2800,
    },
    {
        "origin": "Simpang SKA", "destination": "Bandara SSK II",
        "tipe": "bandara", "distance_km": 8.5,
        "speed_free_flow": 40, "speed_min_macet": 15, "base_volume": 2200,
    },
    {
        "origin": "Panam (UNRI)", "destination": "Simpang SKA",
        "tipe": "kampus", "distance_km": 6.0,
        "speed_free_flow": 38, "speed_min_macet": 10, "base_volume": 2500,
    },
    {
        "origin": "Pasar Pusat", "destination": "Rumbai",
        "tipe": "pasar_industri", "distance_km": 10.0,
        "speed_free_flow": 35, "speed_min_macet": 12, "base_volume": 2000,
    },
    {
        "origin": "Jl. Sudirman (MTQ)", "destination": "Kantor Gubernur",
        "tipe": "perkantoran", "distance_km": 5.0,
        "speed_free_flow": 40, "speed_min_macet": 12, "base_volume": 2400,
    },
    {
        "origin": "Harapan Raya", "destination": "Sudirman",
        "tipe": "komersial", "distance_km": 7.5,
        "speed_free_flow": 35, "speed_min_macet": 10, "base_volume": 2600,
    },
]

# =============================================================================
# 2. CUACA
# =============================================================================
WEATHER_NORMAL = ["Cerah", "Berawan", "Hujan Ringan", "Hujan Deras"]
WEATHER_WEIGHTS_NORMAL = [0.45, 0.30, 0.18, 0.07]
WEATHER_SMOKE = ["Cerah", "Berawan", "Hujan Ringan", "Hujan Deras", "Berkabut/Asap"]
WEATHER_WEIGHTS_SMOKE = [0.20, 0.15, 0.10, 0.05, 0.50]

WEATHER_CONGESTION_FACTOR = {
    "Cerah": 0.0,
    "Berawan": 0.03,
    "Hujan Ringan": 0.12,
    "Hujan Deras": 0.25,
    "Berkabut/Asap": 0.20,
}

# =============================================================================
# 3. KALENDER HARI LIBUR NASIONAL INDONESIA 2025
# =============================================================================
HOLIDAYS_2025 = [
    datetime(2025, 1, 1),   # Tahun Baru
    datetime(2025, 1, 27),  # Isra Mi'raj
    datetime(2025, 1, 29),  # Tahun Baru Imlek
    datetime(2025, 3, 29),  # Hari Raya Nyepi
    datetime(2025, 3, 31),  # Hari Raya Idul Fitri
    datetime(2025, 4, 1),   # Hari Raya Idul Fitri
    datetime(2025, 4, 18),  # Jumat Agung
    datetime(2025, 5, 1),   # Hari Buruh
    datetime(2025, 5, 12),  # Hari Raya Waisak
    datetime(2025, 5, 29),  # Kenaikan Isa Al Masih
    datetime(2025, 6, 1),   # Hari Lahir Pancasila
    datetime(2025, 6, 7),   # Idul Adha
    datetime(2025, 6, 27),  # Tahun Baru Islam
    datetime(2025, 8, 17),  # HUT RI
    datetime(2025, 9, 5),   # Maulid Nabi
    datetime(2025, 12, 25), # Natal
]
HOLIDAY_SET = set(d.date() for d in HOLIDAYS_2025)

# Periode Ramadan 2025: ~1 Mar - 30 Mar
RAMADAN_START = datetime(2025, 3, 1).date()
RAMADAN_END = datetime(2025, 3, 30).date()

# Mudik Lebaran: H-7 s/d H+7 dari Idul Fitri (31 Mar 2025)
IDUL_FITRI = datetime(2025, 3, 31).date()
MUDIK_START = IDUL_FITRI - timedelta(days=7)
MUDIK_END = IDUL_FITRI + timedelta(days=7)

# Libur Semester Kampus
LIBUR_SEMESTER = [
    (datetime(2025, 6, 23).date(), datetime(2025, 8, 31).date()),  # Juni akhir - Agustus
    (datetime(2025, 12, 22).date(), datetime(2025, 12, 31).date()), # Desember akhir
]

# Musim Kabut Asap
SMOKE_START_MONTH = 8  # Agustus
SMOKE_END_MONTH = 10   # Oktober

# Anomali
ANOMALIES = ["Kecelakaan", "Banjir/Genangan", "Perbaikan Jalan", "Kendaraan Mogok", "Demo/Acara Besar"]

# =============================================================================
# 4. FUNGSI HELPER
# =============================================================================

def is_smoke_season(dt):
    return SMOKE_START_MONTH <= dt.month <= SMOKE_END_MONTH

def is_ramadan(dt):
    d = dt.date()
    return RAMADAN_START <= d <= RAMADAN_END

def is_mudik(dt):
    d = dt.date()
    return MUDIK_START <= d <= MUDIK_END

def is_holiday(dt):
    return dt.date() in HOLIDAY_SET

def is_libur_semester(dt):
    d = dt.date()
    for start, end in LIBUR_SEMESTER:
        if start <= d <= end:
            return True
    return False

def pekanbaru_temperature(hour, weather):
    """Kurva suhu harian Pekanbaru: min ~23°C jam 04, max ~34°C jam 14."""
    base = 23.0 + 5.5 * (1 - math.cos(2 * math.pi * (hour - 14) / 24)) / 2
    # Hujan menurunkan suhu
    if weather == "Hujan Deras":
        base -= random.uniform(1.5, 3.0)
    elif weather == "Hujan Ringan":
        base -= random.uniform(0.5, 1.5)
    elif weather == "Berkabut/Asap":
        base -= random.uniform(0.5, 1.0)
    base += random.uniform(-0.5, 0.5)
    return round(max(22.0, min(36.0, base)), 1)


def gaussian_peak(hour, center, sigma, amplitude):
    """Kurva Gaussian untuk jam sibuk."""
    return amplitude * math.exp(-0.5 * ((hour - center) / sigma) ** 2)


def get_congestion_factor_route(route, hour, is_wknd, is_hol, is_ram, is_mud, is_lib_sem):
    """
    Menghitung congestion_factor (0.0 - 1.3+) berdasarkan:
    - Profil jam sibuk spesifik per tipe rute (kurva Gaussian)
    - Hari weekend vs weekday
    - Ramadan, Mudik, Libur Nasional, Libur Semester
    """
    tipe = route["tipe"]
    cf = 0.0

    if tipe == "komuter":
        # Pandau -> Simpang Tiga: puncak 06-08.30 & 16.30-19
        cf = gaussian_peak(hour, 7.0, 1.2, 1.0) + gaussian_peak(hour, 17.5, 1.5, 0.95)
        if is_wknd:
            cf *= 0.30  # Weekend sangat sepi
        if is_hol:
            cf *= 0.25
        if is_mud:
            cf *= 0.50  # Mudik: banyak yang pulang kampung
        if is_ram:
            # Ngabuburit: puncak sore lebih awal
            cf += gaussian_peak(hour, 16.5, 1.0, 0.35)

    elif tipe == "bandara":
        # SKA -> Bandara: jadwal penerbangan pagi & sore
        cf = gaussian_peak(hour, 6.5, 1.5, 0.85) + gaussian_peak(hour, 17.0, 1.5, 0.80)
        # Weekend tetap ramai (penerbangan tetap jalan)
        if is_wknd:
            cf *= 0.85
        if is_hol:
            cf *= 1.10  # Libur justru lebih ramai ke bandara
        if is_mud:
            cf *= 1.20  # Mudik sangat ramai

    elif tipe == "kampus":
        # Panam -> SKA: jam kuliah
        cf = gaussian_peak(hour, 7.5, 0.8, 1.05) + gaussian_peak(hour, 12.5, 0.8, 0.60) + gaussian_peak(hour, 17.0, 1.2, 0.80)
        if is_wknd:
            cf *= 0.20  # Weekend sangat sepi
        if is_hol:
            cf *= 0.15
        if is_lib_sem:
            cf *= 0.25  # Libur semester drastis turun
        if is_mud:
            cf *= 0.30

    elif tipe == "pasar_industri":
        # Pasar Pusat -> Rumbai: pasar pagi buta + shift industri
        cf = gaussian_peak(hour, 6.5, 1.8, 0.95) + gaussian_peak(hour, 15.0, 1.5, 0.50)
        if is_wknd:
            cf *= 0.55  # Pasar masih buka, tapi industri libur
        if is_hol:
            cf *= 0.40
        if is_ram:
            cf += gaussian_peak(hour, 16.5, 1.0, 0.30)  # Ngabuburit belanja

    elif tipe == "perkantoran":
        # Sudirman MTQ -> Kantor Gubernur: jam kantor klasik
        cf = gaussian_peak(hour, 7.5, 0.7, 1.0) + gaussian_peak(hour, 12.5, 0.7, 0.40) + gaussian_peak(hour, 16.5, 0.8, 0.85)
        if is_wknd:
            cf *= 0.10  # Sangat sepi weekend
        if is_hol:
            cf *= 0.08  # Libur nasional hampir kosong
        if is_mud:
            cf *= 0.40
        if is_ram:
            cf += gaussian_peak(hour, 16.0, 0.8, 0.30)

    elif tipe == "komersial":
        # Harapan Raya -> Sudirman: komersial, merata siang-malam
        cf = gaussian_peak(hour, 11.0, 3.0, 0.55) + gaussian_peak(hour, 18.0, 2.5, 0.75) + gaussian_peak(hour, 14.0, 2.0, 0.45)
        if is_wknd:
            cf *= 1.25  # Justru lebih ramai weekend (belanja/kuliner)
        if is_hol:
            cf *= 1.15
        if is_ram:
            cf += gaussian_peak(hour, 17.0, 1.0, 0.40)  # Ngabuburit

    # Baseline malam (selalu ada sedikit traffic)
    cf += 0.03

    return max(0.0, cf)


# =============================================================================
# 5. GENERATE DATA
# =============================================================================

START_DATE = datetime(2025, 1, 1)
DAYS = 365
HOURS_PER_DAY = 24

print("Generating realistic Pekanbaru traffic dataset...")

data = []

# Pre-generate shared weather per timestamp (cuaca satu kota)
weather_cache = {}
current_weather = "Cerah"
weather_hold = 0

for day in range(DAYS):
    for hour in range(HOURS_PER_DAY):
        dt = START_DATE + timedelta(days=day, hours=hour)
        
        weather_hold -= 1
        if weather_hold <= 0:
            if is_smoke_season(dt):
                current_weather = random.choices(WEATHER_SMOKE, weights=WEATHER_WEIGHTS_SMOKE)[0]
            else:
                current_weather = random.choices(WEATHER_NORMAL, weights=WEATHER_WEIGHTS_NORMAL)[0]
            weather_hold = random.randint(2, 6)  # Cuaca bertahan 2-6 jam
        
        weather_cache[(day, hour)] = current_weather

# Generate per rute
for route in ROUTES:
    origin = route["origin"]
    destination = route["destination"]
    distance_km = route["distance_km"]
    speed_ff = route["speed_free_flow"]
    speed_min = route["speed_min_macet"]
    base_vol = route["base_volume"]
    tipe = route["tipe"]
    
    free_flow_time = distance_km / speed_ff * 60  # menit
    
    for day in range(DAYS):
        for hour in range(HOURS_PER_DAY):
            dt = START_DATE + timedelta(days=day, hours=hour)
            
            is_wknd = 1 if dt.weekday() >= 5 else 0
            hol = is_holiday(dt)
            ram = is_ramadan(dt)
            mud = is_mudik(dt)
            lib_sem = is_libur_semester(dt)
            
            weather = weather_cache[(day, hour)]
            
            # --- Congestion Factor ---
            cf = get_congestion_factor_route(route, hour, is_wknd, hol, ram, mud, lib_sem)
            
            # Tambah efek cuaca
            cf += WEATHER_CONGESTION_FACTOR.get(weather, 0.0)
            
            # Noise kecil
            cf += random.uniform(-0.03, 0.03)
            cf = max(0.0, cf)
            
            # --- Anomali (2-5% baris) ---
            extra_delay_anomaly = 0.0
            extra_speed_penalty = 0.0
            anomaly_prob = 0.03  # 3% default
            
            if tipe == "perkantoran" and random.random() < 0.005:
                # Demo dekat Kantor Gubernur (0.5% chance)
                extra_delay_anomaly = random.uniform(15, 35)
                extra_speed_penalty = random.uniform(8, 15)
            elif random.random() < anomaly_prob:
                anomaly_type = random.choice(ANOMALIES[:4])  # Exclude Demo
                if anomaly_type == "Kecelakaan":
                    extra_delay_anomaly = random.uniform(8, 25)
                    extra_speed_penalty = random.uniform(5, 12)
                elif anomaly_type == "Banjir/Genangan":
                    extra_delay_anomaly = random.uniform(5, 20)
                    extra_speed_penalty = random.uniform(3, 10)
                elif anomaly_type == "Perbaikan Jalan":
                    extra_delay_anomaly = random.uniform(5, 15)
                    extra_speed_penalty = random.uniform(3, 8)
                elif anomaly_type == "Kendaraan Mogok":
                    extra_delay_anomaly = random.uniform(3, 12)
                    extra_speed_penalty = random.uniform(2, 6)
            
            # --- Average Speed ---
            cf_capped = min(cf, 1.0)
            avg_speed = speed_ff - (speed_ff - speed_min) * cf_capped - extra_speed_penalty
            avg_speed += random.uniform(-1.5, 1.5)  # noise kecil
            avg_speed = max(speed_min * 0.6, min(speed_ff + 2, avg_speed))  # clamp
            
            # --- Travel Time & Delay ---
            dist_actual = distance_km + random.uniform(-0.05, 0.05)  # noise jarak
            travel_time = dist_actual / avg_speed * 60 + extra_delay_anomaly
            travel_time += random.uniform(-0.3, 0.3)  # noise kecil
            travel_time = max(free_flow_time * 0.95, travel_time)  # tidak lebih cepat dari free flow
            
            delay = travel_time - free_flow_time
            delay = max(0.0, delay)
            
            # --- is_rush_hour ---
            is_rush = 1 if cf >= 0.45 else 0
            
            # --- Traffic Volume ---
            volume = base_vol * (0.3 + 0.9 * cf) * random.uniform(0.9, 1.1)
            volume = max(50, int(volume))
            
            # --- Temperature ---
            temp = pekanbaru_temperature(hour, weather)
            
            data.append({
                "date_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "origin": origin,
                "destination": destination,
                "distance_km": round(distance_km, 1),
                "weather": weather,
                "is_weekend": is_wknd,
                "is_rush_hour": is_rush,
                "traffic_volume": volume,
                "temperature_c": temp,
                "average_speed_kmh": round(avg_speed, 2),
                "travel_time_minutes": round(travel_time, 2),
                "delay_minutes": round(delay, 2)
            })

# =============================================================================
# 6. KONVERSI & SANITY CHECK
# =============================================================================

df = pd.DataFrame(data)
df = df.sort_values(by=["origin", "destination", "date_time"]).reset_index(drop=True)

# Sanity checks
assert (df["delay_minutes"] >= 0).all(), "FAIL: Ada delay negatif!"
assert df.duplicated(subset=["date_time", "origin", "destination"]).sum() == 0, "FAIL: Ada duplikat!"

for route in ROUTES:
    mask = (df["origin"] == route["origin"]) & (df["destination"] == route["destination"])
    route_df = df[mask]
    dist_vals = route_df["distance_km"].unique()
    assert len(dist_vals) == 1, f"FAIL: Jarak tidak konsisten untuk {route['origin']} -> {route['destination']}: {dist_vals}"
    max_speed = route_df["average_speed_kmh"].max()
    assert max_speed <= route["speed_free_flow"] + 5, f"FAIL: Speed melebihi free flow untuk {route['origin']}: {max_speed}"

print("Semua sanity check PASSED!")

# Simpan
output_path = "assets/csv/PKU_Traffic_Dummy.csv"
df.to_csv(output_path, index=False)
print(f"\nData dummy berhasil dibuat: {output_path}")
print(f"Total baris: {len(df)}")
print(f"Rentang tanggal: {df['date_time'].min()} s/d {df['date_time'].max()}")
print(f"\nDistribusi rute:")
print(df.groupby(["origin", "destination"]).size())
print(f"\nDistribusi cuaca:")
print(df["weather"].value_counts())
print(f"\nStatistik delay_minutes:")
print(df["delay_minutes"].describe())
print(f"\nContoh 5 Data Pertama:")
print(df.head())
