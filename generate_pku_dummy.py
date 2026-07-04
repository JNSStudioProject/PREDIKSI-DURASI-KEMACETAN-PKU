import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Definisi Lokasi dan Jarak (dalam KM) dan Base Density
routes = [
    ("Simpang SKA", "Bandara SSK II", 8.5, 300),
    ("Panam (UNRI)", "Simpang SKA", 5.2, 450),
    ("Pasar Pusat", "Rumbai", 7.0, 250),
    ("Jl. Sudirman (MTQ)", "Kantor Gubernur", 4.5, 400),
    ("Pandau", "Simpang Tiga", 6.0, 150),
    ("Harapan Raya", "Sudirman", 3.8, 380)
]

# Cuaca dan Bobotnya
weathers = ["Cerah", "Mendung", "Hujan Ringan", "Hujan Lebat"]
weather_weights = [0.5, 0.3, 0.15, 0.05]

START_DATE = datetime(2023, 1, 1)
# Generate data for 6 months, hourly, for each route
DAYS = 180
HOURS_PER_DAY = 24

data = []

for route in routes:
    origin, destination, distance_km, base_density = route
    
    # We will simulate weather so it doesn't jump randomly every hour, but holds for a few hours
    current_weather = random.choices(weathers, weights=weather_weights)[0]
    weather_hold_hours = random.randint(1, 4)
    
    for day in range(DAYS):
        for hour in range(HOURS_PER_DAY):
            dt = START_DATE + timedelta(days=day, hours=hour)
            
            # Update weather
            weather_hold_hours -= 1
            if weather_hold_hours <= 0:
                current_weather = random.choices(weathers, weights=weather_weights)[0]
                weather_hold_hours = random.randint(1, 4)
                
            weather = current_weather
            
            # Pola berdasarkan Hari
            weekday_idx = dt.weekday()
            is_weekend = 1 if weekday_idx >= 5 else 0
            
            if weekday_idx < 5: # Senin - Jumat
                is_rush_hour = 1 if dt.hour in [7, 8, 12, 13, 16, 17, 18] else 0
                hourly_profiles = {
                    0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.1, 5: 0.3, 
                    6: 0.7, 7: 1.0, 8: 0.9, 9: 0.7, 10: 0.6, 11: 0.6, 
                    12: 0.9, 13: 0.8, 14: 0.6, 15: 0.7, 
                    16: 0.8, 17: 1.0, 18: 0.9, 19: 0.6, 
                    20: 0.4, 21: 0.3, 22: 0.2, 23: 0.1
                }
            elif weekday_idx == 5: # Sabtu
                is_rush_hour = 1 if dt.hour in [16, 17, 18, 19, 20] else 0
                hourly_profiles = {
                    0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.1, 5: 0.2, 
                    6: 0.4, 7: 0.5, 8: 0.6, 9: 0.6, 10: 0.7, 11: 0.8, 
                    12: 0.8, 13: 0.8, 14: 0.9, 15: 0.9, 
                    16: 1.0, 17: 1.1, 18: 1.1, 19: 1.2, 
                    20: 1.1, 21: 0.9, 22: 0.6, 23: 0.3
                }
            else: # Minggu
                is_rush_hour = 1 if dt.hour in [6, 7, 8, 9, 16, 17, 18, 19] else 0
                hourly_profiles = {
                    0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.1, 5: 0.3, 
                    6: 0.9, 7: 1.1, 8: 1.0, 9: 0.9, 10: 0.7, 11: 0.7, 
                    12: 0.8, 13: 0.8, 14: 0.8, 15: 0.9, 
                    16: 1.0, 17: 1.1, 18: 1.0, 19: 0.9, 
                    20: 0.7, 21: 0.5, 22: 0.3, 23: 0.1
                }
            
            # Simulasi Volume Kendaraan (berdasarkan profil jam dan jarak)
            base_volume_route = distance_km * base_density
            volume_multiplier = hourly_profiles.get(dt.hour, 0.5)
                
            noise = random.uniform(0.9, 1.1)
            base_volume = int(base_volume_route * volume_multiplier * noise)
                
            # Simulasi Kecepatan Rata-Rata Normal
            normal_speed = 50.0
            actual_speed = normal_speed
            
            # Pengurangan kecepatan karena volume tinggi (dihitung per kilometer, bukan absolut)
            density = base_volume / distance_km
            if density > 250:
                actual_speed -= 25 + random.uniform(-2, 2)
            elif density > 200:
                actual_speed -= 15 + random.uniform(-2, 2)
            elif density > 150:
                actual_speed -= 5 + random.uniform(-1, 1)
                
            # Pengurangan kecepatan karena cuaca
            if weather == "Hujan Lebat":
                actual_speed -= 18 + random.uniform(-2, 2)
            elif weather == "Hujan Ringan":
                actual_speed -= 8 + random.uniform(-2, 2)
            elif weather == "Mendung":
                actual_speed -= 2 + random.uniform(-1, 1)
                
            # Pastikan kecepatan tidak masuk akal
            actual_speed = max(5.0, actual_speed)
            
            # Simulasi Suhu (Temperature) berdasarkan cuaca
            if weather == "Cerah":
                temperature_c = round(random.uniform(31.0, 35.0), 1)
            elif weather == "Mendung":
                temperature_c = round(random.uniform(28.0, 31.0), 1)
            elif weather == "Hujan Ringan":
                temperature_c = round(random.uniform(26.0, 28.0), 1)
            else: # Hujan Lebat
                temperature_c = round(random.uniform(24.0, 26.0), 1)
            
            # Hitung Waktu Tempuh & Delay
            normal_travel_time_min = (distance_km / normal_speed) * 60
            actual_travel_time_min = (distance_km / actual_speed) * 60
            
            delay_minutes = max(0, actual_travel_time_min - normal_travel_time_min)
            
            data.append({
                "date_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "origin": origin,
                "destination": destination,
                "distance_km": distance_km,
                "weather": weather,
                "is_weekend": is_weekend,
                "is_rush_hour": is_rush_hour,
                "traffic_volume": base_volume,
                "temperature_c": temperature_c,
                "average_speed_kmh": round(actual_speed, 2),
                "travel_time_minutes": round(actual_travel_time_min, 2),
                "delay_minutes": round(delay_minutes, 2)
            })

# Konversi ke DataFrame
df = pd.DataFrame(data)

# Urutkan berdasarkan waktu
df = df.sort_values(by=["date_time", "origin"]).reset_index(drop=True)

# Simpan ke CSV
output_path = "assets/csv/PKU_Traffic_Dummy.csv"
df.to_csv(output_path, index=False)
print(f"Data dummy berhasil dibuat: {output_path}")
print(f"Total baris: {len(df)}")
print("\nContoh 5 Data Pertama:")
print(df.head())
