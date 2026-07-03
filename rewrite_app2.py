import re
import os

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add Temperature Input Main
target_main = 'weather_clean = weather'
replacement_main = 'weather_clean = weather\n    temp_c = st.number_input("Suhu (\xb0C)", 20.0, 45.0, value=float(st.session_state.get("w_temp", 28.0)), step=0.5)'
content = content.replace(target_main, replacement_main)

# Add Temperature Input B
target_b = 'weather_clean_b = st.selectbox("Jenis cuaca (B)", ["Cerah", "Mendung", "Hujan Ringan", "Hujan Lebat"], index=2)'
replacement_b = 'weather_clean_b = st.selectbox("Jenis cuaca (B)", ["Cerah", "Mendung", "Hujan Ringan", "Hujan Lebat"], index=2)\n        temp_c_b = st.number_input("Suhu (\xb0C) (B)", 20.0, 45.0, value=float(st.session_state.get("w_temp", 28.0)), step=0.5)'
content = content.replace(target_b, replacement_b)

# Update Predict calls
content = content.replace(
    'predict_traffic(origin, destination, weather, jam_f, bulan, hari)', 
    'predict_traffic(origin, destination, weather, temp_c, jam_f, bulan, hari)'
)

content = content.replace(
    'predict_traffic(origin_b, destination_b, weather_clean_b, jam_f, bulan, hari)', 
    'predict_traffic(origin_b, destination_b, weather_clean_b, temp_c_b, jam_f, bulan, hari)'
)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("App modified.")
