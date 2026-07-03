import re
import os

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update imports
content = content.replace("from predictor import predict_traffic", "from predictor import predict_traffic, routes")

# 2. Update Weather API coordinates to Pekanbaru
content = content.replace("lat = 44.9537", "lat = 0.5071")
content = content.replace("lon = -93.0900", "lon = 101.4478")

# 3. Update Sidebar (Add Route selection, replace weather)
sidebar_replacement = """    st.markdown('<p class="sidebar-label">INPUT PARAMETERS</p>', unsafe_allow_html=True)
    
    route_options = [f"{k[0]} ➔ {k[1]}" for k in routes.keys()]
    selected_route = st.selectbox("Pilih Rute PKU", route_options, index=0)
    origin, destination = selected_route.split(" ➔ ")"""

content = re.sub(r'st\.markdown\(\'<p class="sidebar-label">INPUT PARAMETERS</p>\', unsafe_allow_html=True\)', sidebar_replacement, content)

# 4. Remove Snow from Weather Options
content = content.replace('"Clear — Cerah", "Clouds — Berawan", "Rain — Hujan", "Snow — Salju"', '"Cerah", "Mendung", "Hujan Ringan", "Hujan Lebat"')
content = content.replace('weather_clean = weather.split(" — ")[0]', 'weather_clean = weather')

# 5. Remove accident/roadwork inputs from Sidebar A and replace with simple spacing
special_scenario_old = """    st.markdown('<hr style="border-color: #881337; margin: 20px 0;">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label" style="margin-bottom: 10px;">SKENARIO KHUSUS</p>', unsafe_allow_html=True)
    is_accident = st.checkbox("Ada Kecelakaan lalu lintas", value=False)
    is_roadwork = st.checkbox("Sedang Perbaikan Jalan", value=False)"""

content = content.replace(special_scenario_old, "")

# 6. Update Scenario B
scenario_b_old = """        weather_b = st.selectbox("Jenis cuaca (B)", ["Clear — Cerah", "Clouds — Berawan", "Rain — Hujan", "Snow — Salju"], index=2)
        weather_clean_b = weather_b.split(" — ")[0]
        temp_c_b = st.number_input("Suhu (°C) (B)", -30.0, 50.0, value=20.0, step=0.5)
        clouds_b = st.slider("Tutupan awan (B)", 0, 100, value=80, format="%d%%")
        rain_b = st.slider("Curah hujan (mm) (B)", 0.0, 50.0, value=15.0, step=0.1)
        snow_b = st.slider("Curah salju (mm) (B)", 0.0, 50.0, value=0.0, step=0.1)
        is_accident_b = st.checkbox("Ada Kecelakaan (B)", value=True)
        is_roadwork_b = st.checkbox("Perbaikan Jalan (B)", value=False)
    else:
        weather_clean_b = temp_c_b = clouds_b = rain_b = snow_b = is_accident_b = is_roadwork_b = None"""

scenario_b_new = """        selected_route_b = st.selectbox("Pilih Rute Alternatif (B)", route_options, index=1)
        origin_b, destination_b = selected_route_b.split(" ➔ ")
        weather_clean_b = st.selectbox("Jenis cuaca (B)", ["Cerah", "Mendung", "Hujan Ringan", "Hujan Lebat"], index=2)
    else:
        origin_b = destination_b = weather_clean_b = None"""
content = content.replace(scenario_b_old, scenario_b_new)

# 7. Update Prediction Logic Calls
pred_call_1_old = "vol, _, _ = predict_traffic(temp_c, rain, snow, clouds, weather_clean, jam_f, accident=is_accident, roadwork=is_roadwork)"
pred_call_1_new = "vol, _, _ = predict_traffic(origin, destination, weather_clean, jam_f, bulan, hari)"

pred_call_2_old = "vol, delay, cat = predict_traffic(temp_c, rain, snow, clouds, weather_clean, jam_f, accident=is_accident, roadwork=is_roadwork)"
pred_call_2_new = "vol, delay, cat = predict_traffic(origin, destination, weather_clean, jam_f, bulan, hari)"

pred_call_b_old = "vol, delay, cat = predict_traffic(temp_c_b, rain_b, snow_b, clouds_b, weather_clean_b, jam_f, accident=is_accident_b, roadwork=is_roadwork_b)"
pred_call_b_new = "vol, delay, cat = predict_traffic(origin_b, destination_b, weather_clean_b, jam_f, bulan, hari)"

content = content.replace(pred_call_1_old, pred_call_1_new)
content = content.replace(pred_call_2_old, pred_call_2_new)
content = content.replace(pred_call_b_old, pred_call_b_new)

# Remove `is_accident` usage from toast and voice messages
content = re.sub(r'if is_accident:.*?icon="💥"\)', '', content, flags=re.DOTALL)
content = re.sub(r'if snow > 10\.0 or rain > 20\.0:.*?icon="🌨️"\)', '', content, flags=re.DOTALL)
content = re.sub(r'if is_accident:.*?elif res\["list_category"\]\[idx\]', 'if res["list_category"][idx]', content, flags=re.DOTALL)

# Update Map Coordinates
content = content.replace("latitude=44.96,", "latitude=0.5071,")
content = content.replace("longitude=-93.17,", "longitude=101.4478,")
content = content.replace("zoom=11.5,", "zoom=12,")

# Update Map titles
content = content.replace("Peta Pantauan Arus Lalu Lintas I-94", "Peta Pantauan Arus Lalu Lintas Pekanbaru")
content = content.replace("Live Heatmap Area Minneapolis - St. Paul", "Live Heatmap Area Pekanbaru")
content = content.replace("Minneapolis Downtown", "Simpang SKA")
content = content.replace("Jembatan I-94", "Sudirman")
content = content.replace("St. Paul Downtown", "Rumbai")

# Update Dataset Interaktif
content = content.replace("Metro_Interstate_Traffic_Volume.csv", "PKU_Traffic_Dummy.csv")
content = content.replace("Metro Interstate Traffic Volume", "PKU Traffic Dummy Dataset")
content = content.replace("I-94", "Pekanbaru")

# Update Time Series Analysis
content = content.replace("df_asli['date_time']", "df_asli['date']")
content = content.replace("'traffic_volume'", "'delay_minutes'")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Modification complete.")
