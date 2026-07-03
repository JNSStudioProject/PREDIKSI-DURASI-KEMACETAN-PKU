import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Tambahkan checkbox kecelakaan di bawah cuaca
target_ui = '    btn_predict = st.button("Jalankan Prediksi", type="primary", use_container_width=True)'
replacement_ui = """    st.markdown('<hr style="border-color: #881337; margin: 20px 0;">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label" style="margin-bottom: 10px;">SKENARIO KHUSUS</p>', unsafe_allow_html=True)
    is_accident = st.checkbox("Ada Kecelakaan lalu lintas", value=False)
    is_roadwork = st.checkbox("Sedang Perbaikan Jalan", value=False)
    
    btn_predict = st.button("Jalankan Prediksi", type="primary", use_container_width=True)"""
content = content.replace(target_ui, replacement_ui)

# Skenario B
target_b = 'weather_clean_b = None'
replacement_b = 'weather_clean_b = None\n        is_accident_b = False\n        is_roadwork_b = False'
content = content.replace(target_b, replacement_b)

target_b_ui = 'temp_c_b = st.number_input("Suhu (\xb0C) (B)", 20.0, 45.0, value=float(st.session_state.get("w_temp", 28.0)), step=0.5)'
replacement_b_ui = 'temp_c_b = st.number_input("Suhu (\xb0C) (B)", 20.0, 45.0, value=float(st.session_state.get("w_temp", 28.0)), step=0.5)\n        is_accident_b = st.checkbox("Ada Kecelakaan (B)", value=True)\n        is_roadwork_b = st.checkbox("Perbaikan Jalan (B)", value=False)'
content = content.replace(target_b_ui, replacement_b_ui)

# Update predict calls
content = content.replace(
    'predict_traffic(origin, destination, weather, temp_c, jam_f, bulan, hari)', 
    'predict_traffic(origin, destination, weather, temp_c, jam_f, bulan, hari, accident=is_accident, roadwork=is_roadwork)'
)

content = content.replace(
    'predict_traffic(origin_b, destination_b, weather_clean_b, temp_c_b, jam_f, bulan, hari)', 
    'predict_traffic(origin_b, destination_b, weather_clean_b, temp_c_b, jam_f, bulan, hari, accident=is_accident_b, roadwork=is_roadwork_b)'
)

# Toast handling if accident
toast_logic = """
    st.session_state.prediction_results = {
        "list_jam": data["jam"], "list_volume": data["volume"],
        "data_b": data_b,
        "list_delay": data["delay"], 
        "list_category": data["cat"], "hari": hari,
        "jam_awal": jam_start, "jam_akhir": jam_end,
        "aktual_jam": aktual["jam"], "aktual_volume": aktual["volume"]
    }
    
    if is_accident:
        st.toast("⚠️ Peringatan: Terdapat Skenario Kecelakaan Aktif!", icon="⚠️")
"""
content = content.replace(
    'st.session_state.prediction_results = {\n        "list_jam": data["jam"], "list_volume": data["volume"],\n        "data_b": data_b,\n        "list_delay": data["delay"], \n        "list_category": data["cat"], "hari": hari,\n        "jam_awal": jam_start, "jam_akhir": jam_end,\n        "aktual_jam": aktual["jam"], "aktual_volume": aktual["volume"]\n    }', 
    toast_logic
)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
