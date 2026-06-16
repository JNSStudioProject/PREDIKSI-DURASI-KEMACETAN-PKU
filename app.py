import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from predictor import predict_traffic

# ==========================================
# CONFIG & INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="TrafficLSTM Forecaster",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def local_css(file_name):
    """Memuat file CSS eksternal."""
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"File CSS tidak ditemukan di: {file_name}")

local_css("assets/style.css")

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-profile">
        <div class="profile-avatar">🚦</div>
        <h3 style="color: white; margin: 0; font-size: 16px; font-weight: 700;">
            DASHBOARD USER
        </h3>
        <p style="color: #fda4af; margin: 0; font-size: 11px;">
            TrafficLSTM Forecaster v1.0
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">INPUT PARAMETERS</p>', unsafe_allow_html=True)

    jam_start, jam_end = st.select_slider(
        "Pilih Rentang Jam Operasional",
        options=[f"{i:02d}:00" for i in range(24)],
        value=("07:00", "12:00")
    )

    hari = st.selectbox("Hari dalam seminggu", [
        "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"
    ], index=1)

    bulan = st.selectbox("Bulan", [
        "Jan", "Feb", "Mar", "Apr", "Mei", "Jun", 
        "Jul", "Agu", "Sep", "Okt", "Nov", "Des"
    ], index=5)

    musim = st.selectbox("Musim", [
        "Summer (Jun-Agu)", "Fall (Sep-Nov)", "Winter (Des-Feb)", "Spring (Mar-Mei)"
    ], index=0)

    st.markdown('<hr style="border-color: #881337; margin: 20px 0;">', unsafe_allow_html=True)

    weather = st.selectbox("Jenis cuaca", [
        "Clear — Cerah", "Clouds — Berawan", "Rain — Hujan", "Snow — Salju"
    ], index=0)
    
    weather_clean = weather.split(" — ")[0]
    temp_c = st.number_input("Suhu (°C)", -30.0, 50.0, 20.0, 0.5)
    clouds = st.slider("Tutupan awan", 0, 100, 40, format="%d%%")
    rain = st.slider("Curah hujan 1 jam (mm)", 0.0, 50.0, 0.0, 0.1)
    snow = st.slider("Curah salju 1 jam (mm)", 0.0, 50.0, 0.0, 0.1)

    btn_predict = st.button("Jalankan Prediksi", type="primary")

# ==========================================
# MAIN HEADER
# ==========================================
st.markdown("""
<div class="main-header">
    <div>
        <h1 style="margin:0; font-size:22px; font-weight:700; color:#4c0519;">
            Overview Analisis Rentang Arus Lalu Lintas
        </h1>
        <p style="margin:4px 0 0 0; font-size:12px; color:#9f1239;">
            Metro Interstate I-94 · Minnesota, USA
        </p>
    </div>
    <div class="env-badge">Environment: Localhost</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# LOGIC & PROCESSING
# ==========================================
if btn_predict:
    start_hour = int(jam_start.split(":")[0])
    end_hour = int(jam_end.split(":")[0])

    data = {"jam": [], "volume": [], "delay": [], "vc": [], "cat": []}

    for h in range(start_hour, end_hour + 1):
        jam_f = f"{h:02d}:00"
        vol, delay, cat = predict_traffic(temp_c, rain, snow, clouds, weather_clean, jam_f)
        
        data["jam"].append(jam_f)
        data["volume"].append(vol)
        data["delay"].append(delay)
        data["vc"].append((vol / 6000) * 100)
        data["cat"].append(cat)

    st.session_state.prediction_results = {
        "list_jam": data["jam"], "list_volume": data["volume"],
        "list_delay": data["delay"], "list_vc_percentage": data["vc"],
        "list_category": data["cat"], "hari": hari,
        "jam_awal": jam_start, "jam_akhir": jam_end
    }

# ==========================================
# DISPLAY RESULTS
# ==========================================
if st.session_state.prediction_results:
    res = st.session_state.prediction_results
    
    # Perhitungan Statistik
    avg_vol = sum(res["list_volume"]) / len(res["list_volume"])
    max_vol = max(res["list_volume"])
    idx = res["list_volume"].index(max_vol)
    
    cat_colors = {"Lancar": "#22c55e", "Sedang": "#eab308", "Padat": "#f97316", "Sangat Padat": "#ef4444"}
    peak_color = cat_colors.get(res["list_category"][idx], "#6b7280")

    # Layout Dashboard
    st.markdown(f'<div class="status-banner">📊 Tren: {res["jam_awal"]} - {res["jam_akhir"]} ({res["hari"]})</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-title">📈 Ringkasan Kondisi Jam Paling Sibuk</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Jam Terpadat", res["list_jam"][idx], f"{max_vol:,.0f} kdr/jam")
        c2.metric("Tundaan Puncak", f"{res['list_delay'][idx]:.3f}", "menit/mil")
        c3.markdown(f'<div class="metric-box"><div class="metric-title">Status</div><div class="metric-value" style="color:{peak_color}">{res["list_category"][idx].upper()}</div></div>', unsafe_allow_html=True)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res["list_jam"], y=res["list_volume"], mode="lines+markers",
        line=dict(color="#9f1239", width=3),
        marker=dict(size=10, color=[cat_colors.get(c, "#6b7280") for c in res["list_category"]])
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Tabel Detail
    df = pd.DataFrame({
        "Jam": res["list_jam"],
        "Volume": [f"{v:,.0f}" for v in res["list_volume"]],
        "Tundaan": [f"{d:.3f}" for d in res["list_delay"]],
        "Status": res["list_category"]
    })
    st.dataframe(df.style.map(lambda x: f"background-color: {cat_colors.get(x, '')}; color: white", subset=["Status"]), use_container_width=True)

else:
    st.info("💡 Atur rentang waktu di sidebar lalu klik 'Jalankan Prediksi'.")