# TrafficLSTM: Enterprise AI Forecaster & Fleet Manager

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)

**TrafficLSTM** is an advanced, enterprise-ready web dashboard powered by Deep Learning (Long Short-Term Memory networks). It is designed to forecast traffic volume on the I-94 Interstate highway by analyzing historical data and simulating the impact of complex weather conditions (snow, rain, cloud cover, and temperature).

Beyond raw predictions, this system features a **B2B Fleet Management Calculator** that translates traffic delays into real-world financial metrics, helping logistics companies optimize their departure schedules and minimize fuel waste.

## Key Enterprise Features
- **Deep Learning Core:** Built on a robust LSTM model to capture sequential time-series traffic patterns.
- **Dynamic Weather Simulation:** Real-time parameter tuning for extreme weather, accidents, and roadwork scenarios.
- **Live Heatmap:** Interactive 3D maps using PyDeck to visualize congestion segments.
- **Fleet Logistics Planner:** Calculates estimated financial losses (in IDR) and fuel waste for logistics convoys trapped in traffic.
- **AI Chat Assistant:** A smart conversational interface that extracts insights directly from the AI's predictive state.

## How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/JNSStudioProject/TrafficLSTM-Deep-Learning-Traffic-Forecaster-Fleet-Optimization-Dashboard.git
   cd TrafficLSTM-Deep-Learning-Traffic-Forecaster-Fleet-Optimization-Dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**
   ```bash
   streamlit run app.py
   ```
