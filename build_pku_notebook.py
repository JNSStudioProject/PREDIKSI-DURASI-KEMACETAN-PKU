import nbformat as nbf

nb = nbf.v4.new_notebook()

# Helper function to create markdown and code cells
def add_md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def add_code(code):
    nb.cells.append(nbf.v4.new_code_cell(code))

add_md("# PKU Traffic Delay Prediction - LSTM Model\nNotebook ini dibuat untuk melatih model LSTM memprediksi durasi kemacetan (`delay_minutes`) dari data rute-rute di Pekanbaru.")

add_md("## 1. Import Libraries")
add_code('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 100
''')

add_md("## 2. Load Data & Exploratory Data Analysis (EDA)")
add_code('''# Load Data
df = pd.read_csv('assets/csv/PKU_Traffic_Dummy.csv')
print(f"Total Baris: {df.shape[0]}, Total Kolom: {df.shape[1]}")
display(df.head())
''')

add_code('''# Cek Missing Values
print("Missing Values:")
print(df.isnull().sum())
''')

add_code('''# Visualisasi Distribusi Delay Minutes (Durasi Kemacetan)
plt.figure(figsize=(10, 5))
sns.histplot(df['delay_minutes'], bins=50, kde=True, color='indianred')
plt.title('Distribusi Waktu Kemacetan (Delay Minutes)')
plt.xlabel('Delay (Menit)')
plt.ylabel('Frekuensi')
plt.show()
''')

add_md("## 3. Preprocessing")
add_code('''# Konversi Datetime dan Sort
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.sort_values(by='date_time').reset_index(drop=True)

# Tambahkan fitur Jam dan Bulan
df['hour'] = df['date_time'].dt.hour
df['month'] = df['date_time'].dt.month
''')

add_code('''# Label Encoding untuk Kolom Kategori
le_origin = LabelEncoder()
le_dest = LabelEncoder()
le_weather = LabelEncoder()

df['origin_encoded'] = le_origin.fit_transform(df['origin'])
df['destination_encoded'] = le_dest.fit_transform(df['destination'])
df['weather_encoded'] = le_weather.fit_transform(df['weather'])

# Simpan encoder (jika dibutuhkan di inferensi)
joblib.dump(le_origin, 'models/le_origin.pkl')
joblib.dump(le_dest, 'models/le_dest.pkl')
joblib.dump(le_weather, 'models/le_weather.pkl')
''')

add_code('''# Memilih fitur untuk model
# Kita akan gunakan: origin_encoded, destination_encoded, distance_km, weather_encoded, is_weekend, is_rush_hour, traffic_volume, hour, month
# Target: delay_minutes

features = [
    'origin_encoded', 'destination_encoded', 'distance_km', 'weather_encoded', 
    'is_weekend', 'is_rush_hour', 'traffic_volume', 'temperature_c', 'hour', 'month'
]
target = ['delay_minutes']

# Scaling Data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

df_scaled_X = scaler_X.fit_transform(df[features])
df_scaled_y = scaler_y.fit_transform(df[target])

# Tambahkan kembali data terskala ke dataframe untuk proses sequence
df_scaled = pd.DataFrame(df_scaled_X, columns=features)
df_scaled['delay_minutes_scaled'] = df_scaled_y
df_scaled['origin'] = df['origin']
df_scaled['destination'] = df['destination']
df_scaled['date_time'] = df['date_time']

joblib.dump(scaler_X, 'models/pku_scaler_X.pkl')
joblib.dump(scaler_y, 'models/pku_scaler_y.pkl')
''')

add_md("## 4. Time-Series Sequencing per Rute")
add_code('''# Karena LSTM butuh urutan waktu yang kontinu, kita buat sequence per pasang rute (Origin-Destination)
TIME_STEPS = 12 # Misal menggunakan 12 langkah waktu sebelumnya (12 data point historis)

X, y = [], []

# Group by Rute
routes = df_scaled.groupby(['origin', 'destination'])

for (orig, dest), group in routes:
    # Sort per rute berdasarkan waktu (meski sudah disort global)
    group = group.sort_values(by='date_time').reset_index(drop=True)
    
    # Ambil nilai array fitur dan target
    feat_values = group[features].values
    target_values = group['delay_minutes_scaled'].values
    
    for i in range(len(group) - TIME_STEPS):
        X.append(feat_values[i:i + TIME_STEPS])
        y.append(target_values[i + TIME_STEPS])

X = np.array(X)
y = np.array(y)

print(f"Shape X: {X.shape}")
print(f"Shape y: {y.shape}")
''')

add_code('''# Train-Validation-Test Split (Sequential Split 70% / 15% / 15%)
train_idx = int(len(X) * 0.70)
val_idx = int(len(X) * 0.85)

X_train, y_train = X[:train_idx], y[:train_idx]
X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
X_test, y_test = X[val_idx:], y[val_idx:]

print(f"Train Shape: {X_train.shape}")
print(f"Validation Shape: {X_val.shape}")
print(f"Test Shape: {X_test.shape}")
''')

add_md("## 5. Build & Train LSTM Model")
add_code('''# Arsitektur Model LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
''')

add_code('''# Training Model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)
''')

add_md("## 6. Evaluation & Save Model")
add_code('''# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()
''')

add_code('''# Prediksi dan Transformasi Balik (Inverse Scaling)
y_pred_scaled = model.predict(X_test)

# Inverse transform
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Hitung Metrik Error
mae = mean_absolute_error(y_test_inv, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.2f} menit")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} menit")
''')

add_code('''# Plot Aktual vs Prediksi
plt.figure(figsize=(15, 5))
plt.plot(y_test_inv[:1500], label='Aktual Delay (Menit)', color='blue', alpha=0.6, linewidth=1)
plt.plot(y_pred[:1500], label='Prediksi Delay (Menit)', color='red', alpha=0.6, linewidth=1)
plt.title('Perbandingan Waktu Delay Aktual vs Prediksi (Sample 1500 Data Pertama)')
plt.xlabel('Sample Data')
plt.ylabel('Delay (Menit)')
plt.legend()
plt.show()
''')

add_code('''# Simpan Model
model.save('models/pku_lstm_model.keras')
print("Model berhasil disimpan ke models/pku_lstm_model.keras")
''')

with open('PKU_Traffic_Model.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook PKU_Traffic_Model.ipynb created successfully!")
