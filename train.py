import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import tensorflow as tf

print("Loading data...")
df = pd.read_csv('assets/csv/PKU_Traffic_Dummy.csv')
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.sort_values(by='date_time').reset_index(drop=True)
df['hour'] = df['date_time'].dt.hour
df['month'] = df['date_time'].dt.month

le_origin = LabelEncoder()
le_dest = LabelEncoder()
le_weather = LabelEncoder()
df['origin_encoded'] = le_origin.fit_transform(df['origin'])
df['destination_encoded'] = le_dest.fit_transform(df['destination'])
df['weather_encoded'] = le_weather.fit_transform(df['weather'])

joblib.dump(le_origin, 'models/le_origin.pkl')
joblib.dump(le_dest, 'models/le_dest.pkl')
joblib.dump(le_weather, 'models/le_weather.pkl')

features = [
    'origin_encoded', 'destination_encoded', 'distance_km', 'weather_encoded', 
    'is_weekend', 'is_rush_hour', 'traffic_volume', 'temperature_c', 'hour', 'month'
]
target = ['delay_minutes']

print("Scaling data...")
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
df_scaled_X = scaler_X.fit_transform(df[features])
df_scaled_y = scaler_y.fit_transform(df[target])
df_scaled = pd.DataFrame(df_scaled_X, columns=features)
df_scaled['delay_minutes_scaled'] = df_scaled_y
df_scaled['origin'] = df['origin']
df_scaled['destination'] = df['destination']
df_scaled['date_time'] = df['date_time']

joblib.dump(scaler_X, 'models/pku_scaler_X.pkl')
joblib.dump(scaler_y, 'models/pku_scaler_y.pkl')

TIME_STEPS = 12
X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

routes = df_scaled.groupby(['origin', 'destination'])
for (orig, dest), group in routes:
    group = group.sort_values(by='date_time').reset_index(drop=True)
    feat_values = group[features].values
    target_values = group['delay_minutes_scaled'].values
    
    route_X, route_y = [], []
    for i in range(len(group) - TIME_STEPS):
        route_X.append(feat_values[i:i + TIME_STEPS])
        route_y.append(target_values[i + TIME_STEPS])
        
    route_X = np.array(route_X)
    route_y = np.array(route_y)
    
    train_idx = int(len(route_X) * 0.70)
    val_idx = int(len(route_X) * 0.85)
    
    X_train.extend(route_X[:train_idx])
    y_train.extend(route_y[:train_idx])
    X_val.extend(route_X[train_idx:val_idx])
    y_val.extend(route_y[train_idx:val_idx])
    X_test.extend(route_X[val_idx:])
    y_test.extend(route_y[val_idx:])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Shuffle training data
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

print(f"Train Shape: {X_train.shape}, {X_train.shape[1]}, {X_train.shape[2]}")

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=Huber(), metrics=['mae'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

print("Training model...")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

model.save("models/pku_lstm_model.keras")
print("Model and scalers saved successfully!")
