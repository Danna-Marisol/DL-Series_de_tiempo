import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import joblib
from preprocessing import preprocess_series

def build_model(model_type, look_back, units, dropout, lr):
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(units, input_shape=(look_back, 1),
                      kernel_initializer=initializer, recurrent_initializer=initializer))
    else:
        model.add(GRU(units, input_shape=(look_back, 1),
                     kernel_initializer=initializer, recurrent_initializer=initializer))

    model.add(Dropout(dropout, seed=42))
    model.add(Dense(1, kernel_initializer=initializer))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="mse", metrics=["mae"])
    return model

def train_and_save(ticker="AAPL", freq="D", look_back=10):
    X, y, scaler, resampled = preprocess_series(ticker, freq, look_back)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = build_model("LSTM", look_back, 64, 0.2, 0.001)

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.1,
              epochs=10, batch_size=32, callbacks=[es], verbose=1)

    y_pred = model.predict(X_test).reshape(-1, 1)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_pred_inv = scaler.inverse_transform(y_pred).reshape(-1)

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    # Guardar modelo y scaler
    model.save(f"{ticker}_{freq}_model.h5")
    joblib.dump(scaler, f"{ticker}_{freq}_scaler.pkl")

    # Guardar métricas
    metrics = {"ticker": ticker, "freq": freq, "MSE": mse, "MAE": mae}
    pd.DataFrame([metrics]).to_csv("training_results.csv", index=False)

    print(f"Entrenamiento {ticker}-{freq} -> MSE: {mse:.4f}, MAE: {mae:.4f}")

if __name__ == "__main__":
    train_and_save("AAPL", "D", 10)

