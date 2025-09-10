import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from preprocessing import preprocess_series

def evaluate_model(ticker="AAPL", freq="D", look_back=10):
    model = tf.keras.models.load_model(f"{ticker}_{freq}_model.h5")
    scaler = joblib.load(f"{ticker}_{freq}_scaler.pkl")
    X, y, _, resampled = preprocess_series(ticker, freq, look_back)

    split_idx = int(len(X) * 0.8)
    X_test, y_test = X[split_idx:], y[split_idx:]

    y_pred = model.predict(X_test).reshape(-1, 1)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_pred_inv = scaler.inverse_transform(y_pred).reshape(-1)

    plt.figure(figsize=(12, 5))
    plt.plot(resampled.index[-len(y_test):], y_test_inv, label="Real")
    plt.plot(resampled.index[-len(y_pred):], y_pred_inv, label="Predicción")
    plt.title(f"Predicción final - {ticker} ({freq})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"final_{ticker}_{freq}.png")
    plt.close()

    print(f"Evaluación final completada: final_{ticker}_{freq}.png")

if __name__ == "__main__":
    evaluate_model("AAPL", "D", 10)



