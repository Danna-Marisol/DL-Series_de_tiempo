import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

csv_path = "data/major-tech-stock-2019-2024.csv"
df = pd.read_csv(csv_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

def create_sequences(values, look_back):
    X, y = [], []
    for i in range(len(values) - look_back):
        X.append(values[i:i+look_back])
        y.append(values[i+look_back])
    return np.array(X), np.array(y)

def preprocess_series(ticker, freq="D", look_back=10):
    df_ticker = df[df["Ticker"] == ticker].set_index("Date")[["Close"]]
    resampled = df_ticker.resample(freq).last().dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(resampled.values.reshape(-1, 1))

    X, y = create_sequences(scaled, look_back)
    return X, y, scaler, resampled

if __name__ == "__main__":
    X, y, scaler, res = preprocess_series("AAPL", "D", 10)
    print(f"Secuencias creadas: {X.shape}, {y.shape}")
