import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.graphics.tsaplots import plot_acf

# Cargar datos
csv_path = os.path.join("data", "major-tech-stock-2019-2024.csv")
df = pd.read_csv(csv_path)
df["Date"] = pd.to_datetime(df["Date"])

def analisis_exploratorio():
    print("=== Análisis Exploratorio ===")
    print(df.head())
    print(df.describe())
    print(df.info())
    print("Valores nulos:", df.isnull().sum())

    # Guardar gráfico general
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
    for ticker in tickers:
        df_ticker = df[df["Ticker"] == ticker].sort_values("Date")
        plt.figure(figsize=(10, 4))
        plt.plot(df_ticker["Date"], df_ticker["Close"], label=ticker)
        plt.title(f"Serie temporal - {ticker}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"eda_{ticker}.png")
        plt.close()

        # Autocorrelación
        plt.figure(figsize=(8, 3))
        plot_acf(df_ticker["Close"], lags=20, alpha=0.05)
        plt.title(f"Autocorrelación - {ticker}")
        plt.tight_layout()
        plt.savefig(f"acf_{ticker}.png")
        plt.close()

if __name__ == "__main__":
    analisis_exploratorio()
