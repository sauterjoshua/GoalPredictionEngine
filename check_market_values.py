import pandas as pd

PATH = "data/raw/fifa_player_performance_market_value.csv"

try:
    df = pd.read_csv(PATH, nrows=5)
    print("🔍 Gefundene Spalten im Marktwert-Datensatz:")
    print(df.columns.tolist())
    print("\n📋 Einblick in die Daten:")
    print(df.head(2))
except Exception as e:
    print(f"❌ Fehler: {e}")