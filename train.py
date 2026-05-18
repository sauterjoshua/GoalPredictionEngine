"""
train.py

Dieses Skript steuert das Modell-Training der CornerPredictionEngine.
Es lädt die aufbereiteten WM-Daten, führt einen strikten, chronologischen
Time-Based Split durch (Training auf historischen Daten, Test auf modernen Turnieren)
und trainiert zwei stark regulierte XGBoost-Regressoren für Heim- und Auswärtstore.
"""

import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import mean_absolute_error


def load_config() -> dict:
    """Lädt die zentrale Konfigurationsdatei (config.yaml).

    Returns:
        dict: Die geladenen Konfigurationsparameter.
    """
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """Hauptfunktion für das Laden der Features, den chronologischen Split,

    das Training der XGBoost-Modelle und die anschließende Evaluierung.
    """
    config = load_config()
    data_path = config["tournaments"]["world_cup"]["processed_path"]

    print(f"⏳ Lade aufbereitete moderne WM-Daten aus '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Fehler: Die Datei '{data_path}' wurde nicht gefunden. Bitte prepare_data.py zuerst ausführen.")
        sys.exit(1)
        
    df["date"] = pd.to_datetime(df["date"])

    # Exakte Feature-Liste, auf der die Bäume trainiert werden
    features = [
        "home_form_attack",
        "home_form_defense",
        "away_form_attack",
        "away_form_defense",
        "home_market_value",
        "away_market_value",
        "home_is_host",
        "away_is_host",
        "neutral",
    ]

    # --- CHRONOLOGISCHER TIME-BASED SPLIT ---
    # Zur Vermeidung von Data Leakage und für ein realistisches Backtesting
    # trainieren wir auf den WMs 2000-2014 und testen auf den WMs 2018-2022.
    print("✂️ Führe zeitbasierten Split durch (Training < 2018 | Test >= 2018)...")
    train_df = df[df["date"].dt.year < 2018].reset_index(drop=True)
    test_df = df[df["date"].dt.year >= 2018].reset_index(drop=True)

    if len(train_df) == 0 or len(test_df) == 0:
        print("❌ Fehler: Split fehlgeschlagen. Überprüfe die Jahreszahlen in den Daten.")
        sys.exit(1)

    print(f"   📊 Trainings-Matches: {len(train_df)} | Test-Matches: {len(test_df)}")

    # Features und Zielvariablen (Scores) trennen
    X_train = train_df[features]
    y_home_train = train_df["home_score"]
    y_away_train = train_df["away_score"]

    X_test = test_df[features]
    y_home_test = test_df["home_score"]
    y_away_test = test_df["away_score"]

    # --- MODELL-TRAINING (HEIMTORE) ---
    # Stark reguliert (max_depth=3, reg_alpha/lambda), um Overfitting im Keim zu ersticken
    print("🤖 Trainiere stark reguliertes Modell für HEIM-Tore...")
    model_home = xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.02,
        max_depth=3,  # Flache Bäume verhindern das Auswendiglernen von Details
        subsample=0.7,  # Nutzt nur 70% der Zeilen pro Baum zur Varianz-Reduktion
        colsample_bytree=0.7,  # Nutzt nur 70% der Spalten pro Baum
        reg_alpha=1.5,  # L1-Regularisierung (Sparsity)
        reg_lambda=1.5,  # L2-Regularisierung (Ridge-Strafe auf extreme Gewichte)
        random_state=42,  # Fixierter Seed für exakte Reproduzierbarkeit
    )
    model_home.fit(X_train, y_home_train)

    # --- MODELL-TRAINING (AUSWÄRTSTORE) ---
    print("🤖 Trainiere stark reguliertes Modell für AUSWÄRTS-Tore...")
    model_away = xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.5,
        reg_lambda=1.5,
        random_state=42,
    )
    model_away.fit(X_train, y_away_train)

    # --- EVALUIERUNG ---
    pred_home = model_home.predict(X_test)
    pred_away = model_away.predict(X_test)

    # Berechnung des mittleren absoluten Fehlers (MAE)
    mae_home = mean_absolute_error(y_home_test, pred_home)
    mae_away = mean_absolute_error(y_away_test, pred_away)

    print("\n--- 📊 PERFORMANCE-DIAGNOSTIK (WM 2018/2022) ---")
    print(f"   Fehler Heimtore: {mae_home:.2f} Tore im Schnitt daneben.")
    print(f"   Fehler Auswärtstore: {mae_away:.2f} Tore im Schnitt daneben.")

    # Statistischer Baseline-Vergleich gegen den globalen Torschnitt (Dummy)
    baseline_home_mae = mean_absolute_error(
        y_home_test, np.full_like(y_home_test, 1.3, dtype=float)
    )
    print(f"   💡 Zum Vergleich der Turnier-Dummy-Tipp: {baseline_home_mae:.2f}")

    # Serialization (Modelle einfrieren für spätere Live-Inferenz)
    joblib.dump(model_home, "models/wm_home_goals_model.pkl")
    joblib.dump(model_away, "models/wm_away_goals_model.pkl")
    print("\n💾 Beide High-End-Modelle erfolgreich im Ordner 'models/' gespeichert!")


if __name__ == "__main__":
    main()