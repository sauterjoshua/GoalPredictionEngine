"""
Modell-Training: chronologischer Split (Train < 2018 / Test >= 2018),
Time-Decay-Gewichtung und zwei XGBoost-Regressoren für Heim-/Auswärtstore.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import mean_absolute_error


# Zentrale Hyperparameter für beide Modelle (Heim & Auswärts identisch).

MODEL_PARAMS = dict(
    n_estimators=220,
    learning_rate=0.02,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
)


def load_config() -> dict:
    """Lädt die zentrale Konfigurationsdatei (config.yaml)."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def outcome_1x2_vec(home, away):
    """1 = Heimsieg, 2 = Auswärtssieg, 0 = Remis (für ganze Arrays)."""
    home, away = np.asarray(home), np.asarray(away)
    return np.where(home > away, 1, np.where(home < away, 2, 0))


def main():
    """Hauptfunktion für das Laden der Features, den chronologischen Split,
    das Training der XGBoost-Modelle und die anschließende Evaluierung.
    """
    config = load_config()
    data_path = config["tournaments"]["world_cup"]["processed_path"]

    # decay_rate steuert, wie stark alte WMs verblassen:
    # 0.0 = gleichgewichtet | 0.15 = moderat (Empfehlung) | 0.5 = aggressiv
    decay_rate = config["model"]["decay_rate"]

    print(f"⏳ Lade aufbereitete moderne WM-Daten aus '{data_path}'...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Fehler: Die Datei '{data_path}' wurde nicht gefunden. Bitte prepare_data.py zuerst ausführen.")
        sys.exit(1)

    df["date"] = pd.to_datetime(df["date"])

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

    # Chronologischer Split verhindert Data Leakage: Train 2000-2014, Test 2018-2022
    print("✂️ Führe zeitbasierten Split durch (Training < 2018 | Test >= 2018)...")
    train_df = df[df["date"].dt.year < 2018].reset_index(drop=True)
    test_df = df[df["date"].dt.year >= 2018].reset_index(drop=True)

    if len(train_df) == 0 or len(test_df) == 0:
        print("❌ Fehler: Split fehlgeschlagen. Überprüfe die Jahreszahlen in den Daten.")
        sys.exit(1)

    print(f"   📊 Trainings-Matches: {len(train_df)} | Test-Matches: {len(test_df)}")

    X_train = train_df[features]
    y_home_train = train_df["home_score"]
    y_away_train = train_df["away_score"]

    X_test = test_df[features]
    y_home_test = test_df["home_score"]
    y_away_test = test_df["away_score"]

    # Time-Decay nur auf Trainingsdaten (Testdaten niemals gewichten → verfälscht MAE)
    # Exponentieller Decay: w = exp(-λ * years_ago)
    # Beispiel λ=0.15: 2002 → w≈0.027 (irrelevant), 2014 → w≈0.165, 2022 → w≈0.549
    current_year = 2026
    years_ago = current_year - train_df["date"].dt.year
    weights = np.exp(-decay_rate * years_ago)

    print(f"⚖️ Time-Decay aktiv (λ={decay_rate}). "
          f"Gewichtsbereich: {weights.min():.3f} bis {weights.max():.3f}")

    print("🤖 Trainiere Modell für HEIM-Tore...")
    model_home = xgb.XGBRegressor(**MODEL_PARAMS)
    model_home.fit(X_train, y_home_train, sample_weight=weights)

    print("🤖 Trainiere Modell für AUSWÄRTS-Tore...")
    model_away = xgb.XGBRegressor(**MODEL_PARAMS)
    model_away.fit(X_train, y_away_train, sample_weight=weights)

    pred_home = model_home.predict(X_test)
    pred_away = model_away.predict(X_test)

    mae_home = mean_absolute_error(y_home_test, pred_home)
    mae_away = mean_absolute_error(y_away_test, pred_away)

    print("\n--- 📊 PERFORMANCE-DIAGNOSTIK (WM 2018/2022) ---")
    print(f"   Fehler Heimtore: {mae_home:.2f} Tore im Schnitt daneben.")
    print(f"   Fehler Auswärtstore: {mae_away:.2f} Tore im Schnitt daneben.")

    baseline_home_mae = mean_absolute_error(
        y_home_test, np.full_like(y_home_test, 1.3, dtype=float)
    )
    print(f"   💡 Zum Vergleich der Turnier-Dummy-Tipp: {baseline_home_mae:.2f}")

    # --- 🎯 Ergebnis-Metriken: messen, was "wer gewinnt" wirklich trifft ---
    # 1X2 aus den KONTINUIERLICHEN Werten ableiten, nicht aus gerundeten Tipps —
    # sonst verschwindet ein knapper Favoriten-Vorsprung (1.41 vs 1.05) im Runden.
    pred_outcome = outcome_1x2_vec(pred_home, pred_away)
    true_outcome = outcome_1x2_vec(y_home_test, y_away_test)
    acc_1x2 = (pred_outcome == true_outcome).mean()

    # Exact-Score: beide Tore (gerundet, ≥0) müssen exakt stimmen
    ph = np.clip(np.round(pred_home), 0, None).astype(int)
    pa = np.clip(np.round(pred_away), 0, None).astype(int)
    acc_exact = ((ph == y_home_test.to_numpy()) & (pa == y_away_test.to_numpy())).mean()

    # Baseline "immer Remis" → trifft genau den Anteil echter Unentschieden
    draw_baseline = (true_outcome == 0).mean()

    print("\n--- 🎯 ERGEBNIS-METRIKEN (WM 2018/2022) ---")
    print(f"   1X2-Trefferquote (wer gewinnt):  {acc_1x2:.1%}")
    print(f"   Exact-Score-Trefferquote:        {acc_exact:.1%}")
    print(f"   💡 Baseline 'immer Remis':       {draw_baseline:.1%}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_home, "models/wm_home_goals_model.pkl")
    joblib.dump(model_away, "models/wm_away_goals_model.pkl")
    print("\n💾 Beide Modelle erfolgreich im Ordner 'models/' gespeichert!")


if __name__ == "__main__":
    main()