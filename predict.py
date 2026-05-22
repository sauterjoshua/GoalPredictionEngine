"""
Inferenz-Phase der Engine: lädt Modelle und prognostiziert WM-Ergebnisse.

Modi:
  CLI:   python predict.py 'Germany' 'France'
  Batch: run_batch_prediction()  →  nächste N Tage → predictions.csv
"""

import sys
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import yaml

# Offizielle Gastgeber-Liste der WM 2026 (zentral, von beiden Modi genutzt)
HOSTS_2026 = ["USA", "United States", "Canada", "Mexico"]


def load_config() -> dict:
    """Lädt die zentrale Konfigurationsdatei (config.yaml)."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_latest_team_stats(team: str, df: pd.DataFrame) -> tuple[float, float, float]:
    """Berechnet Form + Marktwert eines Teams aus seinen letzten 5 Spielen.

    Form wird FRISCH aus Ergebnissen berechnet (inkl. jüngstem Spiel), damit
    ein gerade gespieltes Gruppenspiel sofort in die nächste Vorhersage einfließt.

    Returns:
        tuple: (form_attack, form_defense, market_value)
    """
    team_df = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date")

    # Unbekanntes Team → Median-Marktwert und WM-Durchschnitts-Form als Fallback
    if team_df.empty:
        median_market_value = df["home_market_value"].median()
        return 1.3, 1.3, float(median_market_value)

    # Tore aus Sicht des Teams (egal ob Heim oder Auswärts)
    scored, conceded = [], []
    for _, row in team_df.iterrows():
        if row["home_team"] == team:
            scored.append(row["home_score"])
            conceded.append(row["away_score"])
        else:
            scored.append(row["away_score"])
            conceded.append(row["home_score"])

    form_attack = float(np.mean(scored[-5:]))
    form_defense = float(np.mean(conceded[-5:]))

    # Marktwert aus dem jüngsten Spiel (ändert sich nur langsam zwischen Turnieren)
    last_game = team_df.iloc[-1]
    if last_game["home_team"] == team:
        market_value = float(last_game["home_market_value"])
    else:
        market_value = float(last_game["away_market_value"])

    return form_attack, form_defense, market_value


def predict_match(home_team: str, away_team: str, model_home, model_away,
                  df: pd.DataFrame) -> dict:
    """Tor-Vorhersage für ein einzelnes Spiel (CLI- und Batch-Modus teilen diese Logik)."""
    home_attack, home_defense, home_val = get_latest_team_stats(home_team, df)
    away_attack, away_defense, away_val = get_latest_team_stats(away_team, df)

    input_data = pd.DataFrame([{
        "home_form_attack": home_attack,
        "home_form_defense": home_defense,
        "away_form_attack": away_attack,
        "away_form_defense": away_defense,
        "home_market_value": home_val,
        "away_market_value": away_val,
        "home_is_host": 1 if home_team in HOSTS_2026 else 0,
        "away_is_host": 1 if away_team in HOSTS_2026 else 0,
        "neutral": 1
    }])

    pred_home_goals = max(0.0, float(model_home.predict(input_data)[0]))
    pred_away_goals = max(0.0, float(model_away.predict(input_data)[0]))

    return {
        "home_team": home_team,
        "away_team": away_team,
        "pred_home_goals": round(pred_home_goals, 2),
        "pred_away_goals": round(pred_away_goals, 2),
        "tipp_home": round(pred_home_goals),
        "tipp_away": round(pred_away_goals),
        "home_form_attack": round(home_attack, 2),
        "away_form_attack": round(away_attack, 2),
        "home_market_value": round(home_val, 1),
        "away_market_value": round(away_val, 1),
    }


def load_models_and_data():
    """Lädt die serialisierten XGBoost-Modelle und den aufbereiteten Datensatz."""
    model_home = joblib.load("models/wm_home_goals_model.pkl")
    model_away = joblib.load("models/wm_away_goals_model.pkl")
    df = pd.read_csv("data/processed/cleaned_wm_data.csv")
    return model_home, model_away, df


def log_predictions_to_history(predictions_df: pd.DataFrame, history_path: str):
    """Hängt neue Vorhersagen an die History-CSV an.

    Schreibt nur, wenn sich die exakten Tor-Erwartungen gegenüber dem letzten
    Eintrag geändert haben (verhindert Duplikate bei unveränderter Form).
    """
    now_stamp = datetime.now().isoformat(timespec="seconds")

    # match_key = eindeutige Spiel-ID für späteren Join mit echten Ergebnissen
    df_new = predictions_df.copy()
    df_new["match_key"] = (
        df_new["date"] + "_" + df_new["home_team"] + "_" + df_new["away_team"]
    )
    df_new["predicted_at"] = now_stamp

    if os.path.exists(history_path):
        history = pd.read_csv(history_path)
    else:
        history = pd.DataFrame()

    rows_to_append = []
    for _, row in df_new.iterrows():
        key = row["match_key"]

        if not history.empty and key in history["match_key"].values:
            last = history[history["match_key"] == key].iloc[-1]
            unchanged = (
                abs(last["pred_home_goals"] - row["pred_home_goals"]) < 1e-9
                and abs(last["pred_away_goals"] - row["pred_away_goals"]) < 1e-9
            )
            if unchanged:
                continue

        rows_to_append.append(row)

    if not rows_to_append:
        print("ℹ️ History: keine geänderten Vorhersagen, nichts angehängt.")
        return

    updated = pd.concat([history, pd.DataFrame(rows_to_append)], ignore_index=True)
    updated.to_csv(history_path, index=False)
    print(f"📝 History: {len(rows_to_append)} neue Vorhersage(n) geloggt → '{history_path}'")
    
    
def run_batch_prediction(reference_date: str | None = None) -> pd.DataFrame:
    """Rechnet Vorhersagen für alle TIMED-Spiele im Fenster [heute, heute + N Tage].

    Args:
        reference_date: 'YYYY-MM-DD' zum Testen; None = heute.

    Returns:
        pd.DataFrame: Berechnete Vorhersagen (kann leer sein).
    """
    config = load_config()
    live_path = config["tournaments"]["world_cup"]["live_path"]
    predictions_path = config["tournaments"]["world_cup"]["predictions_path"]
    window_days = config.get("prediction", {}).get("prediction_window_days", 3)

    # Festes Schema auch im Leer-Fall, damit das Dashboard die CSV korrekt liest
    output_columns = [
        "date", "home_team", "away_team",
        "pred_home_goals", "pred_away_goals", "tipp_home", "tipp_away",
        "home_form_attack", "away_form_attack",
        "home_market_value", "away_market_value",
    ]

    if reference_date is None:
        now = datetime.now()
    else:
        now = datetime.strptime(reference_date, "%Y-%m-%d")
    window_end = now + timedelta(days=window_days)
    print(f"📅 Vorhersage-Fenster: {now.date()} bis {window_end.date()} ({window_days} Tage)")

    df_live = pd.read_csv(live_path)
    df_upcoming = df_live[df_live["status"] == "TIMED"].copy()

    # utc=True normalisiert ISO-Timestamps (z.B. 2026-06-14T17:00:00Z);
    # tz_localize(None) entfernt die Timezone für den Vergleich mit datetime.now()
    df_upcoming["date"] = pd.to_datetime(
        df_upcoming["date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    df_upcoming = df_upcoming.dropna(subset=["date"])

    mask = (df_upcoming["date"] >= now) & (df_upcoming["date"] <= window_end)
    df_window = df_upcoming[mask].sort_values("date")

    # Leere CSV mit Headern schreiben → Dashboard kann sie trotzdem einlesen
    if df_window.empty:
        print("ℹ️ Keine Spiele im Vorhersage-Fenster. Schreibe leere CSV.")
        empty_df = pd.DataFrame(columns=output_columns)
        empty_df.to_csv(predictions_path, index=False)
        return empty_df

    # Einmalig laden, nicht pro Spiel
    model_home, model_away, df = load_models_and_data()

    print(f"🔮 Rechne Vorhersagen für {len(df_window)} Spiel(e)...")
    results = []
    for _, match in df_window.iterrows():
        pred = predict_match(
            match["home_team"], match["away_team"], model_home, model_away, df
        )
        pred["date"] = match["date"].date().isoformat()
        results.append(pred)

    predictions_df = pd.DataFrame(results)[output_columns]
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✅ {len(predictions_df)} Vorhersage(n) gespeichert: '{predictions_path}'")

    history_path = config["tournaments"]["world_cup"]["history_path"]
    log_predictions_to_history(predictions_df, history_path)
    
    return predictions_df


def main():
    """CLI-Einzelvorhersage für zwei manuell angegebene Teams."""
    if len(sys.argv) < 3:
        print("💡 Nutzung im Terminal: python predict.py '<Heimteam>' '<Auswärtsteam>'")
        print("   Beispiel: python predict.py 'Germany' 'France'")
        sys.exit(1)

    home_team = sys.argv[1]
    away_team = sys.argv[2]

    try:
        model_home, model_away, df = load_models_and_data()
    except FileNotFoundError as e:
        print(f"❌ Fehler: Wichtige Projektdateien fehlen ({e}).")
        print("   Bitte führe zuerst prepare_data.py und train.py aus.")
        sys.exit(1)

    pred = predict_match(home_team, away_team, model_home, model_away, df)

    print("\n" + "=" * 55)
    print(f"🔮 PREDICTION ENGINE — WM 2026 SIMULATION:")
    print(f"   {home_team} vs. {away_team}")
    print("=" * 55)
    print(f"💰 Kaderwert {home_team:12}: {pred['home_market_value']:6.1f}M € | Form-Angriff: {pred['home_form_attack']:.2f}")
    print(f"💰 Kaderwert {away_team:12}: {pred['away_market_value']:6.1f}M € | Form-Angriff: {pred['away_form_attack']:.2f}")
    print("-" * 55)
    print(f"⚽ Erwartete Tore {home_team}: {pred['pred_home_goals']:.2f}")
    print(f"⚽ Erwartete Tore {away_team}: {pred['pred_away_goals']:.2f}")
    print("-" * 55)
    print(f"🏆 Mathematischer Tipp: {home_team} {pred['tipp_home']} - {pred['tipp_away']} {away_team}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
