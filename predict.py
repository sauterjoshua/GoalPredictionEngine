"""
predict.py

Dieses Skript steuert die Inferenz-Phase (Live-Vorhersage) der Engine.
Es lädt die fertig trainierten Modelle und prognostiziert WM-Ergebnisse.

Zwei Betriebsmodi:
  1. CLI-Einzelvorhersage:  python predict.py 'Germany' 'France'
  2. Batch-Vorhersage:      run_batch_prediction() rechnet alle anstehenden
     Spiele der nächsten N Tage und schreibt sie in eine CSV (für Dashboard).

Marktwerte, aktuelle Form und das Gastgeber-Flag werden automatisch ermittelt.
"""

import sys
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
    """Berechnet die aktuelle Form eines Teams aus seinen letzten 5 Spielen.

    Die Form-Werte werden FRISCH aus den Ergebnissen berechnet (inkl. des
    jüngsten Spiels), statt die vor-Spiel-Form aus der Tabelle auszulesen.
    Dadurch fließt z.B. ein gerade gespieltes WM-Gruppenspiel direkt in die
    nächste Vorhersage ein.

    Args:
        team (str): Der Name der Nationalmannschaft.
        df (pd.DataFrame): Der aufbereitete Datensatz (cleaned_wm_data.csv).

    Returns:
        tuple: (form_attack, form_defense, market_value)
    """
    # Filtere alle Spiele, an denen das Team beteiligt war, chronologisch sortiert
    team_df = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date")

    # Fallback-Logik für Teams, die nicht im Datensatz existieren
    if team_df.empty:
        median_market_value = df["home_market_value"].median()
        return 1.3, 1.3, float(median_market_value)

    # Tore aus Sicht des Teams sammeln (egal ob es Heim oder Auswärts spielte)
    scored, conceded = [], []
    for _, row in team_df.iterrows():
        if row["home_team"] == team:
            scored.append(row["home_score"])
            conceded.append(row["away_score"])
        else:
            scored.append(row["away_score"])
            conceded.append(row["home_score"])

    # Form = Durchschnitt der letzten 5 Spiele (inkl. jüngstem Ergebnis!)
    form_attack = float(np.mean(scored[-5:]))
    form_defense = float(np.mean(conceded[-5:]))

    # Marktwert aus dem jüngsten Spiel ziehen (ändert sich nur langsam)
    last_game = team_df.iloc[-1]
    if last_game["home_team"] == team:
        market_value = float(last_game["home_market_value"])
    else:
        market_value = float(last_game["away_market_value"])

    return form_attack, form_defense, market_value


def predict_match(home_team: str, away_team: str, model_home, model_away,
                  df: pd.DataFrame) -> dict:
    """Rechnet die Tor-Vorhersage für EIN Spiel.

    Zentrale Inferenz-Logik, die sowohl von der CLI-Einzelvorhersage als auch
    von der Batch-Vorhersage genutzt wird (kein duplizierter Code).

    Args:
        home_team, away_team (str): Die beiden Mannschaften.
        model_home, model_away: Die geladenen XGBoost-Modelle.
        df (pd.DataFrame): Der aufbereitete Datensatz für die Form-Ermittlung.

    Returns:
        dict: Vorhersage-Ergebnis mit Toren, Form und Kaderwerten.
    """
    # Features der beiden Teams ermitteln
    home_attack, home_defense, home_val = get_latest_team_stats(home_team, df)
    away_attack, away_defense, away_val = get_latest_team_stats(away_team, df)

    # Input-DataFrame exakt in der Struktur aufbauen, wie es XGBoost erwartet
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

    # Kontinuierliche Tor-Erwartung berechnen (Negative Werte via max(0, x) abfangen)
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
    """Lädt die serialisierten Modelle und den aufbereiteten Datensatz.

    Returns:
        tuple: (model_home, model_away, df)

    Raises:
        FileNotFoundError: Wenn Modelle oder Daten fehlen.
    """
    model_home = joblib.load("models/wm_home_goals_model.pkl")
    model_away = joblib.load("models/wm_away_goals_model.pkl")
    df = pd.read_csv("data/processed/cleaned_wm_data.csv")
    return model_home, model_away, df


def run_batch_prediction(reference_date: str | None = None) -> pd.DataFrame:
    """Rechnet Vorhersagen für alle anstehenden Spiele der nächsten N Tage.

    Liest die anstehenden (TIMED) Spiele aus der Live-CSV, filtert auf das
    Zeitfenster [reference_date, reference_date + window] und schreibt die
    Vorhersagen in eine CSV (für das spätere Dashboard).

    Args:
        reference_date (str | None): Referenzdatum im Format 'YYYY-MM-DD'.
            Wenn None, wird das heutige Datum verwendet. Nützlich zum Testen,
            wenn die WM real noch nicht begonnen hat.

    Returns:
        pd.DataFrame: Die berechneten Vorhersagen (kann leer sein).
    """
    config = load_config()
    live_path = config["tournaments"]["world_cup"]["live_path"]
    predictions_path = config["tournaments"]["world_cup"]["predictions_path"]
    window_days = config.get("prediction", {}).get("prediction_window_days", 3)

    # Spaltenschema für die Output-CSV (auch im Leer-Fall, damit Dashboard es lesen kann)
    output_columns = [
        "date", "home_team", "away_team",
        "pred_home_goals", "pred_away_goals", "tipp_home", "tipp_away",
        "home_form_attack", "away_form_attack",
        "home_market_value", "away_market_value",
    ]

    # Referenzdatum bestimmen (heute oder Test-Override)
    if reference_date is None:
        now = datetime.now()
    else:
        now = datetime.strptime(reference_date, "%Y-%m-%d")
    window_end = now + timedelta(days=window_days)
    print(f"📅 Vorhersage-Fenster: {now.date()} bis {window_end.date()} ({window_days} Tage)")

    # Anstehende Spiele aus der Live-CSV laden
    df_live = pd.read_csv(live_path)
    df_upcoming = df_live[df_live["status"] == "TIMED"].copy()

    # Datum parsen (TIMED-Spiele haben volles ISO-Format wie 2026-06-14T17:00:00Z)
    # utc=True macht die Zeitzone konsistent; danach tz entfernen für Vergleich
    df_upcoming["date"] = pd.to_datetime(
        df_upcoming["date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    df_upcoming = df_upcoming.dropna(subset=["date"])

    # Auf das Zeitfenster der nächsten N Tage filtern
    mask = (df_upcoming["date"] >= now) & (df_upcoming["date"] <= window_end)
    df_window = df_upcoming[mask].sort_values("date")

    # Leer-Fall: CSV mit Headern schreiben, Dashboard zeigt dann "keine Spiele"
    if df_window.empty:
        print("ℹ️ Keine Spiele im Vorhersage-Fenster. Schreibe leere CSV.")
        empty_df = pd.DataFrame(columns=output_columns)
        empty_df.to_csv(predictions_path, index=False)
        return empty_df

    # Modelle + Datensatz einmal laden (nicht pro Spiel!)
    model_home, model_away, df = load_models_and_data()

    print(f"🔮 Rechne Vorhersagen für {len(df_window)} Spiel(e)...")
    results = []
    for _, match in df_window.iterrows():
        pred = predict_match(
            match["home_team"], match["away_team"], model_home, model_away, df
        )
        # Spieldatum für die Ausgabe ergänzen
        pred["date"] = match["date"].date().isoformat()
        results.append(pred)

    # In definierter Spaltenreihenfolge speichern
    predictions_df = pd.DataFrame(results)[output_columns]
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✅ {len(predictions_df)} Vorhersage(n) gespeichert: '{predictions_path}'")

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

    # Zentrale Vorhersage-Logik nutzen (gleiche wie Batch-Modus)
    pred = predict_match(home_team, away_team, model_home, model_away, df)

    # Formatiertes Terminal-Dashboard ausgeben
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
