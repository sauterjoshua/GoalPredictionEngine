"""
predict.py

Dieses Skript steuert die Inferenz-Phase (Live-Vorhersage) der Engine.
Es lädt die fertig trainierten Modelle und prognostiziert für zwei beliebige
Nationalteams das wahrscheinlichste WM-Ergebnis. Marktwerte, aktuelle Form
und das Gastgeber-Flag für die WM 2026 werden automatisch im Hintergrund ermittelt.

Live-Spiele aus wm2026_live.csv fließen direkt in die Formberechnung ein,
ohne dass prepare_data.py neu ausgeführt werden muss.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import yaml


def load_live_finished_games(live_path: str) -> pd.DataFrame:
    """Lädt abgeschlossene Live-Spiele direkt aus der Live-CSV.

    Diese Funktion wird bei jedem Predict-Aufruf ausgeführt, sodass
    manuell hinzugefügte oder per API geholte FINISHED-Spiele sofort
    in die Formberechnung einfließen – ohne prepare_data.py neu starten.

    Args:
        live_path (str): Pfad zur wm2026_live.csv.

    Returns:
        pd.DataFrame: DataFrame mit FINISHED-Spielen (date, home_team,
                      away_team, home_score, away_score) oder leerer DataFrame.
    """
    if not os.path.exists(live_path):
        return pd.DataFrame()

    df_live = pd.read_csv(live_path)
    df_live = df_live[df_live["status"] == "FINISHED"].copy()

    if df_live.empty:
        return pd.DataFrame()

    # Datum parsen (unterstützt reine Jahreszahlen und ISO-Strings)
    df_live["date"] = pd.to_datetime(
        df_live["date"].astype(str), errors="coerce", utc=True
    ).dt.tz_convert(None)

    # Zeilen mit fehlenden Scores oder ungültigem Datum verwerfen
    df_live = df_live.dropna(subset=["date", "home_score", "away_score"])

    return df_live[["date", "home_team", "away_team", "home_score", "away_score"]]


def get_latest_team_stats(
    team: str,
    df: pd.DataFrame,
    df_live: pd.DataFrame | None = None,
) -> tuple[float, float, float]:
    """Berechnet die aktuelle Form eines Teams aus seinen letzten 5 Spielen.

    Kombiniert die aufbereiteten Historien-Daten (cleaned_wm_data.csv) mit
    frischen Live-Spielen aus wm2026_live.csv. Dadurch fließt ein gerade
    gespieltes WM-Gruppenspiel direkt in die nächste Vorhersage ein – auch
    ohne einen neuen prepare_data.py-Lauf.

    Marktwerte werden weiterhin aus den aufbereiteten Historien-Daten bezogen,
    da die Live-CSV keine Kader-Marktwerte enthält.

    Args:
        team (str): Der Name der Nationalmannschaft.
        df (pd.DataFrame): Aufbereiteter Datensatz (cleaned_wm_data.csv).
        df_live (pd.DataFrame | None): Frische FINISHED-Spiele aus der Live-CSV.

    Returns:
        tuple: (form_attack, form_defense, market_value)
    """
    # Live-Spiele deduplizieren: nur Zeilen, die noch nicht in df stehen
    if df_live is not None and not df_live.empty:
        existing_keys = set(zip(df["home_team"], df["away_team"]))
        new_live = df_live[
            ~df_live.apply(
                lambda r: (r["home_team"], r["away_team"]) in existing_keys,
                axis=1,
            )
        ]
        if not new_live.empty:
            # Für die Formberechnung reichen die Kernspalten
            df_all = pd.concat(
                [df[["date", "home_team", "away_team", "home_score", "away_score"]], new_live],
                ignore_index=True,
            ).sort_values("date").reset_index(drop=True)
        else:
            df_all = df
    else:
        df_all = df

    # Filtere alle Spiele des Teams über den kombinierten Datensatz
    team_df_all = df_all[
        (df_all["home_team"] == team) | (df_all["away_team"] == team)
    ].sort_values("date")

    # Fallback für unbekannte Teams
    if team_df_all.empty:
        median_market_value = df["home_market_value"].median()
        return 1.3, 1.3, float(median_market_value)

    # Tore aus Sicht des Teams sammeln (egal ob Heim oder Auswärts)
    scored, conceded = [], []
    for _, row in team_df_all.iterrows():
        if row["home_team"] == team:
            scored.append(row["home_score"])
            conceded.append(row["away_score"])
        else:
            scored.append(row["away_score"])
            conceded.append(row["home_score"])

    # Form = Durchschnitt der letzten 5 Spiele (inkl. Live-Ergebnisse!)
    form_attack = float(np.mean(scored[-5:]))
    form_defense = float(np.mean(conceded[-5:]))

    # Marktwert: aus den aufbereiteten Historien-Daten (haben Kader-Werte)
    team_df_historical = df[
        (df["home_team"] == team) | (df["away_team"] == team)
    ].sort_values("date")

    if team_df_historical.empty:
        market_value = float(df["home_market_value"].median())
    else:
        last_game = team_df_historical.iloc[-1]
        if last_game["home_team"] == team:
            market_value = float(last_game["home_market_value"])
        else:
            market_value = float(last_game["away_market_value"])

    return form_attack, form_defense, market_value


def main():
    """Hauptfunktion zur Validierung der Argumente und Ausführung der Inferenz."""
    if len(sys.argv) < 3:
        print("💡 Nutzung im Terminal: python predict.py '<Heimteam>' '<Auswärtsteam>'")
        print("   Beispiel: python predict.py 'Germany' 'France'")
        sys.exit(1)

    home_team = sys.argv[1]
    away_team = sys.argv[2]

    # Offizielle Gastgeber-Liste der WM 2026 zur automatischen Feature-Injektion
    hosts_2026 = ["USA", "United States", "Canada", "Mexico"]

    # Laden der serialisierten Modelle und Daten
    try:
        model_home = joblib.load("models/wm_home_goals_model.pkl")
        model_away = joblib.load("models/wm_away_goals_model.pkl")
        df = pd.read_csv("data/processed/cleaned_wm_data.csv")
        # Datum als Timestamp parsen, damit es mit Live-Daten sortierbar ist
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except FileNotFoundError as e:
        print(f"❌ Fehler: Wichtige Projektdateien fehlen ({e}).")
        print("   Bitte führe zuerst prepare_data.py und train.py aus.")
        sys.exit(1)

    # Live-Spiele direkt zur Laufzeit einlesen (kein prepare_data.py-Lauf nötig)
    try:
        with open("config.yaml", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)
        live_path = _cfg["tournaments"]["world_cup"]["live_path"]
    except Exception:
        live_path = "data/raw/wm2026_live.csv"

    df_live = load_live_finished_games(live_path)
    live_count = len(df_live) if not df_live.empty else 0
    if live_count:
        print(f"📡 {live_count} neue(s) Live-Spiel(e) direkt aus '{live_path}' geladen.")

    # Features der beiden Teams ermitteln (inkl. aktueller Live-Ergebnisse)
    home_attack, home_defense, home_val = get_latest_team_stats(home_team, df, df_live)
    away_attack, away_defense, away_val = get_latest_team_stats(away_team, df, df_live)

    # Input-DataFrame exakt in der Struktur aufbauen, wie es XGBoost erwartet
    input_data = pd.DataFrame([{
        "home_form_attack": home_attack,
        "home_form_defense": home_defense,
        "away_form_attack": away_attack,
        "away_form_defense": away_defense,
        "home_market_value": home_val,
        "away_market_value": away_val,
        "home_is_host": 1 if home_team in hosts_2026 else 0,
        "away_is_host": 1 if away_team in hosts_2026 else 0,
        "neutral": 1
    }])

    # Kontinuierliche Tor-Erwartung berechnen (Negative Werte via max(0, x) abfangen)
    pred_home_goals = max(0.0, float(model_home.predict(input_data)[0]))
    pred_away_goals = max(0.0, float(model_away.predict(input_data)[0]))

    # Formatiertes Terminal-Dashboard ausgeben
    print("\n" + "=" * 55)
    print(f"🔮 PREDICTION ENGINE — WM 2026 SIMULATION:")
    print(f"   {home_team} vs. {away_team}")
    print("=" * 55)
    print(f"💰 Kaderwert {home_team:12}: {home_val:6.1f}M € | Form-Angriff: {home_attack:.2f}")
    print(f"💰 Kaderwert {away_team:12}: {away_val:6.1f}M € | Form-Angriff: {away_attack:.2f}")
    print("-" * 55)
    print(f"⚽ Erwartete Tore {home_team}: {pred_home_goals:.2f}")
    print(f"⚽ Erwartete Tore {away_team}: {pred_away_goals:.2f}")
    print("-" * 55)
    print(f"🏆 Mathematischer Tipp: {home_team} {round(pred_home_goals)} - {round(pred_away_goals)} {away_team}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
