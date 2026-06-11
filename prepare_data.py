"""
Data-Engineering-Pipeline: harmonisiert Rohdaten, berechnet Formkurven,
aggregiert Kader-Marktwerte und injiziert den WM-Gastgeber-Heimvorteil.
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml

from form import compute_form

MARKET_VALUE_NAME_MAP = {
    "Czech Republic": "Czechia",
    "DR Congo": "Congo DR",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Cape Verde": "Cape Verde Islands",
    "Cura?o": "Curaçao",
}

HISTORY_NAME_MAP = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Cape Verde": "Cape Verde Islands",
    "DR Congo": "Congo DR",
    "Czech Republic": "Czechia",
}


def load_config() -> dict:
    """Lädt die zentrale Konfigurationsdatei (config.yaml)."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_live_data(df_historical: pd.DataFrame, live_path: str) -> pd.DataFrame:
    """Merged abgeschlossene Live-Spiele aus einer separaten CSV in den historischen DataFrame."""
    if not os.path.exists(live_path):
        print(f"⚠️  Live-Datei nicht gefunden ('{live_path}'). Überspringe Merge.")
        return df_historical

    if os.path.getsize(live_path) == 0:
        print(f"⚠️  Live-Datei ist leer ('{live_path}'). Überspringe Merge.")
        return df_historical

    df_live = pd.read_csv(live_path)
    df_live = df_live[df_live["status"] == "FINISHED"]

    if df_live.empty:
        print("⚠️  Keine abgeschlossenen Spiele in der Live-CSV. Überspringe Merge.")
        return df_historical

    if "status" in df_live.columns:
        df_live = df_live.drop(columns=["status"])

    print(f"🔀 Merge {len(df_live)} Live-Spiel(e) aus '{live_path}'...")
    return pd.concat([df_historical, df_live], ignore_index=True)


def harmony_columns(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Harmonisiert unterschiedliche Spaltennamen aus Rohdaten auf ein einheitliches Schema."""
    print("🔍 Analysiere und harmonisiere Spaltenstruktur...")

    mapping = {}
    for col in ["date", "Datetime", "Year", "year", "Date"]:
        if col in df_matches.columns:
            mapping[col] = "date"
            break
    for col in ["home_team", "Home Team Name", "HomeTeam", "Home Team"]:
        if col in df_matches.columns:
            mapping[col] = "home_team"
            break
    for col in ["away_team", "Away Team Name", "AwayTeam", "Away Team"]:
        if col in df_matches.columns:
            mapping[col] = "away_team"
            break
    for col in ["home_score", "Home Team Goals", "FTHG", "Home Goals"]:
        if col in df_matches.columns:
            mapping[col] = "home_score"
            break
    for col in ["away_score", "Away Team Goals", "FTAG", "Away Goals"]:
        if col in df_matches.columns:
            mapping[col] = "away_score"
            break

    df_matches = df_matches.rename(columns=mapping)

    required_cols = ["date", "home_team", "away_team", "home_score", "away_score"]
    missing_cols = [col for col in required_cols if col not in df_matches.columns]

    if missing_cols:
        print(f"❌ Fehler: Kernspalten konnten nicht zugeordnet werden: {missing_cols}")
        sys.exit(1)

    print("   ✅ Spalten erfolgreich harmonisiert!")
    return df_matches


def convert_dates(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Konvertiert die Datumsspalte – unterstützt Jahreszahlen (2022) und ISO-Strings."""
    print("📅 Konvertiere Datumsspalte (gemischte Formate möglich)...")
    df_matches["date"] = pd.to_datetime(
        df_matches["date"].astype(str), errors="coerce", utc=True
    ).dt.tz_convert(None)

    df_matches = df_matches.sort_values("date").reset_index(drop=True)
    return df_matches.dropna(subset=["home_score", "away_score", "home_team", "away_team"])


def calculate_form_curves(df_matches: pd.DataFrame, median_market_value: float) -> pd.DataFrame:
    """Berechnet Formkurven via gemeinsame compute_form()-Funktion (form.py).

    Phase B (Spielart) + Phase C (Gegner-MV) sind beide aktiv.
    Voraussetzung: df_matches enthält bereits 'home_market_value' / 'away_market_value'.
    """
    print("📈 Berechne historische Formkurven (Rolling Window: 5, gewichtet)...")
    team_stats = {}
    home_form_attack, home_form_defense = [], []
    away_form_attack, away_form_defense = [], []

    has_tournament = "tournament" in df_matches.columns
    has_mv = "home_market_value" in df_matches.columns and "away_market_value" in df_matches.columns

    for _, row in df_matches.iterrows():
        home, away = row["home_team"], row["away_team"]

        for team in (home, away):
            if team not in team_stats:
                team_stats[team] = {
                    "scored": [], "conceded": [],
                    "tournaments": [], "opponent_mvs": [],
                }

        # Form VOR dem Spiel (gemeinsame Helper-Funktion → kein Training-Serving-Skew)
        h_atk, h_def = compute_form(
            team_stats[home]["scored"],
            team_stats[home]["conceded"],
            team_stats[home]["tournaments"],
            team_stats[home]["opponent_mvs"],
            median_market_value,
        )
        a_atk, a_def = compute_form(
            team_stats[away]["scored"],
            team_stats[away]["conceded"],
            team_stats[away]["tournaments"],
            team_stats[away]["opponent_mvs"],
            median_market_value,
        )
        home_form_attack.append(h_atk)
        home_form_defense.append(h_def)
        away_form_attack.append(a_atk)
        away_form_defense.append(a_def)

        # Ergebnis NACH dem Spiel in den State pushen
        tourn = row["tournament"] if has_tournament else None
        home_mv = float(row["home_market_value"]) if has_mv else None
        away_mv = float(row["away_market_value"]) if has_mv else None

        team_stats[home]["scored"].append(row["home_score"])
        team_stats[home]["conceded"].append(row["away_score"])
        team_stats[home]["tournaments"].append(tourn)
        team_stats[home]["opponent_mvs"].append(away_mv)  # Gegner-MV aus home-Sicht

        team_stats[away]["scored"].append(row["away_score"])
        team_stats[away]["conceded"].append(row["home_score"])
        team_stats[away]["tournaments"].append(tourn)
        team_stats[away]["opponent_mvs"].append(home_mv)  # Gegner-MV aus away-Sicht

    df_matches["home_form_attack"] = home_form_attack
    df_matches["home_form_defense"] = home_form_defense
    df_matches["away_form_attack"] = away_form_attack
    df_matches["away_form_defense"] = away_form_defense

    return df_matches


def inject_host_advantage(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Setzt home_is_host / away_is_host = 1, falls ein Team echter WM-Gastgeber war."""
    print("🏠 Injiziere echten Heimvorteil (Identifikation der WM-Gastgeber)...")

    world_cup_hosts = {
        2002: ["South Korea", "Japan"],
        2006: ["Germany"],
        2010: ["South Africa"],
        2014: ["Brazil"],
        2018: ["Russia"],
        2022: ["Qatar"],
        2026: ["USA", "United States", "Canada", "Mexico"],
    }

    home_is_host = []
    away_is_host = []

    for _, row in df_matches.iterrows():
        year = row["date"].year
        home_team = row["home_team"]
        away_team = row["away_team"]

        if year in world_cup_hosts and home_team in world_cup_hosts[year]:
            home_is_host.append(1)
        else:
            home_is_host.append(0)

        if year in world_cup_hosts and away_team in world_cup_hosts[year]:
            away_is_host.append(1)
        else:
            away_is_host.append(0)

    df_matches["home_is_host"] = home_is_host
    df_matches["away_is_host"] = away_is_host
    return df_matches


def aggregate_market_values(df_market: pd.DataFrame) -> pd.DataFrame:
    """Extrahiert Kader-Gesamtmarktwerte aus wm_dataset.csv (bereits pro Team aggregiert)."""
    print("💰 Extrahiere Kader-Marktwerte pro Nation (in Mio €)...")

    squad_values = df_market[["team", "squad_total_market_value_eur"]].copy()
    squad_values["team"] = squad_values["team"].replace(MARKET_VALUE_NAME_MAP)

    squad_values["squad_market_value"] = (
        squad_values["squad_total_market_value_eur"] / 1_000_000
    ).round(1)

    squad_values = squad_values[["team", "squad_market_value"]].rename(
        columns={"team": "team_name"}
    )

    squad_values = squad_values.groupby("team_name", as_index=False)["squad_market_value"].max()

    print(f"   ✅ {len(squad_values)} Nationen mit Marktwert geladen.")
    return squad_values


def process_data():
    """Hauptfunktion zur Orchestrierung der gesamten Daten-Pipeline."""
    config = load_config()
    raw_path = config["tournaments"]["world_cup"]["raw_path"]
    processed_path = config["tournaments"]["world_cup"]["processed_path"]
    live_path = config["tournaments"]["world_cup"]["live_path"]
    market_value_path = config["tournaments"]["world_cup"]["team_features_path"]
    match_history_path = config["tournaments"]["world_cup"]["match_history_path"]

    print(f"⏳ Lade Match-Daten aus '{raw_path}'...")
    df_matches = pd.read_csv(raw_path)
    df_matches = merge_live_data(df_matches, live_path)

    print(f"⏳ Lade Marktwert-Daten aus '{market_value_path}'...")
    try:
        df_market = pd.read_csv(market_value_path)
    except FileNotFoundError:
        print(f"❌ Fehler: Die erforderliche Datei '{market_value_path}' fehlt.")
        sys.exit(1)

    df_matches = harmony_columns(df_matches)
    df_matches = convert_dates(df_matches)

    # Quell-Schreibweisen normalisieren (MUSS vor Marktwert-Join laufen)
    df_matches["home_team"] = df_matches["home_team"].replace(HISTORY_NAME_MAP)
    df_matches["away_team"] = df_matches["away_team"].replace(HISTORY_NAME_MAP)

    # tournament-Spalte garantieren + befüllen (für Phase B)
    if "tournament" not in df_matches.columns:
        df_matches["tournament"] = "FIFA World Cup"
    df_matches["tournament"] = df_matches["tournament"].fillna("FIFA World Cup")

    # --- REIHENFOLGE GEÄNDERT: Marktwerte VOR den Formkurven mergen, ---
    # --- damit calculate_form_curves() den Gegner-MV pro Spiel kennt (Phase C). ---
    squad_values = aggregate_market_values(df_market)
    median_value = float(squad_values["squad_market_value"].median())
    print(f"   📊 Median-Marktwert (für Gegner-Faktor): {median_value:.1f}M €")

    print("🔀 Führe Match-Daten und aggregierte Kaderwerte zusammen...")
    df_matches = df_matches.merge(
        squad_values, left_on="home_team", right_on="team_name", how="left"
    ).drop(columns=["team_name"]).rename(columns={"squad_market_value": "home_market_value"})
    df_matches = df_matches.merge(
        squad_values, left_on="away_team", right_on="team_name", how="left"
    ).drop(columns=["team_name"]).rename(columns={"squad_market_value": "away_market_value"})

    df_matches["home_market_value"] = df_matches["home_market_value"].fillna(median_value)
    df_matches["away_market_value"] = df_matches["away_market_value"].fillna(median_value)

    # Jetzt erst Formkurven (mit Zugriff auf Gegner-MV)
    df_matches = calculate_form_curves(df_matches, median_market_value=median_value)
    df_matches = inject_host_advantage(df_matches)

    # --- Serving-Feature-Quelle: ungefilterte Match-Historie (alle Spielarten) ---
    history_cols = [
        "date", "home_team", "away_team",
        "home_score", "away_score",
        "home_market_value", "away_market_value",
        "tournament",
    ]
    os.makedirs(os.path.dirname(match_history_path), exist_ok=True)
    df_matches[history_cols].to_csv(match_history_path, index=False)
    print(f"💾 Serving-Historie: '{match_history_path}' ({len(df_matches)} Spiele)")

    # --- Trainings-Tabelle: nur FIFA World Cup ab 2000 ---
    df_wc = df_matches[df_matches["tournament"] == "FIFA World Cup"].copy()
    df_wc = df_wc[df_wc["date"].dt.year >= 2000].reset_index(drop=True)

    df_wc["neutral"] = 1

    final_features = [
        "date",
        "home_team",
        "away_team",
        "home_form_attack",
        "home_form_defense",
        "away_form_attack",
        "away_form_defense",
        "home_market_value",
        "away_market_value",
        "home_is_host",
        "away_is_host",
        "neutral",
        "home_score",
        "away_score",
    ]

    cleaned_df = df_wc[final_features]
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)
    print(f"\n✅ Pipeline erfolgreich durchgelaufen! Datei gespeichert unter: '{processed_path}'")


if __name__ == "__main__":
    process_data()