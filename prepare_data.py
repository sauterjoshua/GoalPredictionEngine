"""
prepare_data.py

Dieses Skript bildet die Data-Engineering-Pipeline des Projekts.
Es lädt die Rohdaten, bereinigt und harmonisiert die Spaltenstrukturen,
berechnet dynamische Formkurven und aggregiert die Spieler-Marktwerte
zu Kader-Gesamtmarktwerten. Zudem wird ein Feature für den echten 
Heimvorteil von WM-Gastgebern injiziert.
"""

import sys
import numpy as np
import pandas as pd
import yaml


def load_config() -> dict:
    """Lädt die zentrale Konfigurationsdatei (config.yaml).

    Returns:
        dict: Die geladenen Konfigurationsparameter.
    """
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def harmony_columns(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Harmonisiert unterschiedliche Spaltennamen aus historischen Rohdaten

    auf ein einheitliches, intern definiertes Namensschema.
    """
    print("🔍 Analysiere und harmonisiere Spaltenstruktur...")

    # Such-Mapping für potenzielle Spaltennamen in den Rohdaten
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

    # Validierung der zwingend erforderlichen Spalten
    required_cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]
    missing_cols = [
        col for col in required_cols if col not in df_matches.columns
    ]

    if missing_cols:
        print(
            f"❌ Fehler: Kernspalten konnten nicht zugeordnet werden: {missing_cols}"
        )
        sys.exit(1)

    print("   ✅ Spalten erfolgreich harmonisiert!")
    return df_matches


def convert_dates(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Konvertiert die Datumsspalte intelligent – unabhängig davon, ob es sich

    um reine Jahreszahlen oder vollständige Datums-Strings handelt.
    """
    sample_date = (
        str(df_matches["date"].dropna().iloc[0]).split(".")[0].strip()
    )

    if sample_date.isdigit() and len(sample_date) == 4:
        print(
            "📅 Format-Erkennung: Reine Jahreszahlen (z.B. 2022). Setze fixe Zeitachse..."
        )
        df_matches["date"] = pd.to_datetime(
            df_matches["date"].astype(int).astype(str) + "-01-01",
            errors="coerce",
        )
    else:
        print("📅 Format-Erkennung: Vollständige Datums-Strings. Konvertiere...")
        df_matches["date"] = pd.to_datetime(df_matches["date"], errors="coerce")

    # Sortieren und Zeilen ohne essenzielle Match-Daten entfernen
    df_matches = df_matches.sort_values("date").reset_index(drop=True)
    return df_matches.dropna(
        subset=["home_score", "away_score", "home_team", "away_team"]
    )


def calculate_form_curves(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Berechnet dynamische Formkurven (Rolling Average der letzten 5 Spiele)

    über alle historisch verfügbaren Länderspiele hinweg.
    """
    print("📈 Berechne historische Formkurven (Rolling Window: 5)...")
    team_stats = {}
    home_form_attack, home_form_defense = [], []
    away_form_attack, away_form_defense = [], []

    for _, row in df_matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Initialisiere Teams, falls sie zum ersten Mal auftauchen
        for team in [home, away]:
            if team not in team_stats:
                team_stats[team] = {"scored": [], "conceded": []}

        # Form vor dem Spiel sichern (Fallback auf Turnierschnitt 1.3 bei Debüt)
        home_form_attack.append(
            np.mean(team_stats[home]["scored"][-5:])
            if team_stats[home]["scored"]
            else 1.3
        )
        home_form_defense.append(
            np.mean(team_stats[home]["conceded"][-5:])
            if team_stats[home]["conceded"]
            else 1.3
        )
        away_form_attack.append(
            np.mean(team_stats[away]["scored"][-5:])
            if team_stats[away]["scored"]
            else 1.3
        )
        away_form_defense.append(
            np.mean(team_stats[away]["conceded"][-5:])
            if team_stats[away]["conceded"]
            else 1.3
        )

        # Historie mit den aktuellen Spielergebnissen aktualisieren
        team_stats[home]["scored"].append(row["home_score"])
        team_stats[home]["conceded"].append(row["away_score"])
        team_stats[away]["scored"].append(row["away_score"])
        team_stats[away]["conceded"].append(row["home_score"])

    df_matches["home_form_attack"] = home_form_attack
    df_matches["home_form_defense"] = home_form_defense
    df_matches["away_form_attack"] = away_form_attack
    df_matches["away_form_defense"] = away_form_defense

    return df_matches


def inject_host_advantage(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Injiziert ein binäres Flag (1 oder 0), falls ein Team der tatsächliche

    Gastgeber der jeweiligen Weltmeisterschaft war oder sein wird.
    """
    print("🏠 Injiziere echten Heimvorteil (Identifikation der WM-Gastgeber)...")

    # Offizielle Ausrichterländer der Turniere (inkl. Dreiergespann für 2026)
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

        # Prüfen, ob das Team im entsprechenden Jahr echter Gastgeber war
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
    """Aggregiert Einzelspieler-Marktwerte zu einem Gesamt-Kaderwert.
    
    Verhindert Duplikate durch Mehrfachnennungen desselben Spielers (Klon-Armee-Schutz).
    """
    print("💰 Aggregiere Spieler-Marktwerte zu nationalen Kaderwerten (Top 26)...")

    # 1. Identifiziere die Spalte für den Spielernamen
    name_col = None
    for col in ["player_name", "short_name", "long_name", "name", "Player"]:
        if col in df_market.columns:
            name_col = col
            break

    # Erst nach Marktwert sortieren (höchste Werte nach oben)
    df_market_sorted = df_market.sort_values("market_value_million_eur", ascending=False)

    # 2. Wenn eine Namensspalte existiert, Duplikate desselben Spielers löschen
    if name_col:
        print(f"   ℹ️ Duplikate werden basierend auf der Spalte '{name_col}' entfernt...")
        df_market_sorted = df_market_sorted.drop_duplicates(subset=[name_col, "nationality"], keep="first")
    else:
        print("   ⚠️ Warnung: Keine Namensspalte gefunden. Werte könnten verzerrt sein.")

    # Jetzt erst die Top 26 Spieler pro Nation herausschneiden
    top_26_players = df_market_sorted.groupby("nationality").head(26)

    # Kumulierten Gesamtwert pro Nation berechnen
    squad_values = (
        top_26_players.groupby("nationality")["market_value_million_eur"]
        .sum()
        .reset_index()
    )
    squad_values['market_value_million_eur'] = squad_values['market_value_million_eur'] / 4
    squad_values.columns = ['team_name', 'squad_market_value']
    return squad_values


def process_data():
    """Hauptfunktion zur Orchestrierung der gesamten Daten-Pipeline."""
    config = load_config()
    raw_path = config["tournaments"]["world_cup"]["raw_path"]
    processed_path = config["tournaments"]["world_cup"]["processed_path"]
    market_value_path = "data/raw/fifa_player_performance_market_value.csv"

    # Datensätze einlesen
    print(f"⏳ Lade Match-Daten aus '{raw_path}'...")
    df_matches = pd.read_csv(raw_path)

    print(f"⏳ Lade Marktwert-Daten aus '{market_value_path}'...")
    try:
        df_market = pd.read_csv(market_value_path)
    except FileNotFoundError:
        print(
            f"❌ Fehler: Die erforderliche Datei '{market_value_path}' fehlt."
        )
        sys.exit(1)

    # Sequentieller Aufruf der Transformationsschritte
    df_matches = harmony_columns(df_matches)
    df_matches = convert_dates(df_matches)
    df_matches = calculate_form_curves(df_matches)
    df_matches = inject_host_advantage(df_matches)

    # Filtern auf moderne Epoche und das korrekte Turnier (Concept Drift Schutz)
    if "tournament" in df_matches.columns:
        df_wc = df_matches[df_matches["tournament"] == "FIFA World Cup"].copy()
    else:
        df_wc = df_matches.copy()
    df_wc = df_wc[df_wc["date"].dt.year >= 2000].reset_index(drop=True)

    # Marktwerte berechnen und in die WM-Spiele mergen
    squad_values = aggregate_market_values(df_market)

    print("🔀 Führe Match-Daten und aggregierte Kaderwerte zusammen...")
    df_wc = df_wc.merge(
        squad_values, left_on="home_team", right_on="team_name", how="left"
    ).drop(columns=["team_name"])
    df_wc = df_wc.rename(columns={"squad_market_value": "home_market_value"})

    df_wc = df_wc.merge(
        squad_values, left_on="away_team", right_on="team_name", how="left"
    ).drop(columns=["team_name"])
    df_wc = df_wc.rename(columns={"squad_market_value": "away_market_value"})

    # Fehlende Marktwerte (z.B. kleinere Fußball-Nationen) mit dem Median auffüllen
    median_value = squad_values["squad_market_value"].median()
    df_wc["home_market_value"] = df_wc["home_market_value"].fillna(median_value)
    df_wc["away_market_value"] = df_wc["away_market_value"].fillna(median_value)

    # Standardmäßig wird bei WMs von neutralem Boden ausgegangen
    df_wc["neutral"] = 1

    # Finale Feature-Selektion für das Modell-Training
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
    cleaned_df.to_csv(processed_path, index=False)
    print(
        f"\n✅ Pipeline erfolgreich durchgelaufen! Datei gespeichert unter: '{processed_path}'"
    )


if __name__ == "__main__":
    process_data()