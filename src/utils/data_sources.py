"""Live-Daten von football-data.org für WM 2026 Spielergebnisse."""

import os
import requests
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

FOOTBALL_DATA_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")
API_BASE_URL = "https://api.football-data.org/v4"


def fetch_world_cup_2026_matches() -> pd.DataFrame:
    """Holt alle WM-2026-Spiele von football-data.org.

    Vor WM-Start: nur geplante Spiele ohne Scores.
    Während/nach WM: Scores sind gefüllt.

    Returns:
        DataFrame mit Spalten: date, home_team, away_team, home_score, away_score, tournament, status, stage
    """
    if not FOOTBALL_DATA_TOKEN:
        raise ValueError(
            "Kein FOOTBALL_DATA_TOKEN gefunden! "
            "Bitte .env-Datei im Projekt-Root anlegen."
        )

    url = f"{API_BASE_URL}/competitions/WC/matches"
    headers = {"X-Auth-Token": FOOTBALL_DATA_TOKEN}

    print(f"🌐 Hole WM-Daten von {url}")
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise RuntimeError(
            f"API-Fehler: HTTP {response.status_code} — {response.text}"
        )

    data = response.json()
    matches = data.get("matches", [])
    print(f"   ✅ {len(matches)} Spiele erhalten")

    # Transformation in DataFrame mit deinem gewohnten Schema
    rows = []
    for match in matches:
        # Score-Extraktion: kann None sein für noch nicht gespielte Spiele
        score = match.get("score", {}).get("fullTime", {})
        home_score = score.get("home")
        away_score = score.get("away")

        rows.append({
            "date": match.get("utcDate"),
            "home_team": match.get("homeTeam", {}).get("name"),
            "away_team": match.get("awayTeam", {}).get("name"),
            "home_score": home_score,
            "away_score": away_score,
            "tournament": "FIFA World Cup",
            "status": match.get("status"),   # SCHEDULED, FINISHED, IN_PLAY…
            "stage": match.get("stage"),     # GROUP_STAGE, LAST_16, QUARTER_FINALS…
        })

    df = pd.DataFrame(rows)
    print(f"   📊 DataFrame: {len(df)} Zeilen, {len(df.columns)} Spalten")
    return df


def save_matches_to_csv(df: pd.DataFrame, output_path: str = "data/raw/wm2026_live.csv") -> str:
    """
    Speichert die WM-2026-Daten als CSV.
    
    Existiert bereits eine Datei, wird sie überschrieben — wir wollen immer
    den aktuellsten Stand. Historische Snapshots könnten wir später ergänzen.
    """
    # Sicherstellen, dass das Verzeichnis existiert
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"💾 Gespeichert: {output_path} ({len(df)} Spiele)")
    return output_path


if __name__ == "__main__":
    df = fetch_world_cup_2026_matches()
    save_matches_to_csv(df)
    print("\n--- Erste 5 Zeilen ---")
    print(df.head())
    print(f"\n--- Status-Verteilung ---")
    print(df["status"].value_counts())