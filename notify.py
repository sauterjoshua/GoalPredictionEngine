"""
Verschickt die aktuellen WM-2026-Vorhersagen per Telegram-Bot.

Benötigt in der .env:
    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_CHAT_ID=...

CLI-Test: python notify.py
"""

import os

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv


# TODO: in gemeinsames flags.py auslagern (Duplikat von dashboard.py)
COUNTRY_TO_ISO = {
    "Germany": "DE", "France": "FR", "Spain": "ES", "Italy": "IT",
    "England": "GB", "Brazil": "BR", "Argentina": "AR", "Portugal": "PT",
    "Netherlands": "NL", "Belgium": "BE", "Croatia": "HR", "USA": "US",
    "United States": "US", "Canada": "CA", "Mexico": "MX", "Japan": "JP",
    "South Korea": "KR", "Curaçao": "CW", "Curacao": "CW", "Ecuador": "EC",
    "Ivory Coast": "CI", "Morocco": "MA", "Senegal": "SN", "Ghana": "GH",
    "Uruguay": "UY", "Colombia": "CO", "Switzerland": "CH", "Denmark": "DK",
    "Poland": "PL", "Austria": "AT", "Australia": "AU", "Qatar": "QA",
    "Saudi Arabia": "SA", "Nigeria": "NG", "Cameroon": "CM", "Serbia": "RS",
    "Norway": "NO", "Sweden": "SE", "Turkey": "TR", "Egypt": "EG",
    "Czechia": "CZ", "Paraguay": "PY", "Scotland": "GB", "Haiti": "HT",
    "Bosnia-Herzegovina": "BA", "Congo DR": "CD", "Cape Verde Islands": "CV",
}


def flag_emoji(country: str) -> str:
    """Wandelt einen Ländernamen in ein Flaggen-Emoji (Fallback: weiße Flagge)."""
    iso = COUNTRY_TO_ISO.get(country)
    if not iso:
        return "🏳️"
    return "".join(chr(0x1F1E6 + (ord(c) - ord("A"))) for c in iso.upper())


def load_predictions_path() -> str:
    """Liest den Pfad zur predictions.csv aus der config.yaml."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["tournaments"]["world_cup"]["predictions_path"]


def build_message(df: pd.DataFrame) -> str:
    """Baut die formatierte Telegram-Nachricht (HTML-Modus) aus den Vorhersagen."""
    lines = ["<b>⚽ WM 2026 — Tipps für die nächsten Spiele</b>", ""]

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hf, af = flag_emoji(home), flag_emoji(away)
        tipp = f"{int(row['tipp_home'])}:{int(row['tipp_away'])}"
        exact = f"{row['pred_home_goals']:.2f}–{row['pred_away_goals']:.2f}"
        lines.append(
            f"{hf} {home} vs {away} {af}\n"
            f"   Tipp: <b>{tipp}</b>  (exakt {exact})  · {row['date']}"
        )

    return "\n".join(lines)


def send_telegram_message(text: str) -> bool:
    """Schickt eine Nachricht über die Telegram-Bot-API (Token + Chat-ID aus .env)."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("❌ TELEGRAM_BOT_TOKEN oder TELEGRAM_CHAT_ID fehlt in der .env.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
        print("✅ Telegram-Nachricht erfolgreich gesendet.")
        return True
    except requests.RequestException as e:
        print(f"❌ Fehler beim Telegram-Versand: {e}")
        # Body kann den genauen Telegram-Fehler enthalten (z.B. falsche chat_id)
        if e.response is not None:
            print(f"   Antwort: {e.response.text}")
        return False


def run_notification() -> int:
    """Lädt die aktuellen Vorhersagen und verschickt sie per Telegram.

    Returns:
        int: Anzahl benachrichtigter Spiele (0 = nichts gesendet).
    """
    load_dotenv()

    predictions_path = load_predictions_path()

    try:
        df = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print(f"ℹ️ Keine predictions.csv unter '{predictions_path}'. Nichts zu senden.")
        return 0

    if df.empty:
        print("ℹ️ Keine Spiele im Vorhersage-Fenster. Keine Benachrichtigung.")
        return 0

    message = build_message(df)
    success = send_telegram_message(message)
    return len(df) if success else 0


if __name__ == "__main__":
    n = run_notification()
    print(f"Fertig. {n} Spiel(e) benachrichtigt.")