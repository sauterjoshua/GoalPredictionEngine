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

from src.utils.flags import flag_emoji


def _safe_str(row: pd.Series, col: str, default: str) -> str:
    val = row.get(col, default)
    return default if pd.isna(val) else str(val)


def _safe_int(row: pd.Series, col: str, default: int = 0) -> int:
    val = row.get(col, default)
    try:
        return default if pd.isna(val) else int(val)
    except (TypeError, ValueError):
        return default


def _format_score(row: pd.Series) -> str:
    """Gibt den formatierten Tipp-String zurück — inkl. KO-Suffix falls vorhanden."""
    decided_by = _safe_str(row, "decided_by", "90min")
    tipp_home = int(row["tipp_home"])
    tipp_away = int(row["tipp_away"])

    if decided_by == "ET":
        total_home = tipp_home + _safe_int(row, "home_goals_et")
        total_away = tipp_away + _safe_int(row, "away_goals_et")
        return f"{total_home}:{total_away} (n.V.)"

    if decided_by == "penalties":
        winner = _safe_str(row, "predicted_winner", "")
        if winner == "home":
            winner_flag = flag_emoji(row["home_team"])
        elif winner == "away":
            winner_flag = flag_emoji(row["away_team"])
        else:
            winner_flag = "🏆"
        return f"{tipp_home}:{tipp_away} (i.E.) → {winner_flag}"

    return f"{tipp_home}:{tipp_away}"


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
        tipp = _format_score(row)
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