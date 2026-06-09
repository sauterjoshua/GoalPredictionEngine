"""
run_pipeline.py

Zentraler Orchestrator für die WM-2026-PredictionEngine.
Reihenfolge: Live-Daten holen → aufbereiten → trainieren → visualisieren
             → Batch-Vorhersagen → Telegram-Push.
"""

import subprocess
import sys
import time

from predict import run_batch_prediction
from notify import run_notification

try:
    from data_sources import fetch_world_cup_2026_matches, save_matches_to_csv
except ImportError:
    from wm_pipeline.data_sources import fetch_world_cup_2026_matches, save_matches_to_csv


def fetch_live_data() -> None:
    """Schritt 0: Aktuelle WM-Daten von football-data.org holen.

    Schlägt die API fehl (kein Token, Rate-Limit, Netzwerk), wird
    mit einer Warnung fortgefahren – prepare_data.py nutzt dann die
    bereits vorhandene wm2026_live.csv oder überspringt den Merge.
    """
    print("\n🌐 [PIPELINE] Starte: Live-Daten-Abruf von football-data.org...")
    try:
        from data_sources import (
            fetch_world_cup_2026_matches,
            save_matches_to_csv,
        )
        df = fetch_world_cup_2026_matches()
        save_matches_to_csv(df)
        print("✅ [PIPELINE] Live-Daten aktualisiert!")
    except Exception as exc:
        print(f"⚠️  [PIPELINE] Live-Daten-Abruf fehlgeschlagen: {exc}")
        print("   Weiter mit vorhandener wm2026_live.csv (oder ohne, falls nicht existent).")


def run_script(script_name: str) -> None:
    """Führt ein Python-Skript als Subprozess aus und fängt Fehler ab."""
    print(f"\n🚀 [PIPELINE] Starte: {script_name}...")
    start_time = time.time()

    result = subprocess.run([sys.executable, script_name], capture_output=False)

    if result.returncode != 0:
        print(f"❌ [PIPELINE] Fehler in {script_name}! Pipeline abgebrochen.")
        sys.exit(result.returncode)

    duration = time.time() - start_time
    print(f"✅ [PIPELINE] {script_name} erfolgreich beendet! (Dauer: {duration:.2f}s)")


def main() -> None:
    print("=" * 60)
    print("🏁 START DER AUTOMATISCHEN END-TO-END PIPELINE")
    print("=" * 60)

    global_start = time.time()

    # --- Schritt 0: Frische Live-Daten von der API holen -------------------
    fetch_live_data()

    # --- Schritte 1–3: Kern-Skripte sequentiell ----------------------------
    run_script("prepare_data.py")
    run_script("train.py")
    run_script("visualize.py")

    # --- Schritt 4: Batch-Vorhersage (Import statt Subprozess, da predict.py
    #     als CLI nur Einzelvorhersagen macht; wir brauchen run_batch_prediction) ---
    print("\n🚀 [PIPELINE] Starte: Batch-Vorhersage...")
    run_batch_prediction()   # schreibt predictions.csv + prediction_history.csv
    print("✅ [PIPELINE] Batch-Vorhersage beendet!")

    # --- Schritt 5: Telegram-Push ------------------------------------------
    print("\n🚀 [PIPELINE] Starte: Telegram-Benachrichtigung...")
    run_notification()       # Leer-Fall (keine Spiele) = stille Nicht-Sendung
    print("✅ [PIPELINE] Benachrichtigung beendet!")

    total_duration = time.time() - global_start
    print("\n" + "=" * 60)
    print(f"🎉 PIPELINE ERFOLGREICH DURCHGELAUFEN! Gesamtzeit: {total_duration:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()