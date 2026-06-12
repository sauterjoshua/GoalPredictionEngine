"""
run_pipeline.py

Zentraler Orchestrator für die WM-2026-PredictionEngine.
Reihenfolge: Live-Daten holen → aufbereiten → trainieren → visualisieren
             → Batch-Vorhersagen → Telegram-Push.
"""

import sys
import time

from src.utils.data_sources import fetch_world_cup_2026_matches, save_matches_to_csv
from src.pipeline.prepare_data import process_data
from src.pipeline.train import main as train_main
from src.pipeline.visualize import main as visualize_main
from src.pipeline.predict import run_batch_prediction
from src.delivery.notify import run_notification


def fetch_live_data() -> None:
    """Schritt 0: Aktuelle WM-Daten von football-data.org holen.

    Schlägt die API fehl (kein Token, Rate-Limit, Netzwerk), wird
    mit einer Warnung fortgefahren – prepare_data.py nutzt dann die
    bereits vorhandene wm2026_live.csv oder überspringt den Merge.
    """
    print("\n🌐 [PIPELINE] Starte: Live-Daten-Abruf von football-data.org...")
    try:
        df = fetch_world_cup_2026_matches()
        save_matches_to_csv(df)
        print("✅ [PIPELINE] Live-Daten aktualisiert!")
    except Exception as exc:
        print(f"⚠️  [PIPELINE] Live-Daten-Abruf fehlgeschlagen: {exc}")
        print("   Weiter mit vorhandener wm2026_live.csv (oder ohne, falls nicht existent).")


def run_step(name: str, fn) -> None:
    """Führt einen Pipeline-Schritt aus und misst die Laufzeit."""
    print(f"\n🚀 [PIPELINE] Starte: {name}...")
    start_time = time.time()
    try:
        fn()
    except Exception as exc:
        print(f"❌ [PIPELINE] Fehler in {name}: {exc}")
        sys.exit(1)
    duration = time.time() - start_time
    print(f"✅ [PIPELINE] {name} erfolgreich beendet! (Dauer: {duration:.2f}s)")


def main() -> None:
    print("=" * 60)
    print("🏁 START DER AUTOMATISCHEN END-TO-END PIPELINE")
    print("=" * 60)

    global_start = time.time()

    fetch_live_data()

    run_step("prepare_data", process_data)
    run_step("train", train_main)
    run_step("visualize", visualize_main)

    # Batch-Vorhersage schreibt predictions.csv + prediction_history.csv
    run_step("predict", run_batch_prediction)

    # Leer-Fall (keine Spiele) = stille Nicht-Sendung
    run_step("notify", run_notification)

    total_duration = time.time() - global_start
    print("\n" + "=" * 60)
    print(f"🎉 PIPELINE ERFOLGREICH DURCHGELAUFEN! Gesamtzeit: {total_duration:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
