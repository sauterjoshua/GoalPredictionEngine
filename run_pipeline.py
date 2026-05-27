"""
run_pipeline.py

Zentraler Orchestrator für die CornerPredictionEngine.
Führt die Datenaufbereitung, das Modell-Training und die Visualisierung
automatisch nacheinander aus und überwacht die Pipeline-Integrität.
"""

import subprocess
import sys
import time

from predict import run_batch_prediction
from notify import run_notification


def run_script(script_name: str):
    """Führt ein Python-Skript als Subprozess aus und fängt Fehler ab."""
    print(f"\n🚀 [PIPELINE] Starte: {script_name}...")
    start_time = time.time()
    
    # Ausführen des Skripts mit dem aktuellen Python-Interpreter
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    
    # Wenn der Rückgabecode nicht 0 ist, gab es einen Fehler
    if result.returncode != 0:
        print(f"❌ [PIPELINE] Fehler in {script_name}! Pipeline abgebrochen.")
        sys.exit(result.returncode)
        
    duration = time.time() - start_time
    print(f"✅ [PIPELINE] {script_name} erfolgreich beendet! (Dauer: {duration:.2f}s)")


def main():
    print("=" * 60)
    print("🏁 START DER AUTOMATISCHEN END-TO-END PIPELINE")
    print("=" * 60)
    
    global_start = time.time()
    
    # Sequentieller Aufruf der drei Kern-Skripte
    run_script("prepare_data.py")
    run_script("train.py")
    run_script("visualize.py")
    
    # --- Inferenz + Benachrichtigung (Import statt Subprozess, da predict.py
    #     als CLI nur Einzelvorhersagen macht; wir brauchen die Batch-Funktion) ---
    print("\n🚀 [PIPELINE] Starte: Batch-Vorhersage...")
    run_batch_prediction()  # nutzt heute() als Referenzdatum, schreibt predictions.csv + History
    print("✅ [PIPELINE] Batch-Vorhersage beendet!")

    print("\n🚀 [PIPELINE] Starte: Telegram-Benachrichtigung...")
    run_notification()  # sendet Push; Leer-Fall = nichts senden
    print("✅ [PIPELINE] Benachrichtigung beendet!")
    
    total_duration = time.time() - global_start
    print("\n" + "=" * 60)
    print(f"🎉 PIPELINE ERFOLGREICH DURCHGELAUFEN! Gesamtzeit: {total_duration:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()