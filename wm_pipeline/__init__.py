"""
wm_pipeline/__init__.py

Registriert alle Assets, Jobs und Sensors bei Dagster – nur wenn Dagster
installiert ist (im schlanken Docker-Pipeline-Image ist es das nicht).
data_sources.py kann damit auch ohne Dagster importiert werden.
"""

try:
    from dagster import Definitions

    from .assets import (
        wm2026_live_data,
        cleaned_wm_data,
        trained_models,
        match_predictions,
        telegram_notification,
        diagnostic_plots,
        wm_pipeline_job,
        new_finished_matches_sensor,
    )

    # Definitions ist das "Inhaltsverzeichnis" für Dagster:
    # alles was Dagster kennen soll, wird hier registriert.
    defs = Definitions(
        assets=[
            wm2026_live_data,
            cleaned_wm_data,
            trained_models,
            match_predictions,
            telegram_notification,
            diagnostic_plots,
        ],
        sensors=[new_finished_matches_sensor],
        jobs=[wm_pipeline_job],
    )
except ImportError:
    # Dagster nicht installiert (z.B. schlankes Docker-Image) – kein Problem.
    # data_sources, prepare_data etc. funktionieren unabhängig davon.
    pass
