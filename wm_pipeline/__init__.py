"""
wm_pipeline/__init__.py

Registriert alle Assets, Jobs und Sensors dieses Moduls bei Dagster.
"""

from dagster import Definitions

from .assets import (
    wm2026_live_data,
    cleaned_wm_data,
    trained_models,
    diagnostic_plots,
    wm_pipeline_job,
    new_finished_matches_sensor,
)

# Definitions ist das "Inhaltsverzeichnis" für Dagster:
# alles was Dagster kennen soll, wird hier registriert.
# Schedule wurde durch den event-getriebenen Sensor ersetzt.
defs = Definitions(
    assets=[wm2026_live_data, cleaned_wm_data, trained_models, diagnostic_plots],
    jobs=[wm_pipeline_job],
    sensors=[new_finished_matches_sensor],
)
