"""
wm_pipeline/__init__.py

Registriert alle Assets, Jobs und Schedules dieses Moduls bei Dagster.
"""

from dagster import Definitions, load_assets_from_modules

from wm_pipeline import assets
from wm_pipeline.assets import wm_pipeline_job, wm_pipeline_schedule

# Alle Assets aus assets.py automatisch laden
all_assets = load_assets_from_modules([assets])

# Definitions ist das "Inhaltsverzeichnis" für Dagster:
# alles was Dagster kennen soll, wird hier registriert
defs = Definitions(
    assets=all_assets,
    jobs=[wm_pipeline_job],
    schedules=[wm_pipeline_schedule],
)