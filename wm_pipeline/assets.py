"""
wm_pipeline/assets.py

Dagster-Assets für die WM-Vorhersage-Pipeline.
Jedes Asset entspricht einem "Datenprodukt": eine CSV-Datei, ein Modell, ein Plot.
Dagster erkennt automatisch die Abhängigkeiten zwischen den Assets.
"""

from dagster import asset, AssetExecutionContext, define_asset_job, ScheduleDefinition, MaterializeResult, MetadataValue


@asset
def wm2026_live_data(context: AssetExecutionContext) -> MaterializeResult:
    """
    Live-Daten der WM 2026 von football-data.org.
    
    Wird bei jeder Pipeline-Ausführung neu von der API geholt.
    Während der WM wird das automatisch immer aktueller.
    """
    from .data_sources import fetch_world_cup_2026_matches, save_matches_to_csv

    df = fetch_world_cup_2026_matches()
    output_path = save_matches_to_csv(df)

    # Zähle, wie viele Spiele schon abgeschlossen sind
    finished_count = (df["status"] == "FINISHED").sum()
    total_count = len(df)

    context.log.info(
        f"✅ WM 2026 Daten geladen: {total_count} Spiele, "
        f"davon {finished_count} bereits abgeschlossen."
    )

    # Diese Metadaten erscheinen in der Dagster-UI als kleine Statistik-Kacheln
    return MaterializeResult(
        metadata={
            "total_matches": MetadataValue.int(int(total_count)),
            "finished_matches": MetadataValue.int(int(finished_count)),
            "output_path": MetadataValue.path(output_path),
        }
    )
    
    
@asset(deps=[wm2026_live_data])
def cleaned_wm_data(context: AssetExecutionContext):
    """
    Erstes Asset: Bereinigte WM-Daten.
    
    Wrappt prepare_data.py. Hängt jetzt von wm2026_live_data ab —
    das stellt sicher, dass die neuesten WM-Spiele vor dem Cleaning geholt werden.
    """
    from prepare_data import process_data
    process_data()
    context.log.info("✅ WM-Daten erfolgreich aufbereitet.")


@asset(deps=[cleaned_wm_data])
def trained_models(context: AssetExecutionContext):
    """
    Zweites Asset: Trainierte XGBoost-Modelle.
    
    Hängt von cleaned_wm_data ab — Dagster sorgt automatisch dafür,
    dass die Daten vor dem Training fertig sind.
    """
    from train import main as train_main
    train_main()
    context.log.info("✅ Modelle erfolgreich trainiert.")


@asset(deps=[trained_models])
def diagnostic_plots(context: AssetExecutionContext):
    """
    Drittes Asset: Diagnose-Plots.
    
    Hängt von trained_models ab — Plots können nur erstellt werden,
    wenn die Modelle existieren.
    """
    from visualize import main as visualize_main
    visualize_main()
    context.log.info("✅ Diagnose-Plots generiert.")
    

# === JOB DEFINITION ===
# Ein Job ist eine ausführbare Bündelung aller Assets.
# AssetSelection.all() bedeutet: nimm wirklich alle definierten Assets.
wm_pipeline_job = define_asset_job(
    name="wm_pipeline_job",
    selection="*",  # "*" = alle Assets
)


# === SCHEDULE DEFINITION ===
# Cron-Ausdruck "0 */12 * * *" = "zur Minute 0, alle 12 Stunden"
# Also: 00:00 Uhr und 12:00 Uhr (Mitternacht und Mittag)
wm_pipeline_schedule = ScheduleDefinition(
    name="wm_pipeline_12h_schedule",
    job=wm_pipeline_job,
    cron_schedule="0 */12 * * *",
)