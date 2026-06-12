"""
Dagster-Assets der WM-Vorhersage-Pipeline.
Jedes Asset ist ein Datenprodukt (CSV, Modell, Plot); Abhängigkeiten werden
über `deps=` deklariert und von Dagster automatisch aufgelöst.
"""

from dagster import (
    asset,
    AssetExecutionContext,
    define_asset_job,
    sensor,
    RunRequest,
    SkipReason,
    SensorEvaluationContext,
    MaterializeResult,
    MetadataValue,
)


@asset
def wm2026_live_data(context: AssetExecutionContext) -> MaterializeResult:
    """Live-Daten der WM 2026 von football-data.org (bei jedem Lauf neu geholt)."""
    from src.utils.data_sources import fetch_world_cup_2026_matches, save_matches_to_csv

    df = fetch_world_cup_2026_matches()
    output_path = save_matches_to_csv(df)

    finished_count = (df["status"] == "FINISHED").sum()
    total_count = len(df)

    context.log.info(
        f"✅ WM 2026 Daten geladen: {total_count} Spiele, "
        f"davon {finished_count} bereits abgeschlossen."
    )

    # Statistik-Kacheln in der Dagster-UI
    return MaterializeResult(
        metadata={
            "total_matches": MetadataValue.int(int(total_count)),
            "finished_matches": MetadataValue.int(int(finished_count)),
            "output_path": MetadataValue.path(output_path),
        }
    )


@asset(deps=[wm2026_live_data])
def cleaned_wm_data(context: AssetExecutionContext):
    """Bereinigte WM-Daten (Wrapper um prepare_data.py)."""
    from src.pipeline.prepare_data import process_data
    process_data()
    context.log.info("✅ WM-Daten erfolgreich aufbereitet.")


@asset(deps=[cleaned_wm_data])
def trained_models(context: AssetExecutionContext):
    """Trainierte XGBoost-Modelle (Wrapper um train.py)."""
    from src.pipeline.train import main as train_main
    train_main()
    context.log.info("✅ Modelle erfolgreich trainiert.")


@asset(deps=[trained_models])
def diagnostic_plots(context: AssetExecutionContext):
    """Diagnose-Plots im Ordner 'plots/' (Wrapper um visualize.py)."""
    from src.pipeline.visualize import main as visualize_main
    visualize_main()
    context.log.info("✅ Diagnose-Plots generiert.")


wm_pipeline_job = define_asset_job(
    name="wm_pipeline_job",
    selection="*",  # alle Assets
)


# Pollt alle 5 Minuten; triggert den Job sobald neue FINISHED-Spiele auftauchen
@sensor(
    job=wm_pipeline_job,
    minimum_interval_seconds=300,
    description="Triggert die Pipeline, sobald ein neues Spiel FINISHED ist.",
)
def new_finished_matches_sensor(context: SensorEvaluationContext):
    """Cursor = Anzahl bekannter FINISHED-Spiele (bleibt über Daemon-Neustarts erhalten)."""
    last_count = int(context.cursor) if context.cursor else 0

    try:
        from src.utils.data_sources import fetch_world_cup_2026_matches
        df = fetch_world_cup_2026_matches()
        current_count = int((df["status"] == "FINISHED").sum())
    except Exception as exc:
        return SkipReason(f"API-Fehler beim Abrufen der WM-Daten: {exc}")

    context.log.info(f"FINISHED-Spiele: {current_count} (vorher: {last_count})")

    if current_count > last_count:
        # Cursor VOR dem RunRequest setzen, damit ein Job-Fehler kein Endlos-Retriggern auslöst
        context.update_cursor(str(current_count))
        return RunRequest(
            run_key=f"finished_count_{current_count}",
            tags={"trigger": "sensor", "finished_matches": str(current_count)},
        )

    return SkipReason(
        f"Keine neuen FINISHED-Spiele ({current_count}). "
        f"Nächster Check in ~5 Minuten."
    )

@asset(deps=["trained_models"])
def match_predictions(context):
    from src.pipeline.predict import run_batch_prediction
    df = run_batch_prediction()  # nutzt heute() als Referenzdatum
    return MaterializeResult(
        metadata={"predicted_matches": len(df)}
    )

@asset(deps=["match_predictions"])
def telegram_notification(context):
    from src.delivery.notify import run_notification
    count = run_notification()
    context.log.info(f"{count} Spiel(e) per Telegram benachrichtigt")
    return MaterializeResult(
        metadata={"notified_matches": MetadataValue.int(count)}
    )
