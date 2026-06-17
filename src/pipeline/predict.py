"""
Inferenz-Phase der Engine: lädt Modelle und prognostiziert WM-Ergebnisse.

Modi:
  CLI:   python predict.py 'Germany' 'France'
  Batch: run_batch_prediction()  →  nächste N Tage → predictions.csv
"""

import sys
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import yaml

from src.utils.form import compute_form, FALLBACK_FORM

HOSTS_2026 = ["USA", "United States", "Canada", "Mexico"]


def load_config() -> dict:
    """Lädt die zentrale Konfigurationsdatei (config.yaml)."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _median_market_value(df: pd.DataFrame) -> float:
    """Median-Marktwert über alle Spiele der Match-Historie.

    Wird als Referenz für den Gegner-Faktor (Phase C) gebraucht. Konsistent zum
    Trainings-Median, weil die match_history aus demselben Frame stammt.
    """
    if "home_market_value" not in df.columns or "away_market_value" not in df.columns:
        return 0.0
    all_mv = pd.concat([df["home_market_value"], df["away_market_value"]]).dropna()
    if all_mv.empty:
        return 0.0
    return float(all_mv.median())


def get_latest_team_stats(team: str, df: pd.DataFrame) -> tuple[float, float, float]:
    """Form + Marktwert eines Teams aus seinen letzten 5 Spielen.

    Nutzt die gemeinsame compute_form()-Funktion aus form.py (identisch zum Training)
    inkl. Spielart-Gewichtung (Phase B) und Gegner-Gewichtung (Phase C).
    """
    team_df = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date")
    median_mv = _median_market_value(df)

    if team_df.empty:
        return FALLBACK_FORM, FALLBACK_FORM, median_mv if median_mv > 0 else 0.0

    scored, conceded, tournaments, opponent_mvs = [], [], [], []
    for _, row in team_df.iterrows():
        if row["home_team"] == team:
            scored.append(row["home_score"])
            conceded.append(row["away_score"])
            opponent_mvs.append(row.get("away_market_value", None))
        else:
            scored.append(row["away_score"])
            conceded.append(row["home_score"])
            opponent_mvs.append(row.get("home_market_value", None))
        tournaments.append(row.get("tournament", None))

    form_attack, form_defense = compute_form(
        scored, conceded, tournaments, opponent_mvs, median_mv
    )

    # Marktwert aus dem jüngsten Spiel
    last_game = team_df.iloc[-1]
    if last_game["home_team"] == team:
        market_value = float(last_game["home_market_value"])
    else:
        market_value = float(last_game["away_market_value"])

    return form_attack, form_defense, market_value


def _wc_titles(team: str, df_features: pd.DataFrame) -> float:
    """Gibt world_cup_titles_before für ein Team aus dem Team-Features-DataFrame zurück."""
    rows = df_features[df_features["team"] == team]
    if rows.empty:
        return 0.0
    return float(rows.sort_values("version").iloc[-1]["world_cup_titles_before"])


def _norm_pair(a: float, b: float) -> tuple[float, float]:
    """Relative Normalisierung: a / (a + b). Bei Summe 0 → 0.5 / 0.5."""
    total = a + b
    if total == 0:
        return 0.5, 0.5
    return a / total, b / total


def predict_match(home_stats: dict, away_stats: dict,
                  is_knockout: bool = False) -> dict:
    """Wendet KO-Logik auf vorberechnete Team-Stats an.

    home_stats / away_stats erwartete Schlüssel:
      form_attack              – xG-Basis (Modell-Output oder direkte Form-Metrik)
      squad_total_market_value_eur – Marktwert (beliebige Einheit, nur relativ genutzt)
      world_cup_titles_before  – WM-Titel vor dem Turnier
    """
    pred_home_goals = max(0.0, float(home_stats["form_attack"]))
    pred_away_goals = max(0.0, float(away_stats["form_attack"]))
    home_val = float(home_stats.get("squad_total_market_value_eur", 0))
    away_val = float(away_stats.get("squad_total_market_value_eur", 0))
    home_wct = float(home_stats.get("world_cup_titles_before", 0))
    away_wct = float(away_stats.get("world_cup_titles_before", 0))

    home_goals_90 = round(pred_home_goals)
    away_goals_90 = round(pred_away_goals)

    home_goals_et = 0
    away_goals_et = 0
    decided_by = "90min"

    if is_knockout and home_goals_90 == away_goals_90:
        fatigue_factor = 0.75
        home_goals_et = round(pred_home_goals * (30 / 90) * fatigue_factor)
        away_goals_et = round(pred_away_goals * (30 / 90) * fatigue_factor)
        decided_by = "ET"

    home_total = home_goals_90 + home_goals_et
    away_total = away_goals_90 + away_goals_et

    if is_knockout and decided_by == "ET" and home_total == away_total:
        decided_by = "penalties"

    if decided_by == "penalties":
        norm_home_mv, _ = _norm_pair(home_val, away_val)
        norm_home_wct, _ = _norm_pair(home_wct, away_wct)
        home_penalty_score = 0.6 * norm_home_mv + 0.4 * norm_home_wct
        predicted_winner = "home" if home_penalty_score > 0.5 else "away"
    elif is_knockout:
        predicted_winner = "home" if home_total > away_total else "away"
    else:
        if home_goals_90 > away_goals_90:
            predicted_winner = "home"
        elif home_goals_90 < away_goals_90:
            predicted_winner = "away"
        else:
            predicted_winner = "draw"

    return {
        "pred_home_goals": round(pred_home_goals, 2),
        "pred_away_goals": round(pred_away_goals, 2),
        "tipp_home": home_goals_90,
        "tipp_away": away_goals_90,
        "home_goals_et": home_goals_et,
        "away_goals_et": away_goals_et,
        "decided_by": decided_by,
        "predicted_winner": predicted_winner,
    }


def _build_stats_dict(team: str, df: pd.DataFrame,
                      df_team_features: pd.DataFrame | None,
                      model_pred_goals: float) -> dict:
    """Baut den Stats-Dict für predict_match aus Modell-Output und Team-Daten."""
    _, _, market_val = get_latest_team_stats(team, df)
    wct = _wc_titles(team, df_team_features) if df_team_features is not None else 0.0
    return {
        "form_attack": model_pred_goals,
        "squad_total_market_value_eur": market_val,
        "world_cup_titles_before": wct,
    }


def load_models_and_data():
    """Lädt die serialisierten XGBoost-Modelle und die Match-Historie (Form-Quelle)."""
    config = load_config()
    model_home = joblib.load("models/wm_home_goals_model.pkl")
    model_away = joblib.load("models/wm_away_goals_model.pkl")
    df = pd.read_csv(config["tournaments"]["world_cup"]["match_history_path"])
    return model_home, model_away, df


def log_predictions_to_history(predictions_df: pd.DataFrame, history_path: str):
    """Hängt neue Vorhersagen an die History-CSV an."""
    now_stamp = datetime.now().isoformat(timespec="seconds")

    df_new = predictions_df.copy()
    df_new["match_key"] = (
        df_new["date"] + "_" + df_new["home_team"] + "_" + df_new["away_team"]
    )
    df_new["predicted_at"] = now_stamp

    if os.path.exists(history_path):
        history = pd.read_csv(history_path)
    else:
        history = pd.DataFrame()

    rows_to_append = []
    for _, row in df_new.iterrows():
        key = row["match_key"]

        if not history.empty and key in history["match_key"].values:
            last = history[history["match_key"] == key].iloc[-1]
            unchanged = (
                abs(last["pred_home_goals"] - row["pred_home_goals"]) < 1e-9
                and abs(last["pred_away_goals"] - row["pred_away_goals"]) < 1e-9
            )
            if unchanged:
                continue

        rows_to_append.append(row)

    if not rows_to_append:
        print("ℹ️ History: keine geänderten Vorhersagen, nichts angehängt.")
        return

    updated = pd.concat([history, pd.DataFrame(rows_to_append)], ignore_index=True)
    updated.to_csv(history_path, index=False)
    print(f"📝 History: {len(rows_to_append)} neue Vorhersage(n) geloggt → '{history_path}'")


def run_batch_prediction(reference_date: str | None = None) -> pd.DataFrame:
    """Rechnet Vorhersagen für alle TIMED-Spiele im Fenster [heute, heute + N Tage]."""
    config = load_config()
    live_path = config["tournaments"]["world_cup"]["live_path"]
    predictions_path = config["tournaments"]["world_cup"]["predictions_path"]
    team_features_path = config["tournaments"]["world_cup"]["team_features_path"]
    window_days = config.get("prediction", {}).get("prediction_window_days", 3)

    output_columns = [
        "date", "home_team", "away_team", "stage",
        "pred_home_goals", "pred_away_goals", "tipp_home", "tipp_away",
        "home_goals_et", "away_goals_et",
        "decided_by", "predicted_winner",
        "home_form_attack", "away_form_attack",
        "home_market_value", "away_market_value",
    ]

    if reference_date is None:
        now = datetime.now()
    else:
        now = datetime.strptime(reference_date, "%Y-%m-%d")
    window_end = now + timedelta(days=window_days)
    print(f"📅 Vorhersage-Fenster: {now.date()} bis {window_end.date()} ({window_days} Tage)")

    df_live = pd.read_csv(live_path)
    df_upcoming = df_live[df_live["status"] == "TIMED"].copy()

    if "stage" not in df_upcoming.columns:
        df_upcoming["stage"] = "GROUP_STAGE"

    df_upcoming["date"] = pd.to_datetime(
        df_upcoming["date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    df_upcoming = df_upcoming.dropna(subset=["date"])

    mask = (df_upcoming["date"] >= now) & (df_upcoming["date"] <= window_end)
    df_window = df_upcoming[mask].sort_values("date")

    if df_window.empty:
        print("ℹ️ Keine Spiele im Vorhersage-Fenster. Schreibe leere CSV.")
        empty_df = pd.DataFrame(columns=output_columns)
        empty_df.to_csv(predictions_path, index=False)
        return empty_df

    model_home, model_away, df = load_models_and_data()
    df_team_features = pd.read_csv(team_features_path)

    print(f"🔮 Rechne Vorhersagen für {len(df_window)} Spiel(e)...")
    results = []
    for _, match in df_window.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        stage = match.get("stage", "GROUP_STAGE")
        is_knockout = stage != "GROUP_STAGE"

        home_attack, home_defense, home_val = get_latest_team_stats(home_team, df)
        away_attack, away_defense, away_val = get_latest_team_stats(away_team, df)

        input_data = pd.DataFrame([{
            "home_form_attack": home_attack,
            "home_form_defense": home_defense,
            "away_form_attack": away_attack,
            "away_form_defense": away_defense,
            "home_market_value": home_val,
            "away_market_value": away_val,
            "home_is_host": 1 if home_team in HOSTS_2026 else 0,
            "away_is_host": 1 if away_team in HOSTS_2026 else 0,
            "neutral": 1,
        }])
        model_home_pred = max(0.0, float(model_home.predict(input_data)[0]))
        model_away_pred = max(0.0, float(model_away.predict(input_data)[0]))

        home_stats = _build_stats_dict(home_team, df, df_team_features, model_home_pred)
        away_stats = _build_stats_dict(away_team, df, df_team_features, model_away_pred)

        pred = predict_match(home_stats, away_stats, is_knockout=is_knockout)
        pred["home_team"] = home_team
        pred["away_team"] = away_team
        pred["date"] = match["date"].date().isoformat()
        pred["stage"] = stage
        pred["home_form_attack"] = round(home_attack, 2)
        pred["away_form_attack"] = round(away_attack, 2)
        pred["home_market_value"] = round(home_val, 1)
        pred["away_market_value"] = round(away_val, 1)
        results.append(pred)

    predictions_df = pd.DataFrame(results)[output_columns]
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✅ {len(predictions_df)} Vorhersage(n) gespeichert: '{predictions_path}'")

    history_path = config["tournaments"]["world_cup"]["history_path"]
    log_predictions_to_history(predictions_df, history_path)

    return predictions_df


def main():
    """CLI-Einzelvorhersage für zwei manuell angegebene Teams."""
    if len(sys.argv) < 3:
        print("💡 Nutzung im Terminal: python predict.py '<Heimteam>' '<Auswärtsteam>'")
        print("   Beispiel: python predict.py 'Germany' 'France'")
        sys.exit(1)

    home_team = sys.argv[1]
    away_team = sys.argv[2]

    try:
        model_home, model_away, df = load_models_and_data()
    except FileNotFoundError as e:
        print(f"❌ Fehler: Wichtige Projektdateien fehlen ({e}).")
        print("   Bitte führe zuerst prepare_data.py und train.py aus.")
        sys.exit(1)

    config = load_config()
    df_team_features = pd.read_csv(config["tournaments"]["world_cup"]["team_features_path"])

    home_attack, home_defense, home_val = get_latest_team_stats(home_team, df)
    away_attack, away_defense, away_val = get_latest_team_stats(away_team, df)

    input_data = pd.DataFrame([{
        "home_form_attack": home_attack, "home_form_defense": home_defense,
        "away_form_attack": away_attack, "away_form_defense": away_defense,
        "home_market_value": home_val, "away_market_value": away_val,
        "home_is_host": 1 if home_team in HOSTS_2026 else 0,
        "away_is_host": 1 if away_team in HOSTS_2026 else 0,
        "neutral": 1,
    }])
    model_home_pred = max(0.0, float(model_home.predict(input_data)[0]))
    model_away_pred = max(0.0, float(model_away.predict(input_data)[0]))

    home_stats = _build_stats_dict(home_team, df, df_team_features, model_home_pred)
    away_stats = _build_stats_dict(away_team, df, df_team_features, model_away_pred)
    pred = predict_match(home_stats, away_stats)

    print("\n" + "=" * 55)
    print(f"🔮 PREDICTION ENGINE — WM 2026 SIMULATION:")
    print(f"   {home_team} vs. {away_team}")
    print("=" * 55)
    print(f"💰 Kaderwert {home_team:12}: {home_val:6.1f}M € | Form-Angriff: {home_attack:.2f}")
    print(f"💰 Kaderwert {away_team:12}: {away_val:6.1f}M € | Form-Angriff: {away_attack:.2f}")
    print("-" * 55)
    print(f"⚽ Erwartete Tore {home_team}: {pred['pred_home_goals']:.2f}")
    print(f"⚽ Erwartete Tore {away_team}: {pred['pred_away_goals']:.2f}")
    print("-" * 55)
    print(f"🏆 Mathematischer Tipp: {home_team} {pred['tipp_home']} - {pred['tipp_away']} {away_team}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()