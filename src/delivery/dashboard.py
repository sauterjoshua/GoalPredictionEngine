"""
Streamlit-Dashboard für die WM-2026 Tor-Vorhersagen.

Reine Anzeige-Schicht: liest nur CSVs, rechnet nichts.
Vorhersagen werden von der Dagster-Pipeline (predict.py) erzeugt.

Start: streamlit run dashboard.py  (aus dem Projekt-Root)
"""

from datetime import datetime

import pandas as pd
import streamlit as st
import yaml


def load_config() -> dict:
    """Lädt die zentrale config.yaml (gleiche Pfade wie die Pipeline)."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=60)
def load_predictions(path: str) -> pd.DataFrame:
    """Lädt die aktuellen Vorhersagen (3-Tage-Fenster).

    Cache wird alle 60 Sekunden invalidiert, damit frische Pipeline-Läufe
    sichtbar werden, ohne dass man die App neu starten muss.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_history(path: str) -> pd.DataFrame:
    """Lädt die geloggte Vorhersage-Historie (alle bisherigen Vorhersagen)."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


from src.utils.flags import flag_emoji

STAGE_LABELS = {
    "GROUP_STAGE": "Gruppenphase",
    "LAST_16": "Achtelfinale",
    "QUARTER_FINALS": "Viertelfinale",
    "SEMI_FINALS": "Halbfinale",
    "FINAL": "Finale",
}


def _safe(row: pd.Series, col: str, default=None):
    val = row.get(col, default)
    try:
        return default if pd.isna(val) else val
    except TypeError:
        return default if val is None else val


def render_match_card(row: pd.Series):
    """Zeichnet eine einzelne Spielkarte mit Tipp, Toren, Form und Kaderwert."""
    home, away = row["home_team"], row["away_team"]
    home_flag, away_flag = flag_emoji(home), flag_emoji(away)

    decided_by = _safe(row, "decided_by", "90min")
    stage = _safe(row, "stage", "GROUP_STAGE")
    stage_label = STAGE_LABELS.get(stage, stage)
    home_et = int(_safe(row, "home_goals_et", 0) or 0)
    away_et = int(_safe(row, "away_goals_et", 0) or 0)
    tipp_home = int(row["tipp_home"])
    tipp_away = int(row["tipp_away"])

    if decided_by == "ET":
        score_display = f"{tipp_home + home_et} : {tipp_away + away_et}"
        score_sub = "(n.V.)"
    elif decided_by == "penalties":
        score_display = f"{tipp_home} : {tipp_away}"
        score_sub = "(i.E.)"
    else:
        score_display = f"{tipp_home} : {tipp_away}"
        score_sub = ""

    with st.container(border=True):
        st.caption(f"📅 {row['date']}  ·  {stage_label}")

        col_home, col_score, col_away = st.columns([3, 2, 3])

        with col_home:
            st.markdown(f"### {home_flag} {home}")
            st.caption(f"Form-Angriff: {row['home_form_attack']:.2f}")
            st.caption(f"Kaderwert: {row['home_market_value']:.0f} M€")

        with col_score:
            st.markdown(
                f"<div style='text-align:center;font-size:2.2rem;"
                f"font-weight:700;line-height:1.1'>{score_display}</div>",
                unsafe_allow_html=True,
            )
            if score_sub:
                st.markdown(
                    f"<div style='text-align:center;color:#888;font-size:0.9rem'>"
                    f"{score_sub}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='text-align:center;color:#888;font-size:0.85rem'>"
                f"exakt: {row['pred_home_goals']:.2f} – {row['pred_away_goals']:.2f}</div>",
                unsafe_allow_html=True,
            )

        with col_away:
            st.markdown(f"### {away_flag} {away}")
            st.caption(f"Form-Angriff: {row['away_form_attack']:.2f}")
            st.caption(f"Kaderwert: {row['away_market_value']:.0f} M€")

        if decided_by == "ET":
            st.markdown(
                "<div style='text-align:center;color:#4a9eff;font-size:0.85rem;"
                "padding-top:0.3rem'>⏱️ nach Verlängerung</div>",
                unsafe_allow_html=True,
            )
        elif decided_by == "penalties":
            winner = _safe(row, "predicted_winner", "")
            if winner == "home":
                winner_flag, winner_name = home_flag, home
            elif winner == "away":
                winner_flag, winner_name = away_flag, away
            else:
                winner_flag, winner_name = "🏆", "—"
            st.markdown(
                f"<div style='text-align:center;font-size:0.85rem;padding-top:0.3rem'>"
                f"🥅 Elfmeterschießen &nbsp;·&nbsp; "
                f"<strong>Sieger: {winner_flag} {winner_name}</strong></div>",
                unsafe_allow_html=True,
            )


def main():
    st.set_page_config(
        page_title="WM 2026 Predictions",
        page_icon="⚽",
        layout="centered",
    )

    config = load_config()
    predictions_path = config["tournaments"]["world_cup"]["predictions_path"]
    history_path = config["tournaments"]["world_cup"]["history_path"]

    st.markdown(
        """
        <style>
        .block-container { padding-top: 2.5rem; }
        h1 { letter-spacing: -0.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚽ WM 2026 — Tor-Vorhersagen")
    st.markdown(
        "<p style='color:#888;margin-top:-0.6rem'>"
        "Automatisierte Prognosen, die mit jedem Spiel dazulernen.</p>",
        unsafe_allow_html=True,
    )

    quelle = st.radio(
        "Datenquelle",
        ["Aktuelles 3-Tage-Fenster", "Historie durchblättern"],
        horizontal=True,
    )

    if quelle == "Aktuelles 3-Tage-Fenster":
        df = load_predictions(predictions_path)
        if df.empty:
            st.info(
                "🔭 Aktuell keine Spiele in den nächsten 3 Tagen. "
                "Sobald die WM läuft, erscheinen hier die anstehenden Partien."
            )
            return
    else:
        history = load_history(history_path)
        if history.empty:
            st.info("📭 Noch keine Vorhersage-Historie vorhanden.")
            return

        verfuegbare_tage = sorted(history["date"].unique())
        gewaehlter_tag = st.select_slider(
            "Spieltag wählen", options=verfuegbare_tage, value=verfuegbare_tage[-1]
        )

        # Pro Spiel nur die jüngste Vorhersage zeigen (Form kann sich zwischen Läufen ändern)
        tag_df = history[history["date"] == gewaehlter_tag].copy()
        tag_df = tag_df.sort_values("predicted_at")
        df = tag_df.groupby("match_key", as_index=False).last()

    st.markdown(f"**{len(df)} Spiel(e)**")
    for _, row in df.iterrows():
        render_match_card(row)


if __name__ == "__main__":
    main()
    