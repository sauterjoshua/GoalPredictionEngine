"""
dashboard.py

Streamlit-Dashboard für die WM-2026 Tor-Vorhersage-Engine.

ETAPPE 2 (Basis): Zeigt die anstehenden Spiele aus predictions.csv als
dezente, moderne Karten mit Flaggen-Emojis. Über einen Datums-Filter lässt
sich durch die geloggte Vorhersage-Historie (prediction_history.csv) blättern.

Reine Anzeige-Schicht: Das Dashboard liest nur CSVs, es rechnet nichts.
Die Vorhersagen werden von der Dagster-Pipeline (predict.py) erzeugt.

Start:  streamlit run dashboard.py   (aus dem Projekt-Root!)
"""

from datetime import datetime

import pandas as pd
import streamlit as st
import yaml


# --------------------------------------------------------------------------
# Konfiguration & Daten laden
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Flaggen-Emoji aus Ländernamen
# --------------------------------------------------------------------------
# Mapping Ländername -> ISO-2-Code (nur die wichtigsten WM-Nationen;
# Fallback ist eine neutrale Flagge, falls ein Land fehlt).
COUNTRY_TO_ISO = {
    "Germany": "DE", "France": "FR", "Spain": "ES", "Italy": "IT",
    "England": "GB", "Brazil": "BR", "Argentina": "AR", "Portugal": "PT",
    "Netherlands": "NL", "Belgium": "BE", "Croatia": "HR", "USA": "US",
    "United States": "US", "Canada": "CA", "Mexico": "MX", "Japan": "JP",
    "South Korea": "KR", "Curaçao": "CW", "Curacao": "CW", "Ecuador": "EC",
    "Ivory Coast": "CI", "Morocco": "MA", "Senegal": "SN", "Ghana": "GH",
    "Uruguay": "UY", "Colombia": "CO", "Switzerland": "CH", "Denmark": "DK",
    "Poland": "PL", "Austria": "AT", "Australia": "AU", "Qatar": "QA",
    "Saudi Arabia": "SA", "Nigeria": "NG", "Cameroon": "CM", "Serbia": "RS",
    "Norway": "NO", "Sweden": "SE", "Turkey": "TR", "Egypt": "EG",
}


def flag_emoji(country: str) -> str:
    """Wandelt einen Ländernamen in ein Flaggen-Emoji.

    Flaggen-Emojis bestehen aus zwei 'Regional Indicator Symbols'.
    Beispiel: 'DE' -> 🇩🇪. Unbekannte Länder bekommen eine weiße Flagge.
    """
    iso = COUNTRY_TO_ISO.get(country)
    if not iso:
        return "🏳️"
    # 0x1F1E6 ist 'A' als Regional Indicator; Offset vom normalen Buchstaben
    return "".join(chr(0x1F1E6 + (ord(c) - ord("A"))) for c in iso.upper())


# --------------------------------------------------------------------------
# Karten-Darstellung
# --------------------------------------------------------------------------
def render_match_card(row: pd.Series):
    """Zeichnet eine einzelne Spielkarte mit Tipp, Toren, Form und Kaderwert."""
    home, away = row["home_team"], row["away_team"]
    home_flag, away_flag = flag_emoji(home), flag_emoji(away)

    with st.container(border=True):
        # Datum klein oben
        st.caption(f"📅 {row['date']}")

        # Drei Spalten: Heim | Tipp | Auswärts
        col_home, col_score, col_away = st.columns([3, 2, 3])

        with col_home:
            st.markdown(f"### {home_flag} {home}")
            st.caption(f"Form-Angriff: {row['home_form_attack']:.2f}")
            st.caption(f"Kaderwert: {row['home_market_value']:.0f} M€")

        with col_score:
            # Großer gerundeter Tipp, darunter die exakte Erwartung
            st.markdown(
                f"<div style='text-align:center;font-size:2.2rem;"
                f"font-weight:700;line-height:1.1'>"
                f"{int(row['tipp_home'])} : {int(row['tipp_away'])}</div>",
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


# --------------------------------------------------------------------------
# Haupt-App
# --------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="WM 2026 Predictions",
        page_icon="⚽",
        layout="centered",
    )

    config = load_config()
    predictions_path = config["tournaments"]["world_cup"]["predictions_path"]
    history_path = config["tournaments"]["world_cup"]["history_path"]

    # Dezentes globales Styling
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2.5rem; }
        h1 { letter-spacing: -0.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Header ----
    st.title("⚽ WM 2026 — Tor-Vorhersagen")
    st.markdown(
        "<p style='color:#888;margin-top:-0.6rem'>"
        "Automatisierte Prognosen, die mit jedem Spiel dazulernen.</p>",
        unsafe_allow_html=True,
    )

    # ---- Datenquelle wählen: aktuelles Fenster oder Historie durchblättern ----
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

        # Verfügbare Spieltage aus der Historie zur Auswahl anbieten
        verfuegbare_tage = sorted(history["date"].unique())
        gewaehlter_tag = st.select_slider(
            "Spieltag wählen", options=verfuegbare_tage, value=verfuegbare_tage[-1]
        )

        # Pro Spiel die JÜNGSTE (beste) Vorhersage dieses Tages nehmen
        tag_df = history[history["date"] == gewaehlter_tag].copy()
        tag_df = tag_df.sort_values("predicted_at")
        df = tag_df.groupby("match_key", as_index=False).last()

    # ---- Karten rendern ----
    st.markdown(f"**{len(df)} Spiel(e)**")
    for _, row in df.iterrows():
        render_match_card(row)


if __name__ == "__main__":
    main()
    