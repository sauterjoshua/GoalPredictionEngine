"""
predict.py

Dieses Skript steuert die Inferenz-Phase (Live-Vorhersage) der Engine.
Es lädt die fertig trainierten Modelle und prognostiziert für zwei beliebige
Nationalteams das wahrscheinlichste WM-Ergebnis. Marktwerte, historische Form
und das Gastgeber-Flag für die WM 2026 werden automatisch im Hintergrund ermittelt.
"""

import sys
import joblib
import pandas as pd


def get_latest_team_stats(team: str, df: pd.DataFrame) -> tuple[float, float, float]:
    """Sucht die aktuellsten Form- und Marktwerte eines Teams aus dem Datensatz.

    Args:
        team (str): Der Name der Nationalmannschaft.
        df (pd.DataFrame): Der historische, aufbereitete Datensatz.

    Returns:
        tuple: (form_attack, form_defense, market_value)
    """
    # Filtere alle Spiele, an denen das Team beteiligt war
    team_df = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date")
    
    # Fallback-Logik für Teams, die nicht im historischen WM-Set ab 2000 existieren
    if team_df.empty:
        median_market_value = df["home_market_value"].median()
        return 1.3, 1.3, float(median_market_value)
        
    last_game = team_df.iloc[-1]
    
    # Werte extrahieren, je nachdem ob das Team zuletzt Heim oder Auswärts spielte
    if last_game["home_team"] == team:
        return (
            float(last_game["home_form_attack"]), 
            float(last_game["home_form_defense"]), 
            float(last_game["home_market_value"])
        )
    else:
        return (
            float(last_game["away_form_attack"]), 
            float(last_game["away_form_defense"]), 
            float(last_game["away_market_value"])
        )


def main():
    """Hauptfunktion zur Validierung der Argumente und Ausführung der Inferenz."""
    if len(sys.argv) < 3:
        print("💡 Nutzung im Terminal: python predict.py '<Heimteam>' '<Auswärtsteam>'")
        print("   Beispiel: python predict.py 'Germany' 'France'")
        sys.exit(1)
        
    home_team = sys.argv[1]
    away_team = sys.argv[2]
    
    # Offizielle Gastgeber-Liste der WM 2026 zur automatischen Feature-Injektion
    hosts_2026 = ["USA", "United States", "Canada", "Mexico"]
    
    # Laden der serialisierten Modelle und Daten
    try:
        model_home = joblib.load("models/wm_home_goals_model.pkl")
        model_away = joblib.load("models/wm_away_goals_model.pkl")
        df = pd.read_csv("data/processed/cleaned_wm_data.csv")
    except FileNotFoundError as e:
        print(f"❌ Fehler: Wichtige Projektdateien fehlen ({e}).")
        print("   Bitte führe zuerst prepare_data.py und train.py aus.")
        sys.exit(1)
        
    # Features der beiden Teams ermitteln
    home_attack, home_defense, home_val = get_latest_team_stats(home_team, df)
    away_attack, away_defense, away_val = get_latest_team_stats(away_team, df)
    
    # Input-DataFrame exakt in der Struktur aufbauen, wie es XGBoost erwartet
    input_data = pd.DataFrame([{
        "home_form_attack": home_attack,
        "home_form_defense": home_defense,
        "away_form_attack": away_attack,
        "away_form_defense": away_defense,
        "home_market_value": home_val,
        "away_market_value": away_val,
        "home_is_host": 1 if home_team in hosts_2026 else 0,
        "away_is_host": 1 if away_team in hosts_2026 else 0,
        "neutral": 1
    }])
    
    # Kontinuierliche Tor-Erwartung berechnen (Negative Werte via max(0, x) abfangen)
    pred_home_goals = max(0.0, float(model_home.predict(input_data)[0]))
    pred_away_goals = max(0.0, float(model_away.predict(input_data)[0]))
    
    # Formatiertes Terminal-Dashboard ausgeben
    print("\n" + "=" * 55)
    print(f"🔮 PREDICTION ENGINE — WM 2026 SIMULATION:")
    print(f"   {home_team} vs. {away_team}")
    print("=" * 55)
    print(f"💰 Kaderwert {home_team:12}: {home_val:6.1f}M € | Form-Angriff: {home_attack:.2f}")
    print(f"💰 Kaderwert {away_team:12}: {away_val:6.1f}M € | Form-Angriff: {away_attack:.2f}")
    print("-" * 55)
    print(f"⚽ Erwartete Tore {home_team}: {pred_home_goals:.2f}")
    print(f"⚽ Erwartete Tore {away_team}: {pred_away_goals:.2f}")
    print("-" * 55)
    print(f"🏆 Mathematischer Tipp: {home_team} {round(pred_home_goals)} - {round(pred_away_goals)} {away_team}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()