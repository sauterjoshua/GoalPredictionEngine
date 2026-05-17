import pandas as pd
import joblib
import sys

def get_latest_team_stats(team, df):
    """Sucht die aktuellsten historischen Werte eines Teams aus dem Datensatz."""
    # Filtere alle Spiele des Teams
    team_df = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date')
    
    if team_df.empty:
        print(f"⚠️ Team '{team}' wurde in der WM-Historie nicht gefunden. Nutze Standardwerte.")
        return 1.2, 1.2 # Fallback-Werte
        
    # Nimm das allerletzte Spiel
    last_game = team_df.iloc[-1]
    
    # Prüfe, ob das Team im letzten Spiel Heim- oder Auswärtsmannschaft war
    if last_game['home_team'] == team:
        return last_game['home_hist_attack'], last_game['home_hist_defense']
    else:
        return last_game['away_hist_attack'], last_game['away_hist_defense']

def main():
    if len(sys.argv) < 3:
        print("💡 Nutzung: python predict.py '<Heimteam>' '<Auswärtsteam>'")
        print("Format beachten (z.B. 'Germany', 'Brazil', 'France', 'Argentina')")
        sys.exit(1)
        
    home_team = sys.argv[1]
    away_team = sys.argv[2]
    
    # 1. Modelle und Daten laden
    try:
        model_home = joblib.load("models/wm_home_goals_model.pkl")
        model_away = joblib.load("models/wm_away_goals_model.pkl")
        df = pd.read_csv("data/processed/cleaned_wm_data.csv")
    except FileNotFoundError as e:
        print(f"❌ Fehler beim Laden der Dateien: {e}")
        sys.exit(1)
        
    # 2. Die aktuellsten Features für beide Teams holen
    home_attack, home_defense = get_latest_team_stats(home_team, df)
    away_attack, away_defense = get_latest_team_stats(away_team, df)
    
    # 3. Feature-DataFrame für das Modell bauen (muss exakt wie beim Training sein!)
    input_data = pd.DataFrame([{
        'home_hist_attack': home_attack,
        'home_hist_defense': home_defense,
        'away_hist_attack': away_attack,
        'away_hist_defense': away_defense,
        'neutral': 1  # WM-Spiele sind standardmäßig auf neutralem Boden
    }])
    
    # 4. Vorhersage berechnen
    pred_home_goals = model_home.predict(input_data)[0]
    pred_away_goals = model_away.predict(input_data)[0]
    
    # 5. Ergebnis präsentieren
    print("\n" + "="*40)
    print(f"🔮 KI-VORHERSAGE FÜR DAS MATCH:")
    print(f"   {home_team} vs. {away_team}")
    print("="*40)
    print(f"⚽ Erwartete Tore {home_team}: {pred_home_goals:.2f}")
    print(f"⚽ Erwartete Tore {away_team}: {pred_away_goals:.2f}")
    print("-"*40)
    print(f"🏆 Tipp: {home_team} {round(pred_home_goals)} - {round(pred_away_goals)} {away_team}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()