import pandas as pd
import joblib
import sys

def get_latest_team_stats(team, df):
    """Sucht die aktuellsten Form-Werte eines Teams aus dem modernen Datensatz."""
    team_df = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date')
    
    if team_df.empty:
        print(f"⚠️ Team '{team}' wurde in den modernen Daten nicht gefunden. Nutze Standardwerte.")
        return 1.3, 1.3
        
    last_game = team_df.iloc[-1]
    
    # Prüfe, ob das Team im letzten Spiel Heim- oder Auswärtsmannschaft war
    if last_game['home_team'] == team:
        return last_game['home_form_attack'], last_game['home_form_defense']
    else:
        return last_game['away_form_attack'], last_game['away_form_defense']

def main():
    if len(sys.argv) < 3:
        print("💡 Nutzung: python predict.py '<Heimteam>' '<Auswärtsteam>'")
        print("Beispiel: python predict.py 'Germany' 'Brazil'")
        sys.exit(1)
        
    home_team = sys.argv[1]
    away_team = sys.argv[2]
    
    try:
        model_home = joblib.load("models/wm_home_goals_model.pkl")
        model_away = joblib.load("models/wm_away_goals_model.pkl")
        df = pd.read_csv("data/processed/cleaned_wm_data.csv")
    except FileNotFoundError as e:
        print(f"❌ Fehler beim Laden der Dateien: {e}")
        sys.exit(1)
        
    home_attack, home_defense = get_latest_team_stats(home_team, df)
    away_attack, away_defense = get_latest_team_stats(away_team, df)
    
    input_data = pd.DataFrame([{
        'home_form_attack': home_attack,
        'home_form_defense': home_defense,
        'away_form_attack': away_attack,
        'away_form_defense': away_defense,
        'neutral': 1
    }])
    
    pred_home_goals = model_home.predict(input_data)[0]
    pred_away_goals = model_away.predict(input_data)[0]
    
    print("\n" + "="*40)
    print(f"🔮 KI-VORHERSAGE (MODERNES FORM-MODELL):")
    print(f"   {home_team} vs. {away_team}")
    print("="*40)
    print(f"⚽ Erwartete Tore {home_team}: {max(0, pred_home_goals):.2f}")
    print(f"⚽ Erwartete Tore {away_team}: {max(0, pred_away_goals):.2f}")
    print("-"*40)
    print(f"🏆 Tipp: {home_team} {round(max(0, pred_home_goals))} - {round(max(0, pred_away_goals))} {away_team}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()