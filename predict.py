import pandas as pd
import joblib
import sys

def get_latest_team_stats(team, df):
    """Sucht die aktuellsten Form- und Marktwerte eines Teams."""
    team_df = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date')
    
    if team_df.empty:
        # Falls ein Team völlig neu ist (Fallback auf Median-Marktwert)
        return 1.3, 1.3, 150.0 
        
    last_game = team_df.iloc[-1]
    
    if last_game['home_team'] == team:
        return last_game['home_form_attack'], last_game['home_form_defense'], last_game['home_market_value']
    else:
        return last_game['away_form_attack'], last_game['away_form_defense'], last_game['away_market_value']

def main():
    if len(sys.argv) < 3:
        print("💡 Nutzung: python predict.py '<Heimteam>' '<Auswärtsteam>'")
        sys.exit(1)
        
    home_team = sys.argv[1]
    away_team = sys.argv[2]
    
    # 2026 Gastgeber-Liste für die automatische Erkennung
    hosts_2026 = ['USA', 'United States', 'Canada', 'Mexico']
    
    try:
        model_home = joblib.load("models/wm_home_goals_model.pkl")
        model_away = joblib.load("models/wm_away_goals_model.pkl")
        df = pd.read_csv("data/processed/cleaned_wm_data.csv")
    except FileNotFoundError as e:
        print(f"❌ Fehler: {e}")
        sys.exit(1)
        
    home_attack, home_defense, home_val = get_latest_team_stats(home_team, df)
    away_attack, away_defense, away_val = get_latest_team_stats(away_team, df)
    
    # Input-DataFrame exakt wie im Training bauen
    input_data = pd.DataFrame([{
        'home_form_attack': home_attack,
        'home_form_defense': home_defense,
        'away_form_attack': away_attack,
        'away_form_defense': away_defense,
        'home_market_value': home_val,
        'away_market_value': away_val,
        'home_is_host': 1 if home_team in hosts_2026 else 0,
        'away_is_host': 1 if away_team in hosts_2026 else 0,
        'neutral': 1
    }])
    
    pred_home_goals = max(0, model_home.predict(input_data)[0])
    pred_away_goals = max(0, model_away.predict(input_data)[0])
    
    print("\n" + "="*50)
    print(f"🔮 PREDICTION ENGINE — WM 2026 SIMULATION:")
    print(f"   {home_team} vs. {away_team}")
    print("="*50)
    print(f"💰 Kaderwert {home_team}: {home_val:.1f}M € | {away_team}: {away_val:.1f}M €")
    print(f"⚽ Erwartete Tore {home_team}: {pred_home_goals:.2f}")
    print(f"⚽ Erwartete Tore {away_team}: {pred_away_goals:.2f}")
    print("-"*50)
    print(f"🏆 Mathematischer Tipp: {home_team} {round(pred_home_goals)} - {round(pred_away_goals)} {away_team}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()