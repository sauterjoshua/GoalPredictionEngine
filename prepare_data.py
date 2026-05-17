import pandas as pd
import numpy as np
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def process_data():
    config = load_config()
    raw_path = config['tournaments']['world_cup']['raw_path']
    processed_path = config['tournaments']['world_cup']['processed_path']
    
    df = pd.read_csv(raw_path)
    
    # [Spalten-Mapping weggelassen für die Übersicht, bleibt wie vorher]
    # ... (Dein funktionierendes Spalten-Mapping von vorhin) ...
    
    # 1. RADIKALER ZEITFILTER: Nur moderner Fußball ab dem Jahr 2000
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].dt.year >= 2000].reset_index(drop=True)
    
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"🏟️ Verarbeite {len(df)} MODERNE Weltmeisterschafts-Spiele (Ab 2000)...")
    
    team_stats = {}
    home_form_attack, home_form_defense = [], []
    away_form_attack, away_form_defense = [], []
    
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        for team in [home, away]:
            if team not in team_stats:
                # Wir tracken weiterhin die Tore
                team_stats[team] = {'scored': [], 'conceded': []}
        
        # 2. NEUER FAKTOR: Nur die letzten 5 Spiele fließen in die Form ein (Rolling Window)
        # Wenn keine Daten da sind, starten wir mit dem Turnierschnitt (ca. 1.3 Tore)
        home_form_attack.append(np.mean(team_stats[home]['scored'][-5:]) if team_stats[home]['scored'] else 1.3)
        home_form_defense.append(np.mean(team_stats[home]['conceded'][-5:]) if team_stats[home]['conceded'] else 1.3)
        
        away_form_attack.append(np.mean(team_stats[away]['scored'][-5:]) if team_stats[away]['scored'] else 1.3)
        away_form_defense.append(np.mean(team_stats[away]['conceded'][-5:]) if team_stats[away]['conceded'] else 1.3)
        
        # Historie aktualisieren
        team_stats[home]['scored'].append(row['home_score'])
        team_stats[home]['conceded'].append(row['away_score'])
        team_stats[away]['scored'].append(row['away_score'])
        team_stats[away]['conceded'].append(row['home_score'])
        
    df['home_form_attack'] = home_form_attack
    df['home_form_defense'] = home_form_defense
    df['away_form_attack'] = away_form_attack
    df['away_form_defense'] = away_defense_feat = away_form_defense # Zuweisung korrigiert
    
    df['neutral'] = 1
    
    final_features = [
        'date', 'home_team', 'away_team', 
        'home_form_attack', 'home_form_defense',
        'away_form_attack', 'away_form_defense',
        'neutral', 'home_score', 'away_score'
    ]
    
    cleaned_df = df[final_features]
    cleaned_df.to_csv(processed_path, index=False)
    print(f"✅ Daten auf modernen Stand gebracht und gespeichert!")

if __name__ == "__main__":
    process_data()