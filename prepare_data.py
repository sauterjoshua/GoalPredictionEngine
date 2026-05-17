import pandas as pd
import numpy as np
import yaml
import sys

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def process_data():
    config = load_config()
    raw_path = config['tournaments']['world_cup']['raw_path']
    processed_path = config['tournaments']['world_cup']['processed_path']
    
    print(f"⏳ Lade Rohdaten aus {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # --- AUTOMATISCHES SPALTEN-MAPPING ---
    print("\n🔍 Analysiere Spaltenstruktur...")
    print(f"Gefundene Spalten in deiner Datei: {df.columns.tolist()}")
    
    # Wir bauen ein Wörterbuch, um unterschiedliche Kaggle-Formate abzufangen
    mapping = {}
    
    # 1. Datum / Jahr finden
    for col in ['date', 'Datetime', 'Year', 'year', 'Date']:
        if col in df.columns:
            mapping[col] = 'date'
            break
            
    # 2. Teams finden
    for col in ['home_team', 'Home Team Name', 'HomeTeam', 'Home Team']:
        if col in df.columns:
            mapping[col] = 'home_team'
            break
    for col in ['away_team', 'Away Team Name', 'AwayTeam', 'Away Team']:
        if col in df.columns:
            mapping[col] = 'away_team'
            break
            
    # 3. Tore finden
    for col in ['home_score', 'Home Team Goals', 'FTHG', 'Home Goals']:
        if col in df.columns:
            mapping[col] = 'home_score'
            break
    for col in ['away_score', 'Away Team Goals', 'FTAG', 'Away Goals']:
        if col in df.columns:
            mapping[col] = 'away_score'
            break

    # Spalten umbenennen
    df = df.rename(columns=mapping)
    
    # Prüfen, ob alle kritischen Spalten jetzt existieren
    required_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n❌ Fehler: Die KI konnte folgende Kernspalten nicht zuordnen: {missing_cols}")
        print("Bitte passe das Mapping im Skript manuell an deine Spaltennamen an.")
        sys.exit(1)
        
    print("✅ Spalten erfolgreich harmonisiert!")
    
    # 4. Falls keine 'neutral'-Spalte da ist (bei reinen WM-Listen oft so), erstellen wir sie (WM = fast immer neutral)
    if 'neutral' not in df.columns:
        df['neutral'] = 1
    else:
        df['neutral'] = df['neutral'].astype(int)
        
    # Falls es ein gemischter Datensatz ist, filtern wir nach WM. Wenn es eine reine WM-Datei ist, ignorieren wir den Filter.
    if 'tournament' in df.columns:
        df = df[df['tournament'] == 'FIFA World Cup']

    # Datums-Formatierung & Sortierung
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Falls nur "Year" (z.B. 1930) drin stand, füllen wir es auf
    df['date'] = df['date'].fillna(pd.to_datetime(df['date'].dt.year.astype(str) + '-01-01', errors='coerce'))
    
    df = df.sort_values('date').reset_index(drop=True)
    df = df.dropna(subset=['home_score', 'away_score', 'home_team', 'away_team'])
    
    print(f"🏟️ Verarbeite {len(df)} historische Match-Datensätze...")
    
    # 5. Historische Stärken berechnen (Rolling Features)
    team_stats = {}
    home_attack_feat, home_defense_feat = [], []
    away_attack_feat, away_defense_feat = [], []
    
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        for team in [home, away]:
            if team not in team_stats:
                team_stats[team] = {'scored': [], 'conceded': []}
                
        home_attack_feat.append(np.mean(team_stats[home]['scored']) if team_stats[home]['scored'] else 1.2)
        home_defense_feat.append(np.mean(team_stats[home]['conceded']) if team_stats[home]['conceded'] else 1.2)
        
        away_attack_feat.append(np.mean(team_stats[away]['scored']) if team_stats[away]['scored'] else 1.2)
        away_defense_feat.append(np.mean(team_stats[away]['conceded']) if team_stats[away]['conceded'] else 1.2)
        
        team_stats[home]['scored'].append(row['home_score'])
        team_stats[home]['conceded'].append(row['away_score'])
        team_stats[away]['scored'].append(row['away_score'])
        team_stats[away]['conceded'].append(row['home_score'])
        
    df['home_hist_attack'] = home_attack_feat
    df['home_hist_defense'] = home_defense_feat
    df['away_hist_attack'] = away_attack_feat
    df['away_hist_defense'] = away_defense_feat
    
    final_features = [
        'date', 'home_team', 'away_team', 
        'home_hist_attack', 'home_hist_defense',
        'away_hist_attack', 'away_hist_defense',
        'neutral', 'home_score', 'away_score'
    ]
    
    cleaned_df = df[final_features]
    cleaned_df.to_csv(processed_path, index=False)
    print(f"✅ Daten erfolgreich aufbereitet und unter '{processed_path}' gespeichert!")

if __name__ == "__main__":
    process_data()