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
    mapping = {}
    
    for col in ['date', 'Datetime', 'Year', 'year', 'Date']:
        if col in df.columns:
            mapping[col] = 'date'
            break
            
    for col in ['home_team', 'Home Team Name', 'HomeTeam', 'Home Team']:
        if col in df.columns:
            mapping[col] = 'home_team'
            break
    for col in ['away_team', 'Away Team Name', 'AwayTeam', 'Away Team']:
        if col in df.columns:
            mapping[col] = 'away_team'
            break
            
    for col in ['home_score', 'Home Team Goals', 'FTHG', 'Home Goals']:
        if col in df.columns:
            mapping[col] = 'home_score'
            break
    for col in ['away_score', 'Away Team Goals', 'FTAG', 'Away Goals']:
        if col in df.columns:
            mapping[col] = 'away_score'
            break

    df = df.rename(columns=mapping)
    
    required_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n❌ Fehler: Die KI konnte folgende Kernspalten nicht zuordnen: {missing_cols}")
        sys.exit(1)
        
    print("✅ Spalten erfolgreich harmonisiert!")
    
    # --- INTELLIGENTE DATUMS-KONVERTIERUNG ---
    # Wir prüfen die erste Zeile, um das Format zu erraten
    sample_date = str(df['date'].dropna().iloc[0]).split('.')[0].strip()
    
    if sample_date.isdigit() and len(sample_date) == 4:
        print("📅 Format-Erkennung: Reine Jahreszahlen (z.B. 2022). Fixe Zeitachse...")
        df['date'] = pd.to_datetime(df['date'].astype(int).astype(str) + '-01-01', errors='coerce')
    else:
        print("📅 Format-Erkennung: Vollständige Datums-Strings. Konvertiere...")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # --- EPOCCHENFILTER (Ab 2000) ---
    df = df[df['date'].dt.year >= 2000].reset_index(drop=True)
    df = df.dropna(subset=['home_score', 'away_score', 'home_team', 'away_team'])
    
    # HIER DIE SICHERHEITSBARRIERE:
    if len(df) == 0:
        print("\n❌ PIPELINE-STOPP: Nach dem Epochenfilter sind 0 Zeilen übrig geblieben!")
        print("Das bedeutet, die Datums-Konvertierung hat nicht gegriffen. Überprüfe deine CSV.")
        sys.exit(1)
        
    df = df.sort_values('date').reset_index(drop=True)
    print(f"🏟️ Erfolg! {len(df)} MODERNE Weltmeisterschafts-Spiele im Datensatz gefunden.")
    
    # --- FEATURE ENGINEERING ---
    team_stats = {}
    home_form_attack, home_form_defense = [], []
    away_form_attack, away_form_defense = [], []
    
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        for team in [home, away]:
            if team not in team_stats:
                team_stats[team] = {'scored': [], 'conceded': []}
                
        home_form_attack.append(np.mean(team_stats[home]['scored'][-5:]) if team_stats[home]['scored'] else 1.3)
        home_form_defense.append(np.mean(team_stats[home]['conceded'][-5:]) if team_stats[home]['conceded'] else 1.3)
        
        away_form_attack.append(np.mean(team_stats[away]['scored'][-5:]) if team_stats[away]['scored'] else 1.3)
        away_form_defense.append(np.mean(team_stats[away]['conceded'][-5:]) if team_stats[away]['conceded'] else 1.3)
        
        team_stats[home]['scored'].append(row['home_score'])
        team_stats[home]['conceded'].append(row['away_score'])
        team_stats[away]['scored'].append(row['away_score'])
        team_stats[away]['conceded'].append(row['home_score'])
        
    df['home_form_attack'] = home_form_attack
    df['home_form_defense'] = home_form_defense
    df['away_form_attack'] = away_form_attack
    df['away_form_defense'] = away_form_defense
    
    df['neutral'] = 1
    
    final_features = [
        'date', 'home_team', 'away_team', 
        'home_form_attack', 'home_form_defense',
        'away_form_attack', 'away_form_defense',
        'neutral', 'home_score', 'away_score'
    ]
    
    cleaned_df = df[final_features]
    cleaned_df.to_csv(processed_path, index=False)
    print(f"✅ Daten sauber aufbereitet und unter '{processed_path}' gespeichert!")

if __name__ == "__main__":
    process_data()