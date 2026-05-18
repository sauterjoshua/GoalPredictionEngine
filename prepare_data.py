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
    market_value_path = "data/raw/fifa_player_performance_market_value.csv"
    
    print(f"⏳ Lade Match-Daten aus {raw_path}...")
    df_matches = pd.read_csv(raw_path)
    print(f"⏳ Lade Marktwert-Daten aus {market_value_path}...")
    df_market = pd.read_csv(market_value_path)

    # --- SPALTEN-MAPPING ---
    mapping = {}
    for col in ['date', 'Datetime', 'Year', 'year', 'Date']:
        if col in df_matches.columns: mapping[col] = 'date'; break
    for col in ['home_team', 'Home Team Name', 'HomeTeam', 'Home Team']:
        if col in df_matches.columns: mapping[col] = 'home_team'; break
    for col in ['away_team', 'Away Team Name', 'AwayTeam', 'Away Team']:
        if col in df_matches.columns: mapping[col] = 'away_team'; break
    for col in ['home_score', 'Home Team Goals', 'FTHG', 'Home Goals']:
        if col in df_matches.columns: mapping[col] = 'home_score'; break
    for col in ['away_score', 'Away Team Goals', 'FTAG', 'Away Goals']:
        if col in df_matches.columns: mapping[col] = 'away_score'; break

    df_matches = df_matches.rename(columns=mapping)
    
    # Intelligente Datums-Konvertierung
    sample_date = str(df_matches['date'].dropna().iloc[0]).split('.')[0].strip()
    if sample_date.isdigit() and len(sample_date) == 4:
        df_matches['date'] = pd.to_datetime(df_matches['date'].astype(int).astype(str) + '-01-01', errors='coerce')
    else:
        df_matches['date'] = pd.to_datetime(df_matches['date'], errors='coerce')

    df_matches = df_matches.sort_values('date').reset_index(drop=True)
    df_matches = df_matches.dropna(subset=['home_score', 'away_score', 'home_team', 'away_team'])

    # --- FEATURE ENGINEERING: FORM ---
    print("📈 Berechne Formkurven aus allen Länderspielen...")
    team_stats = {}
    home_form_attack, home_form_defense = [], []
    away_form_attack, away_form_defense = [], []
    
    for idx, row in df_matches.iterrows():
        home = row['home_team']
        away = row['away_team']
        for team in [home, away]:
            if team not in team_stats: team_stats[team] = {'scored': [], 'conceded': []}
                
        home_form_attack.append(np.mean(team_stats[home]['scored'][-5:]) if team_stats[home]['scored'] else 1.3)
        home_form_defense.append(np.mean(team_stats[home]['conceded'][-5:]) if team_stats[home]['conceded'] else 1.3)
        away_form_attack.append(np.mean(team_stats[away]['scored'][-5:]) if team_stats[away]['scored'] else 1.3)
        away_form_defense.append(np.mean(team_stats[away]['conceded'][-5:]) if team_stats[away]['conceded'] else 1.3)
        
        team_stats[home]['scored'].append(row['home_score'])
        team_stats[home]['conceded'].append(row['away_score'])
        team_stats[away]['scored'].append(row['away_score'])
        team_stats[away]['conceded'].append(row['home_score'])
        
    df_matches['home_form_attack'] = home_form_attack
    df_matches['home_form_defense'] = home_form_defense
    df_matches['away_form_attack'] = away_form_attack
    df_matches['away_form_defense'] = away_form_defense

    # --- HISTORISCHE GASTGEBER-LOGIK (NEU!) ---
    print("🏠 Injiziere echten Heimvorteil (Gastgeber-Länder)...")
    world_cup_hosts = {
        2002: ['South Korea', 'Japan'],
        2006: ['Germany'],
        2010: ['South Africa'],
        2014: ['Brazil'],
        2018: ['Russia'],
        2022: ['Qatar'],
        2026: ['USA', 'United States', 'Canada', 'Mexico']
    }

    home_is_host = []
    away_is_host = []

    for idx, row in df_matches.iterrows():
        year = row['date'].year
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Prüfen, ob das Jahr im Host-Diktionary existiert und das Team darin liegt
        if year in world_cup_hosts and home_team in world_cup_hosts[year]:
            home_is_host.append(1)
        else:
            home_is_host.append(0)
            
        if year in world_cup_hosts and away_team in world_cup_hosts[year]:
            away_is_host.append(1)
        else:
            away_is_host.append(0)

    df_matches['home_is_host'] = home_is_host
    df_matches['away_is_host'] = away_is_host

    # --- FILTERN AUF MODERNE WM ---
    if 'tournament' in df_matches.columns:
        df_wc = df_matches[df_matches['tournament'] == 'FIFA World Cup'].copy()
    else:
        df_wc = df_matches.copy()
    df_wc = df_wc[df_wc['date'].dt.year >= 2000].reset_index(drop=True)

    # --- MARKTWERTE ---
    print("💰 Aggregiere Spieler-Marktwerte...")
    df_market_sorted = df_market.sort_values('market_value_million_eur', ascending=False)
    top_26_players = df_market_sorted.groupby('nationality').head(26)
    squad_values = top_26_players.groupby('nationality')['market_value_million_eur'].sum().reset_index()
    squad_values.columns = ['team_name', 'squad_market_value']

    df_wc = df_wc.merge(squad_values, left_on='home_team', right_on='team_name', how='left').drop(columns=['team_name'])
    df_wc = df_wc.rename(columns={'squad_market_value': 'home_market_value'})
    df_wc = df_wc.merge(squad_values, left_on='away_team', right_on='team_name', how='left').drop(columns=['team_name'])
    df_wc = df_wc.rename(columns={'squad_market_value': 'away_market_value'})

    median_value = squad_values['squad_market_value'].median()
    df_wc['home_market_value'] = df_wc['home_market_value'].fillna(median_value)
    df_wc['away_market_value'] = df_wc['away_market_value'].fillna(median_value)

    df_wc['neutral'] = 1

    final_features = [
        'date', 'home_team', 'away_team', 
        'home_form_attack', 'home_form_defense',
        'away_form_attack', 'away_form_defense',
        'home_market_value', 'away_market_value', 
        'home_is_host', 'away_is_host', # UNSERE NEUEN COPILOTEN
        'neutral', 'home_score', 'away_score'
    ]
    
    cleaned_df = df_wc[final_features]
    cleaned_df.to_csv(processed_path, index=False)
    print(f"✅ Pipeline erfolgreich durchgelaufen!")

if __name__ == "__main__":
    process_data()