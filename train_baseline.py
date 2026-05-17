import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def load_data():
    print("⏳ Lade historische Bundesliga-Daten...")
    # Wir nehmen die Saisons 23/24 und 24/25 als Trainingsbasis
    urls = [
        "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/D1.csv"
    ]
    dfs = [pd.read_csv(url) for url in urls]
    # Kombiniere die Saisons zu einem großen DataFrame
    return pd.concat(dfs, ignore_index=True)

def engineer_features(df):
    print("⚙️ Starte team-agnostisches Feature Engineering...")
    
    # Relevante Spalten filtern und leere Zeilen löschen
    df = df[['HomeTeam', 'AwayTeam', 'HC', 'AC', 'HS', 'AS', 'HST', 'AST']].dropna()
    
    # --- HIER PASSIERT DIE ABSTRAKTION (Team-Agnostisch) ---
    # Wir berechnen den historischen Durchschnitt für JEDES Team bis zu diesem Spieltag
    # Hinweis: In einem produktionsreifen System nutzt man hierfür rollierende Durchschnitte (rolling windows)
    
    # Durchschnittlich geschossene Ecken zu Hause
    home_corners_avg = df.groupby('HomeTeam')['HC'].transform('mean')
    # Durchschnittlich zugelassene Ecken der Auswärtsmannschaft
    away_corners_conceded_avg = df.groupby('AwayTeam')['HC'].transform('mean')
    
    # Schüsse-Schnitt (Wichtiger Indikator für Ecken)
    home_shots_avg = df.groupby('HomeTeam')['HS'].transform('mean')
    away_shots_conceded_avg = df.groupby('AwayTeam')['HS'].transform('mean')
    
    # Wir bauen unseren Feature-DataFrame (X) auf
    X = pd.DataFrame({
        'home_corners_avg': home_corners_avg,
        'away_corners_conceded_avg': away_corners_conceded_avg,
        'home_shots_avg': home_shots_avg,
        'away_shots_conceded_avg': away_shots_conceded_avg
    })
    
    # Unser Target (y) sind die echten Ecken der Heimmannschaft in diesem Spiel
    y = df['HC']
    
    return X, y

def main():
    # 1. Daten laden
    raw_data = load_data()
    
    # 2. Features vorbereiten
    X, y = engineer_features(raw_data)
    
    # 3. Train-Test-Split (80% Training, 20% Validierung)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(# 4. Modell initialisieren und trainieren
"🤖 Trainiere XGBoost Regressor...")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluieren
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"✅ Training abgeschlossen! Durchschnittlicher Fehler (MAE): {mae:.2f} Ecken.")
    
    # 6. Modell für die Pipeline abspeichern
    model_path = "baseline_corner_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Modell erfolgreich als '{model_path}' gespeichert.")

if __name__ == "__main__":
    main()