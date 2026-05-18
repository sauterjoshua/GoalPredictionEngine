import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    data_path = config['tournaments']['world_cup']['processed_path']
    
    print(f"⏳ Lade aufbereitete moderne WM-Daten aus '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # HIER GEÄNDERT: Die neuen Form-Features statt der alten Hist-Features
    features = ['home_form_attack', 'home_form_defense', 'away_form_attack', 'away_form_defense', 'neutral']
    X = df[features]
    
    # Wir haben weiterhin zwei Zielvariablen: Heimtore und Auswärtstore
    y_home = df['home_score']
    y_away = df['away_score']
    
    # 2. Train-Test-Split (80% Training, 20% Test)
    X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
    _, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)
    
    # 3. Modell 1: Heimtore vorhersagen
    print("🤖 Trainiere modernes Modell für HEIM-Tore...")
    model_home = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model_home.fit(X_train, y_home_train)
    
    # 4. Modell 2: Auswärtstore vorhersagen
    print("🤖 Trainiere modernes Modell für AUSWÄRTS-Tore...")
    model_away = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model_away.fit(X_train, y_away_train)
    
    # 5. Evaluierung
    pred_home = model_home.predict(X_test)
    pred_away = model_away.predict(X_test)
    
    mae_home = mean_absolute_error(y_home_test, pred_home)
    mae_away = mean_absolute_error(y_away_test, pred_away)
    
    print("\n--- 📊 Neue Modell-Performance (MAE) ---")
    print(f"Fehler Heimtore: {mae_home:.2f} Tore im Schnitt daneben.")
    print(f"Fehler Auswärtstore: {mae_away:.2f} Tore im Schnitt daneben.")
    
    # 6. Modelle abspeichern
    joblib.dump(model_home, "models/wm_home_goals_model.pkl")
    joblib.dump(model_away, "models/wm_away_goals_model.pkl")
    print("\n💾 Beide modernisierten Modelle erfolgreich im Ordner 'models/' gespeichert!")

if __name__ == "__main__":
    main()