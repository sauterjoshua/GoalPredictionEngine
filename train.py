import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib
import yaml
import numpy as np

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    data_path = config['tournaments']['world_cup']['processed_path']
    
    print(f"⏳ Lade Daten aus '{data_path}'...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Features um die Gastgeber-Flags erweitert
    features = [
        'home_form_attack', 'home_form_defense', 
        'away_form_attack', 'away_form_defense', 
        'home_market_value', 'away_market_value', 
        'home_is_host', 'away_is_host',
        'neutral'
    ]
    
    train_df = df[df['date'].dt.year < 2018].reset_index(drop=True)
    test_df = df[df['date'].dt.year >= 2018].reset_index(drop=True)
    
    X_train = train_df[features]
    y_home_train = train_df['home_score']
    y_away_train = train_df['away_score']
    
    X_test = test_df[features]
    y_home_test = test_df['home_score']
    y_away_test = test_df['away_score']
    
    print("🤖 Trainiere optimierte Modelle...")
    model_home = xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.02,
        max_depth=3, # Leicht angehoben, da wir jetzt komplexere Interaktionen (Host) erlauben können
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.5,
        reg_lambda=1.5,
        random_state=42
    )
    model_home.fit(X_train, y_home_train)
    
    model_away = xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.5,
        reg_lambda=1.5,
        random_state=42
    )
    model_away.fit(X_train, y_away_train)
    
    pred_home = model_home.predict(X_test)
    pred_away = model_away.predict(X_test)
    
    mae_home = mean_absolute_error(y_home_test, pred_home)
    mae_away = mean_absolute_error(y_away_test, pred_away)
    
    print("\n--- 📊 PERFORMANCE NACH DEM HOST-UPGRADE ---")
    print(f"Fehler Heimtore: {mae_home:.2f} Tore im Schnitt daneben.")
    print(f"Fehler Auswärtstore: {mae_away:.2f} Tore im Schnitt daneben.")
    
    baseline_home_mae = mean_absolute_error(y_home_test, np.full_like(y_home_test, 1.3, dtype=float))
    print(f"💡 Zum Vergleich der Dummy-Tipp: {baseline_home_mae:.2f}")
    
    joblib.dump(model_home, "models/wm_home_goals_model.pkl")
    joblib.dump(model_away, "models/wm_away_goals_model.pkl")
    print("\n💾 High-End-Modelle mit echtes Heimvorteil-Faktor gespeichert!")

if __name__ == "__main__":
    main()