import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_data_and_models():
    """Lädt die Testdaten und die trainierten XGBoost-Modelle."""
    try:
        df = pd.read_csv("data/processed/cleaned_wm_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        
        model_home = joblib.load("models/wm_home_goals_model.pkl")
        model_away = joblib.load("models/wm_away_goals_model.pkl")
        return df, model_home, model_away
    except FileNotFoundError as e:
        print(f"❌ Fehler beim Laden der Dateien: {e}")
        return None, None, None

def plot_all():
    df, model_home, model_away = load_data_and_models()
    if df is None: return

    # Features exakt wie im Training
    features = [
        'home_form_attack', 'home_form_defense', 'away_form_attack', 'away_form_defense', 
        'home_market_value', 'away_market_value', 'home_is_host', 'away_is_host', 'neutral'
    ]
    
    # Wir isolieren die Validierungsdaten (WM 2018 & 2022)
    test_df = df[df['date'].dt.year >= 2018].reset_index(drop=True)
    X_test = test_df[features]
    y_home_true = test_df['home_score']
    y_away_true = test_df['away_score']
    
    # Vorhersagen generieren
    pred_home = np.clip(model_home.predict(X_test), 0, None)
    pred_away = np.clip(model_away.predict(X_test), 0, None)
    
    # Styling setzen
    sns.set_theme(style="whitegrid")
    
    # =========================================================================
    # PLOT 1: FEATURE IMPORTANCE (Aktualisiert)
    # =========================================================================
    print("🎨 1/4 Generiere Feature-Importance-Plot...")
    df_fi_home = pd.DataFrame({'Feature': features, 'Importance': model_home.feature_importances_, 'Modell': 'Heimtore-Modell'})
    df_fi_away = pd.DataFrame({'Feature': features, 'Importance': model_away.feature_importances_, 'Modell': 'Auswärtstore-Modell'})
    df_fi = pd.concat([df_fi_home, df_fi_away])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', hue='Modell', data=df_fi, palette='magma')
    plt.title('WM-Modell: Welchen Einfluss haben Marktwerte & Gastgeberstatus?', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Relative Wichtigkeit (Informationsgewinn)')
    plt.tight_layout()
    plt.savefig("plots/1_feature_importance.png", dpi=300)
    plt.close()

    # =========================================================================
    # PLOT 2: ACTUAL VS PREDICTED (Wahrheit vs. Vorhersage)
    # =========================================================================
    print("🎨 2/4 Generiere Actual-vs-Predicted-Plot...")
    plt.figure(figsize=(10, 5))
    
    # Da Tore Ganzzahlen sind, fügen wir minimales Rauschen (Jitter) hinzu, damit man die Punktdichte sieht
    jitter_home = np.random.normal(0, 0.08, size=len(y_home_true))
    jitter_away = np.random.normal(0, 0.08, size=len(y_away_true))
    
    plt.scatter(y_home_true + jitter_home, pred_home, alpha=0.6, color='#6a1b9a', label='Heimtore')
    plt.scatter(y_away_true + jitter_away, pred_away, alpha=0.6, color='#d32f2f', label='Auswärtstore')
    
    # Perfekte Diagonale zeichnen
    max_val = int(max(y_home_true.max(), y_away_true.max()))
    plt.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=1.5, label='Perfekte Vorhersage')
    
    plt.title('Wahrheit vs. KI-Vorhersage (WM 2018 & 2022)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Tatsächlich geschossene Tore (inkl. Jitter)')
    plt.ylabel('Von der KI vorhergesagte Tore (kontinuierlich)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/2_actual_vs_predicted.png", dpi=300)
    plt.close()

    # =========================================================================
    # PLOT 3: ERGEBNIS-MATRIX (Confusion Matrix der gerundeten Tore)
    # =========================================================================
    print("🎨 3/4 Generiere Ergebnis-Matrix-Heatmap...")
    pred_home_round = np.round(pred_home).astype(int)
    
    # Matrix berechnen (beschränkt auf max 4 Tore für bessere Übersicht)
    cm = confusion_matrix(y_home_true, pred_home_round, labels=[0, 1, 2, 3, 4])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
                xticklabels=[0, 1, 2, 3, '4+'], yticklabels=[0, 1, 2, 3, '4+'])
    plt.title('Ergebnis-Matrix: Wie oft lag die KI goldrichtig? (Heimtore)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Gerundeter KI-Tipp')
    plt.ylabel('Tatsächliche Tore')
    plt.tight_layout()
    plt.savefig("plots/3_ergebnis_matrix.png", dpi=300)
    plt.close()

    # =========================================================================
    # PLOT 4: RESIDUAL HISTOGRAM (Die Fehler-Anatomie)
    # =========================================================================
    print("🎨 4/4 Generiere Fehler-Residual-Histogramm...")
    residuals_home = y_home_true - pred_home
    residuals_away = y_away_true - pred_away
    
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals_home, kde=True, color='#6a1b9a', label='Fehler Heimtore', alpha=0.5, bins=15)
    sns.histplot(residuals_away, kde=True, color='#d32f2f', label='Fehler Auswärtstore', alpha=0.5, bins=15)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    plt.title('Fehler-Anatomie: Verteilung der Abweichungen (Residuen)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Abweichung (Tatsächliche Tore - Vorhersage)')
    plt.ylabel('Anzahl der Spiele')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/4_fehler_anatomie.png", dpi=300)
    plt.close()
    
    print("\n🎉 Alle 4 Grafiken wurden erfolgreich im Ordner 'plots/' gespeichert!")

if __name__ == "__main__":
    plot_all()