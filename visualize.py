import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance():
    print("⏳ Lade trainierte Modelle für die Visualisierung...")
    
    # 1. Modelle laden
    try:
        model_home = joblib.load("models/wm_home_goals_model.pkl")
        model_away = joblib.load("models/wm_away_goals_model.pkl")
    except FileNotFoundError:
        print("❌ Fehler: Trainierte Modelle wurden nicht im Ordner 'models/' gefunden.")
        return

    # Deine exakten Feature-Namen aus dem Training
    features = ['home_hist_attack', 'home_hist_defense', 'away_hist_attack', 'away_hist_defense', 'neutral']
    
    # 2. DataFrames für die Wichtigkeiten erstellen
    df_home = pd.DataFrame({
        'Feature': features,
        'Importance': model_home.feature_importances_,
        'Modell': 'Heimtore-Modell'
    })
    
    df_away = pd.DataFrame({
        'Feature': features,
        'Importance': model_away.feature_importances_,
        'Modell': 'Auswärtstore-Modell'
    })
    
    # Beide Datensätze für den Plot kombinieren
    df_total = pd.concat([df_home, df_away])
    
    # 3. Das Diagramm mit Seaborn stylen
    print("🎨 Generiere professionelles Feature-Importance-Diagramm...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Ein schickes Balkendiagramm (Barplot) erstellen
    plot = sns.barplot(
        x='Importance', 
        y='Feature', 
        hue='Modell', 
        data=df_total, 
        palette='viridis'
    )
    
    # Titel und Achsen beschriften
    plt.title('WM 2026 KI-Modell: Welche Features beeinflussen die Tore am meisten?', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Relative Wichtigkeit (Feature Importance)', fontsize=12)
    plt.ylabel('Feature (Einflussfaktor)', fontsize=12)
    plt.tight_layout()
    
    # 4. Grafik abspeichern
    output_path = "plots/feature_importance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Grafik erfolgreich unter '{output_path}' gespeichert! Schau sie dir mal an.")

if __name__ == "__main__":
    plot_feature_importance()