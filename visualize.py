import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance():
    print("⏳ Lade modernisierte Modelle für die Visualisierung...")
    
    try:
        model_home = joblib.load("models/wm_home_goals_model.pkl")
        model_away = joblib.load("models/wm_away_goals_model.pkl")
    except FileNotFoundError:
        print("❌ Fehler: Modelle nicht gefunden.")
        return

    # Die neuen Features
    features = ['home_form_attack', 'home_form_defense', 'away_form_attack', 'away_form_defense', 'neutral']
    
    df_home = pd.DataFrame({'Feature': features, 'Importance': model_home.feature_importances_, 'Modell': 'Heimtore-Modell'})
    df_away = pd.DataFrame({'Feature': features, 'Importance': model_away.feature_importances_, 'Modell': 'Auswärtstore-Modell'})
    df_total = pd.concat([df_home, df_away])
    
    print("🎨 Generiere neues Feature-Importance-Diagramm...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    sns.barplot(x='Importance', y='Feature', hue='Modell', data=df_total, palette='viridis')
    
    plt.title('Moderne WM-KI: Welchen Einfluss hat die aktuelle Form (Letzte 5 Spiele)?', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Relative Wichtigkeit (Feature Importance)', fontsize=12)
    plt.ylabel('Feature (Einflussfaktor)', fontsize=12)
    plt.tight_layout()
    
    output_path = "plots/feature_importance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Neue Grafik unter '{output_path}' gespeichert!")

if __name__ == "__main__":
    plot_feature_importance()