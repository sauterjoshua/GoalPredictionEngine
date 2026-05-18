"""
visualize.py

Dieses Skript ist für die Modell-Diagnostik und Qualitätskontrolle zustabel.
Es lädt die trainierten XGBoost-Modelle sowie die Testdaten (WM 2018 & 2022)
und generiert vier mathematische Analyse-Grafiken im Ordner 'plots/',
die das Verhalten der KI ungeschönt offenlegen.
"""

import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_data_and_models() -> tuple:
    """Lädt die aufbereiteten Daten und die frosch trainierten Modelle.

    Returns:
        tuple: (DataFrame, home_model, away_model) oder (None, None, None) bei Fehler.
    """
    try:
        df = pd.read_csv("data/processed/cleaned_wm_data.csv")
        df["date"] = pd.to_datetime(df["date"])
        
        model_home = joblib.load("models/wm_home_goals_model.pkl")
        model_away = joblib.load("models/wm_away_goals_model.pkl")
        return df, model_home, model_away
    except FileNotFoundError as e:
        print(f"❌ Fehler beim Laden der Dateien: {e}")
        sys.exit(1)


def plot_feature_importance(features: list, model_home, model_away):
    """Generiert den Plot für die relative Feature-Wichtigkeit (Gain)."""
    print("🎨 1/4 Generiere Feature-Importance-Plot...")
    
    df_fi_home = pd.DataFrame({
        "Feature": features, 
        "Importance": model_home.feature_importances_, 
        "Modell": "Heimtore-Modell"
    })
    df_fi_away = pd.DataFrame({
        "Feature": features, 
        "Importance": model_away.feature_importances_, 
        "Modell": "Auswärtstore-Modell"
    })
    df_fi = pd.concat([df_fi_home, df_fi_away])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", hue="Modell", data=df_fi, palette="magma")
    plt.title("WM-Modell: Welchen Einfluss haben Marktwerte & Gastgeberstatus?", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Relative Wichtigkeit (Informationsgewinn)")
    plt.tight_layout()
    plt.savefig("plots/1_feature_importance.png", dpi=300)
    plt.close()


def plot_actual_vs_predicted(y_home_true, y_away_true, pred_home, pred_away):
    """Visualisiert die Regression zur Mitte (Wahrheit vs. Vorhersage)."""
    print("🎨 2/4 Generiere Actual-vs-Predicted-Plot...")
    plt.figure(figsize=(10, 5))
    
    # Minimaler Gauß'scher Jitter, um die Punktdichte bei Ganzzahlen sichtbar zu machen
    jitter_home = np.random.normal(0, 0.08, size=len(y_home_true))
    jitter_away = np.random.normal(0, 0.08, size=len(y_away_true))
    
    plt.scatter(y_home_true + jitter_home, pred_home, alpha=0.6, color="#6a1b9a", label="Heimtore")
    plt.scatter(y_away_true + jitter_away, pred_away, alpha=0.6, color="#d32f2f", label="Auswärtstore")
    
    # Zeichnen der perfekten Identitätslinie (y = x)
    max_val = int(max(y_home_true.max(), y_away_true.max()))
    plt.plot([0, max_val], [0, max_val], color="black", linestyle="--", linewidth=1.5, label="Perfekte Vorhersage")
    
    plt.title("Wahrheit vs. KI-Vorhersage (WM 2018 & 2022)", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Tatsächlich geschossene Tore (inkl. Jitter)")
    plt.ylabel("Von der KI vorhergesagte Tore (kontinuierlich)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/2_actual_vs_predicted.png", dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, pred_continuous):
    """Erstellt eine Heatmap der gerundeten exakten Tortipps."""
    print("🎨 3/4 Generiere Ergebnis-Matrix-Heatmap...")
    pred_round = np.round(pred_continuous).astype(int)
    
    # Beschränkung der Matrix auf max 4 Tore für maximale Übersicht im Plot
    cm = confusion_matrix(y_true, pred_round, labels=[0, 1, 2, 3, 4])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False,
                xticklabels=[0, 1, 2, 3, "4+"], yticklabels=[0, 1, 2, 3, "4+"])
    plt.title("Ergebnis-Matrix: Wie oft lag die KI goldrichtig? (Heimtore)", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Gerundeter KI-Tipp")
    plt.ylabel("Tatsächliche Tore")
    plt.tight_layout()
    plt.savefig("plots/3_ergebnis_matrix.png", dpi=300)
    plt.close()


def plot_residuals(y_home_true, y_away_true, pred_home, pred_away):
    """Plottet die Fehler-Anatomie (Verteilung der Residuen)."""
    print("🎨 4/4 Generiere Fehler-Residual-Histogramm...")
    residuals_home = y_home_true - pred_home
    residuals_away = y_away_true - pred_away
    
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals_home, kde=True, color="#6a1b9a", label="Fehler Heimtore", alpha=0.5, bins=15)
    sns.histplot(residuals_away, kde=True, color="#d32f2f", label="Fehler Auswärtstore", alpha=0.5, bins=15)
    
    # Die Null-Linie zeigt an, wo der Fehler 0 wäre
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
    plt.title("Fehler-Anatomie: Verteilung der Abweichungen (Residuen)", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Abweichung (Tatsächliche Tore - Vorhersage)")
    plt.ylabel("Anzahl der Spiele")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/4_fehler_anatomie.png", dpi=300)
    plt.close()


def main():
    df, model_home, model_away = load_data_and_models()
    
    features = [
        "home_form_attack", "home_form_defense", "away_form_attack", "away_form_defense", 
        "home_market_value", "away_market_value", "home_is_host", "away_is_host", "neutral"
    ]
    
    # Isolation der Out-of-Time Testdaten (WM 2018 & 2022)
    test_df = df[df["date"].dt.year >= 2018].reset_index(drop=True)
    X_test = test_df[features]
    y_home_true = test_df["home_score"]
    y_away_true = test_df["away_score"]
    
    # Inferenz auf Testset ausführen (Negative Vorhersagen via clip auf 0 abfangen)
    pred_home = np.clip(model_home.predict(X_test), 0, None)
    pred_away = np.clip(model_away.predict(X_test), 0, None)
    
    # Globales Styling für die Plots festlegen
    sns.set_theme(style="whitegrid")
    
    # Aufruf der einzelnen Plot-Funktionen
    plot_feature_importance(features, model_home, model_away)
    plot_actual_vs_predicted(y_home_true, y_away_true, pred_home, pred_away)
    plot_confusion_matrix(y_home_true, pred_home)
    plot_residuals(y_home_true, y_away_true, pred_home, pred_away)
    
    print("\n🎉 Alle Diagnosen fehlerfrei berechnet. Grafiken im Ordner 'plots/' aktualisiert!")


if __name__ == "__main__":
    main()