"""
visualize.py

Dieses Skript ist für die Modell-Diagnostik und Qualitätskontrolle zustabel.
Es lädt die trainierten XGBoost-Modelle sowie die Testdaten (WM 2018 & 2022)
und generiert vier mathematische Analyse-Grafiken im Ordner 'plots/',
die das Verhalten der KI ungeschönt offenlegen.
"""

import sys
import os
import yaml
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
    
def _load_paths() -> dict:
    """Liest die relevanten CSV-Pfade aus der config.yaml."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    wc = config["tournaments"]["world_cup"]
    return {
        "history": wc["history_path"],
        "live": wc["live_path"],
    }
 
 
def build_evaluation_frame() -> pd.DataFrame:
    """Verknüpft die letzte Vorhersage pro Spiel mit dem echten Ergebnis.
 
    Gemeinsame Helper-Funktion für alle ergebnis-abhängigen Auswertungen
    (MAE-Plot hier, Trefferquote im Dashboard). Nimmt pro match_key die
    JÜNGSTE Vorhersage und joint sie mit dem tatsächlichen FINISHED-Ergebnis.
 
    Returns:
        pd.DataFrame: Spalten u.a. date, home_team, away_team,
            pred_home_goals, pred_away_goals, real_home, real_away.
            Kann leer sein, wenn es noch keine gespielten Spiele gibt.
    """
    paths = _load_paths()
 
    # Keine History -> leeres Ergebnis
    if not os.path.exists(paths["history"]):
        return pd.DataFrame()
 
    history = pd.read_csv(paths["history"])
    if history.empty:
        return pd.DataFrame()
 
    # Pro Spiel die jüngste (beste) Vorhersage behalten
    history = history.sort_values("predicted_at")
    latest = history.groupby("match_key", as_index=False).last()
 
    # Echte Ergebnisse aus den FINISHED-Spielen laden
    live = pd.read_csv(paths["live"])
    finished = live[live["status"] == "FINISHED"].copy()
    if finished.empty:
        return pd.DataFrame()
 
    # match_key in den Live-Daten identisch aufbauen wie beim Logging:
    #   {date}_{home_team}_{away_team}   (date als reines Datum YYYY-MM-DD)
    finished["date_only"] = (
        pd.to_datetime(finished["date"], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.date.astype(str)
    )
    finished["match_key"] = (
        finished["date_only"] + "_" + finished["home_team"] + "_" + finished["away_team"]
    )
    finished = finished.rename(
        columns={"home_score": "real_home", "away_score": "real_away"}
    )
 
    # Vorhersage + Realität über den match_key zusammenführen
    merged = latest.merge(
        finished[["match_key", "real_home", "real_away"]],
        on="match_key",
        how="inner",  # nur Spiele, die BEIDES haben
    )
    return merged
 
 
def outcome_1x2(home_goals, away_goals) -> str:
    """Bestimmt den Spielausgang: '1' (Heimsieg), 'X' (Remis), '2' (Auswärtssieg)."""
    if home_goals > away_goals:
        return "1"
    if home_goals < away_goals:
        return "2"
    return "X"
 
 
def plot_mae_over_time(eval_df: pd.DataFrame):
    """Plottet den kumulierten MAE über die gespielten Spiele (chronologisch).
 
    Zeigt, ob das Modell über das Turnier hinweg besser wird (fallender MAE).
    Bei zu wenigen Spielen wird ein Hinweis-Plot erzeugt.
    """
    print("🎨 Generiere MAE-über-Zeit-Plot...")
 
    plt.figure(figsize=(10, 5))
 
    if eval_df.empty or len(eval_df) < 1:
        # Leerer Platzhalter, solange keine echten Spiele vorliegen
        plt.text(0.5, 0.5, "Noch keine gespielten Spiele\nfür MAE-Auswertung",
                 ha="center", va="center", fontsize=13, color="#888")
        plt.axis("off")
    else:
        df = eval_df.sort_values("date").reset_index(drop=True)
        # Absoluter Fehler pro Spiel (Heim + Auswärts gemittelt)
        df["abs_err"] = (
            (df["pred_home_goals"] - df["real_home"]).abs()
            + (df["pred_away_goals"] - df["real_away"]).abs()
        ) / 2.0
        # Kumulierter MAE = laufender Durchschnitt
        df["cum_mae"] = df["abs_err"].expanding().mean()
 
        x = range(1, len(df) + 1)
        plt.plot(x, df["cum_mae"], marker="o", color="#6a1b9a", linewidth=2)
        plt.title("Live-MAE über das Turnier (kumuliert)",
                  fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("Anzahl gespielter Spiele")
        plt.ylabel("Kumulierter MAE (Tore)")
        plt.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("plots/5_mae_over_time.png", dpi=300)
    plt.close()
 
 
def plot_prediction_evolution(top_n: int = 6):
    """Plottet, wie sich die erwarteten Heimtore pro Spiel über die Zeit ändern.
 
    Nutzt NUR die prediction_history.csv (keine echten Ergebnisse nötig).
    Zeigt die Spiele mit den meisten Vorhersage-Generationen.
    """
    print("🎨 Generiere Vorhersage-Entwicklungs-Plot...")
    paths = _load_paths()
 
    plt.figure(figsize=(10, 5))
 
    if not os.path.exists(paths["history"]):
        plt.text(0.5, 0.5, "Noch keine Vorhersage-Historie",
                 ha="center", va="center", fontsize=13, color="#888")
        plt.axis("off")
    else:
        history = pd.read_csv(paths["history"])
        history["predicted_at"] = pd.to_datetime(history["predicted_at"])
 
        # Spiele mit den meisten Vorhersage-Zeitpunkten auswählen
        counts = history.groupby("match_key").size().sort_values(ascending=False)
        top_keys = counts[counts >= 2].head(top_n).index.tolist()
 
        if not top_keys:
            plt.text(0.5, 0.5, "Noch zu wenig Verlauf\n(min. 2 Vorhersagen pro Spiel nötig)",
                     ha="center", va="center", fontsize=13, color="#888")
            plt.axis("off")
        else:
            for key in top_keys:
                sub = history[history["match_key"] == key].sort_values("predicted_at")
                label = f"{sub.iloc[0]['home_team']} vs {sub.iloc[0]['away_team']}"
                plt.plot(sub["predicted_at"], sub["pred_home_goals"],
                         marker="o", label=label)
            plt.title("Entwicklung der erwarteten Heimtore über die Zeit",
                      fontsize=12, fontweight="bold", pad=15)
            plt.xlabel("Zeitpunkt der Vorhersage")
            plt.ylabel("Erwartete Heimtore")
            plt.legend(fontsize=8, loc="best")
            plt.xticks(rotation=30, ha="right")
            plt.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("plots/6_prediction_evolution.png", dpi=300)
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
    
    eval_df = build_evaluation_frame()
    plot_mae_over_time(eval_df)
    plot_prediction_evolution()
    print(f"   ℹ️ Auswertung basiert auf {len(eval_df)} gespielten Spiel(en).")
    
    print("\n🎉 Alle Diagnosen fehlerfrei berechnet. Grafiken im Ordner 'plots/' aktualisiert!")


if __name__ == "__main__":
    main()