Dieses Projekt stellt eine vollständige, produktionsreife Machine-Learning-Pipeline dar, die die Tor-Ergebnisse für die FIFA Weltmeisterschaft 2026 prognostiziert. Der gesamte Code ist modular aufgebaut und gliedert sich in drei Kernphasen:
1. Data Engineering & Feature Pipeline (prepare_data.py)

    Bereinigung & Harmonisierung: Lädt historische Spielergebnisse sowie globale Marktwert-Tabellen und führt die Spuren über ein automatisches Mapping fehlerfrei zusammen.

    Zeit-Filterung: Eliminiert veraltete Taktik-Epochen (Concept Drift), indem nur moderne Spiele ab dem Jahr 2000 betrachtet werden.

    Formkurven-Berechnung: Kalkuliert dynamisch die sportliche Verfassung (Tore/Gegentore der letzten 5 Spiele) über alle Länderspiele direkt vor einem Turnier.

    Finanz-Aggregation: Sortiert Einzelspieler nach Marktwert und summiert die Top 26 Akteure zu einem realistischen Gesamt-Kaderwert pro Nation.

    Gastgeber-Injektion: Identifiziert echte WM-Ausrichter (z. B. Deutschland 2006 oder USA 2026), um den atmosphärischen Heimvorteil mathematisch greifbar zu machen.

2. Modell-Training & Validierung (train.py)

    Architektur: Trainiert zwei getrennte, hoch-regulierte XGBoost-Regressoren – eines für die Tore des nominellen Heimteams, eines für das Auswärtsteam.

    Zeitbasierter Split: Verhindert das Erschummeln von Trends (Data Leakage). Das Modell lernt ausschließlich auf den WMs 2000–2014 und wird an den für die KI völlig unbekannten Turnieren 2018 und 2022 validiert.

    Overfitting-Schutz: Durch extrem flache Entscheidungsbäume (max_depth=3) und strenge Regularisierungsstrafen wird verhindert, dass die KI statistisches Rauschen auswendig lernt. Das Modell agiert dadurch hocheffizient (Regression zur Mitte).

3. Diagnostik & Live-Inferenz (visualize.py & predict.py)

    Automatisierte Qualitätskontrolle: Erstellt bei jedem Durchlauf vier mathematische Diagnose-Plots (Feature Importance, Wahrheit vs. Vorhersage, Ergebnis-Matrix und Fehler-Anatomie) im Ordner plots/.

    WM 2026 Live-Simulator: Lädt die fertig trainierten Modelle aus der Cold-Storage-Datei (.pkl) und berechnet für jede eingegebene Paarung der kommenden WM 2026 in Sekundenbruchteilen den exakten, mathematisch wahrscheinlichsten Ergebnistipp.