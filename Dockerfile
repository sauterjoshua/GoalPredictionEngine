# 1. Nutzen eines offiziellen, schlanken Python-Images als Basis
FROM python:3.14-slim

# 2. Setzen des Arbeitsverzeichnisses im Container
WORKDIR /app

# 3. System-Abhängigkeiten installieren (wichtig für C-Extensions wie bei manchen XGBoost/Numpy Versionen)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /lib/apt/lists/*

# 4. requirements.txt kopieren und Pakete installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Den gesamten restlichen Projektcode in den Container kopieren
COPY . .

# 6. Standard-Befehl beim Starten des Containers: Die Pipeline ausführen
CMD ["python", "run_pipeline.py"]