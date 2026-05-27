# 1. Stabiles, schlankes Python-Image (3.12 LTS – alle Wheels verfügbar)
FROM python:3.12-slim

# 2. Arbeitsverzeichnis
WORKDIR /app

# 3. System-Abhängigkeiten für C-Extensions (XGBoost, NumPy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Nur Pipeline-relevante Pakete installieren (kein Selenium, kein GPU-Kram)
COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

# 5. Projektcode kopieren (.dockerignore filtert Secrets, virtuelle Envs,
#    generierte Outputs — data/raw/ mit historischen CSVs bleibt drin)
COPY . .

# 6. Pipeline starten
CMD ["python", "run_pipeline.py"]