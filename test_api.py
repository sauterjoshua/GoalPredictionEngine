"""
test_api.py — Quick-Check, ob Token und API funktionieren.
Dieses Script ist nur zum Ausprobieren, kommt später weg.
"""

import os
import requests
from dotenv import load_dotenv

# Lädt die .env-Datei und macht ihre Variablen verfügbar
load_dotenv()

# Token aus Umgebungsvariable holen
TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")

if not TOKEN:
    print("❌ Kein Token gefunden! Check deine .env-Datei.")
    exit(1)

print(f"✅ Token geladen: {TOKEN[:8]}... (gekürzt für Sicherheit)")

# Test-Request: Hole Info über die WM 2026
url = "https://api.football-data.org/v4/competitions/WC"
headers = {"X-Auth-Token": TOKEN}

print(f"\n🌐 Teste API-Zugriff: {url}")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(f"✅ API funktioniert!")
    print(f"   Competition: {data.get('name')}")
    print(f"   Code: {data.get('code')}")
    print(f"   Letzte aktualisiert: {data.get('lastUpdated')}")
else:
    print(f"❌ Fehler: HTTP {response.status_code}")
    print(f"   Response: {response.text}")