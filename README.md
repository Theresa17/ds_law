# VerdictIQ (ds_law)

VerdictIQ ist eine Webanwendung zur KI-gestuetzten Einschaetzung von Diesel-Faellen.
Das Projekt besteht aus einem React-Frontend und einem Flask-Backend mit ML-Inferenz.

Wichtig: Die Ausgabe ist eine unverbindliche KI-Einschaetzung und keine Rechtsberatung.

## Inhalt

1. Voraussetzungen
2. Projektstruktur
3. Installation
4. Start (Development)
5. Funktionscheck
6. API-Endpunkte
7. Bekannte Stolpersteine
8. NPM-Skripte

## 1) Voraussetzungen

- Node.js 18+ (empfohlen: aktuelle LTS)
- Python 3.10+ (empfohlen: 3.11/3.12)
- `pip`
- PowerShell (Windows) oder ein vergleichbares Terminal

## 2) Projektstruktur

```text
.
|- src/                         # Frontend (React + Vite)
|- public/
|- backend/
|  |- api/main.py               # Flask API
|  |- services/                 # Prediction-Pipeline
|  |- artifacts/                # Trainierte Modelle/Feature-Configs
|  |- data/                     # Textdaten
|  |- requirements.txt
|- package.json
```

## 3) Installation

Im Projekt-Root (`ds_law`) ausfuehren:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m spacy download de_core_news_lg
npm install
```

Warum `de_core_news_lg`?
Die Pipeline laedt dieses spaCy-Modell direkt. Ohne dieses Modell schlagen Analysen fehl.

## 4) Start (Development)

Zwei Terminals nutzen.

Terminal 1 (Backend, im Projekt-Root):

```powershell
.\.venv\Scripts\Activate.ps1
python -m backend.api.main
```

Backend-URL:

- `http://127.0.0.1:5000`

Terminal 2 (Frontend, im Projekt-Root):

```powershell
npm run dev
```

Frontend-URL (Standard):

- `http://127.0.0.1:5173`

## 5) Funktionscheck

Backend Health:

- Browser: `http://127.0.0.1:5000/health`
- Erwartet: `{"status":"ok"}`

Root-Info:

- Browser: `http://127.0.0.1:5000/`
- Erwartet Service- und Endpoint-Infos

## 6) API-Endpunkte

- `GET /`
- `GET /health`
- `POST /predict`
- `POST /predict-file`

### `POST /predict` (JSON)

```json
{
  "text": "Fallbeschreibung ...",
  "case_id": "optional",
  "features": {
    "Kaufpreis": "25900",
    "Fahrzeugstatus": "Neuwagen"
  }
}
```

### `POST /predict-file` (multipart/form-data)

- Feldname: `file`
- Backend akzeptiert: `.txt`, `.pdf`, `.docx`
- Hinweis zur aktuellen UI: Im Frontend wird derzeit nur `.txt` durchgelassen.

## 7) Bekannte Stolpersteine

- spaCy-Modell fehlt:
  `python -m spacy download de_core_news_lg`
- Virtuelle Umgebung nicht aktiv:
  `.\.venv\Scripts\Activate.ps1`
- PowerShell blockiert Skriptausfuehrung:
  `Set-ExecutionPolicy -Scope Process Bypass`
- Port 5000 belegt:
  anderen Prozess beenden oder Port in `backend/api/main.py` anpassen
- Frontend erreicht Backend nicht:
  pruefen, ob Backend wirklich auf `127.0.0.1:5000` laeuft

## 8) NPM-Skripte

- `npm run dev` startet Vite im Dev-Modus
- `npm run build` erstellt das Production-Build
- `npm run preview` startet lokale Build-Vorschau
- `npm run lint` fuehrt ESLint aus

