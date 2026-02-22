# Backend - VerdictIQ

Dieses Backend stellt die API und Inferenz-Pipeline fuer VerdictIQ bereit.
Es nimmt Text oder Dateien entgegen und liefert eine KI-Einschaetzung
(Anspruch ja/nein, Klasse, Betrag, Confidence).

Wichtig: Keine Rechtsberatung, nur technische Modellprognose.

## Voraussetzungen

- Python 3.10+ (empfohlen: 3.11/3.12)
- `pip`
- spaCy Modell `de_core_news_lg`

## Installation (aus Projekt-Root)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m spacy download de_core_news_lg
```

## Starten

Aus dem Projekt-Root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m backend.api.main
```

Backend laeuft dann auf:

- `http://127.0.0.1:5000`

## Endpunkte

- `GET /`
- `GET /health`
- `POST /predict`
- `POST /predict-file`

## Request-Formate

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
- Dateitypen: `.txt`, `.pdf`, `.docx`

## Architektur (kurz)

- `backend/api/main.py`:
  Flask API, Request-Parsing, Fehlerbehandlung, Dateileser fuer txt/pdf/docx.
- `backend/services/prediction_service.py`:
  Service-Layer, delegiert an `PredictionPipeline`.
- `backend/services/pipeline.py`:
  Feature-Engineering + Modellinferenz + Fallback-Logik.
- `backend/artifacts/`:
  Geladene Modellartefakte und Feature-Spalten.

## Wichtige Artefakte

Die Pipeline erwartet unter anderem:

- `backend/artifacts/claim_v2_model.joblib`
- `backend/artifacts/w2v_claim_v2.model`
- `backend/artifacts/word_weights_claim_v2.json`
- `backend/artifacts/claim_v2_feature_columns.json`
- `backend/artifacts/range_model.joblib`
- `backend/artifacts/w2v_range.model`
- `backend/artifacts/word_weights_range.json`
- `backend/artifacts/range_feature_columns.json`

## Troubleshooting

- `de_core_news_lg` fehlt:
  `python -m spacy download de_core_news_lg`
- Importfehler `backend...`:
  Backend aus Projekt-Root starten (`python -m backend.api.main`), nicht aus `backend/api`.
- Port 5000 belegt:
  Prozess beenden oder Port in `backend/api/main.py` anpassen.
- Datei kann nicht gelesen werden:
  Nur `.txt`, `.pdf`, `.docx` erlaubt.

