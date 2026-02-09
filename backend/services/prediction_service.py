"""
prediction_service.py

Zweck:
- Enthält die Logik zur Analyse von Urteilstexten
- Wird von der Flask-API (backend/api/main.py) aufgerufen
- Das Training der Modelle erfolgt ausschließlich in den Jupyter Notebooks
"""

from backend.services.pipeline import PredictionPipeline

PIPELINE = PredictionPipeline()


def predict(
    text: str,
    source: str = "text",
    file_name: str | None = None,
    case_id: str | None = None,
    features: dict | None = None,
) -> dict:
    """
    Führt eine Analyse auf Basis des übergebenen Textes durch.

    Parameter:
    - text: Urteilstext als String
    - source: 'text' oder 'file'
    - file_name: Dateiname bei Datei-Uploads (optional)

    Rückgabe:
    - Dictionary mit Analyseergebnis
    """
    if case_id:
        result = PIPELINE.predict_from_case_id(case_id)
    else:
        result = PIPELINE.predict_from_payload(text, features)

    result.setdefault("meta", {})
    result["meta"].update(
        {
            "eingabe": source,
            "fileName": file_name,
            "chars": len((text or "").strip()),
        }
    )

    return result
