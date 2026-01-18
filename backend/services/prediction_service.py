"""
prediction_service.py

Zweck:
- Enthält die Logik zur Analyse von Urteilstexten
- Wird von der Flask-API (backend/api/main.py) aufgerufen
- Das Training der Modelle erfolgt ausschließlich in den Jupyter Notebooks

Aktueller Stand:
- Es wird eine einfache Dummy-Analyse verwendet
- Die Struktur ist bewusst minimal gehalten
- Später kann hier problemlos ein trainiertes Modell integriert werden
"""


def predict(text: str, source: str = "text", file_name: str | None = None) -> dict:
    """
    Führt eine Analyse auf Basis des übergebenen Textes durch.

    Parameter:
    - text: Urteilstext als String
    - source: 'text' oder 'file'
    - file_name: Dateiname bei Datei-Uploads (optional)

    Rückgabe:
    - Dictionary mit Analyseergebnis
    """
    text = (text or "").strip()

    if len(text) < 10:
        raise ValueError("Der Text ist zu kurz für eine Analyse.")

    # Dummy-Bewertung basierend auf Textlänge
    confidence = min(0.95, 0.55 + len(text) / 2000)

    return {
        "klasse": "Schadensersatz",
        "entscheidung": "ja",
        "betrag_eur": 23542.23,
        "confidence": round(confidence, 2),
        "meta": {
            "mode": "dummy",
            "eingabe": source,
            "fileName": file_name,
            "chars": len(text),
        },
    }


# Hinweis:
# Sobald ein trainiertes Modell vorliegt, kann die Dummy-Logik
# durch eine modellbasierte Vorhersage ersetzt werden.
# Die API-Schnittstellen müssen dafür nicht angepasst werden.
