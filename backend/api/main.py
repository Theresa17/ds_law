"""
API für Webapp

Diese Datei stellt später die Schnittstelle zwischen
Frontend (React) und dem Analyse-Modell dar.

Geplant:
- Entgegennahme von Text / Dateien
- Laden von Modell & Vectorizer
- Rückgabe der Analyseergebnisse als JS
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.get("/")
def root():
    return jsonify ({
        "service": "ds-law-backend",
        "status": "running", 
        "endpoints":["/health", "/predict"]
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


#Testet ob Backend funktioniert
@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if len(text) < 10:
        return jsonify({"error": "Bitte sende mindestens 10 Zeichen Text."}), 400

    # Dummy-Antwort (später ersetzten durch echtes Modell)
    return jsonify({
        "klasse": "Schadensersatz",
        "entscheidung": "ja",
        "betrag_eur": 23542.23,
        "confidence": 0.78,
        "meta": {"mode": "dummy", "chars": len(text)}
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)