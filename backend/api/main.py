"""
API für Webapp

Diese Datei stellt später die Schnittstelle zwischen
Frontend (React) und dem Analyse-Modell dar.

Geplant:
- Entgegennahme von Text / Dateien
- Laden von Modell & Vectorizer
- Rückgabe der Analyseergebnisse als JS
"""

from flask import Flask, request, jsonify
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

def dummy_predict(text: str):
    base = min(0.95, 0.55 + len(text) / 2000) if text else 0.78
    return {
        "klasse": "Schadensersatz",
        "entscheidung": "ja",
        "betrag_eur": 23542.23,
        "confidence": round(base, 2),
        "meta": {"mode": "dummy", "chars": len(text)}
    }

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if len(text) < 10:
        return jsonify({"error": "Bitte sende mindestens 10 Zeichen Text."}), 400

    return jsonify(dummy_predict(text))

@app.post("/predict-file")
def predict_file():
    # erwartet multipart/form-data mit Feldname: file
    if "file" not in request.files:
        return jsonify({"error": "Keine Datei gefunden (Feldname muss 'file' sein)."}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".txt"):
        return jsonify({"error": "Bitte nur .txt Dateien hochladen."}), 400

    raw = f.read()
    # TXT: meistens UTF-8, sonst fallback
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    text = text.strip()
    if len(text) < 10:
        return jsonify({"error": "TXT-Datei enthält zu wenig Text."}), 400

    return jsonify(dummy_predict(text))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)