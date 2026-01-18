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

# Service importieren
from backend.services.prediction_service import predict

app = Flask(__name__)
CORS(app)


@app.get("/")
def root():
    return jsonify(
        {
            "service": "ds-law-backend",
            "status": "running",
            "endpoints": ["/health", "/predict", "/predict-file"],
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict_text():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    try:
        result = predict(text, source="text")
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Interner Serverfehler"}), 500


@app.post("/predict-file")
def predict_file():
    # erwartet multipart/form-data mit Feldname: file
    if "file" not in request.files:
        return jsonify({"error": "Keine Datei gefunden (Feldname muss 'file' sein)."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Dateiname fehlt."}), 400

    # nur TXT (wie du wolltest)
    if not f.filename.lower().endswith(".txt"):
        return jsonify({"error": "Bitte nur .txt Dateien hochladen."}), 400

    raw = f.read()

    # TXT: meist UTF-8, sonst fallback
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    text = text.strip()

    try:
        result = predict(text, source="file", file_name=f.filename)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Interner Serverfehler"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)