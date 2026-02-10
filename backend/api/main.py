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
from io import BytesIO
from typing import Callable
from flask_cors import CORS

# Service importieren
from backend.services.prediction_service import predict

try:
    from docx import Document
except Exception:  # pragma: no cover
    Document = None

try:
    import PyPDF2
except Exception:  # pragma: no cover
    PyPDF2 = None

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
    case_id = (data.get("case_id") or "").strip()
    features = data.get("features")
    if not isinstance(features, dict):
        features = None

    try:
        result = predict(
            text,
            source="text",
            case_id=case_id or None,
            features=features,
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Interner Serverfehler"}), 500


def _read_txt(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _read_pdf(raw: bytes) -> str:
    if PyPDF2 is None:
        raise ValueError("PDF-Unterstützung fehlt (PyPDF2 nicht installiert).")
    reader = PyPDF2.PdfReader(BytesIO(raw))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _read_docx(raw: bytes) -> str:
    if Document is None:
        raise ValueError("DOCX-Unterstützung fehlt (python-docx nicht installiert).")
    doc = Document(BytesIO(raw))
    return "\n".join(p.text for p in doc.paragraphs)


@app.post("/predict-file")
def predict_file():
    # erwartet multipart/form-data mit Feldname: file
    if "file" not in request.files:
        return jsonify({"error": "Keine Datei gefunden (Feldname muss 'file' sein)."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Dateiname fehlt."}), 400

    filename = f.filename
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    readers: dict[str, Callable[[bytes], str]] = {
        "txt": _read_txt,
        "pdf": _read_pdf,
        "docx": _read_docx,
    }
    if ext not in readers:
        return jsonify({"error": "Bitte nur .txt, .pdf oder .docx Dateien hochladen."}), 400

    raw = f.read()
    try:
        text = readers[ext](raw)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Datei konnte nicht gelesen werden."}), 400

    text = (text or "").strip()
    if not text:
        return jsonify({"error": "Datei enthält keinen lesbaren Text."}), 400

    try:
        result = predict(
            text,
            source="file",
            file_name=f.filename,
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Interner Serverfehler"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
