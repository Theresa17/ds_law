# Backend – VerdictIQ

Dieses Backend bildet die technische Grundlage für die Analyse juristischer
Urteilstexte in der Web-Anwendung *VerdictIQ*.

Es umfasst sowohl den **Trainingsprozess der Machine-Learning-Modelle**
als auch die **API zur Nutzung der trainierten Modelle** durch das
Frontend (React mit Vite).

Die Architektur trennt bewusst zwischen Training, Modellablage und
produktiver Nutzung (Inference).

---

## Überblick

Das Backend besteht aus vier zentralen Bereichen:

- **Jupyter Notebook**: Training, Testing und Evaluation der Modelle  
- **Artifacts**: gespeicherte, trainierte Modelle  
- **Services**: Nutzung der Modelle zur Analyse neuer Texte  
- **API (Flask)**: Schnittstelle zwischen Frontend und Analyse-Logik  

---

## Ordnerstruktur

```text
backend/
├─ api/
│  └─ main.py
│     - Flask-API mit Endpunkten (z. B. /predict)
│     - Nimmt Anfragen vom Frontend entgegen
│     - Übergibt Texte an die Service-Schicht
│
├─ services/
│  └─ prediction_service.py
│     - Zentrale Analyse-Logik (Inference)
│     - Lädt trainierte Modellartefakte aus /artifacts
│     - Führt Vorhersagen auf neuen Texteingaben durch
│     - Enthält keine HTTP-Logik und kein Modelltraining
│
├─ artifacts/
│  ├─ model.joblib
│  └─ vectorizer.joblib
│     - Ergebnis des Trainings aus dem Jupyter Notebook
│     - Werden von der Service-Schicht geladen und genutzt
│
├─ jupyter_notebook/
│  └─ analyse.ipynb
│     - Training der Modelle
│     - Vorverarbeitung und Feature-Engineering
│     - Testing und Evaluation (z. B. Accuracy, Precision, Recall, F1, MSE)
│     - Export der finalen Modelle nach /artifacts
│
└─ data/
    - Rohdaten (Urteilstexte)
    - Verwendung für Training, Validierung und Test im Notebook
    - Die laufende API nutzt diese Rohdaten nicht direkt
