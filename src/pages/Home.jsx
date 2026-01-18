import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { addAnalysis } from "../lib/storage";

const API_BASE = "http://127.0.0.1:5000";

export default function Home() {
  const nav = useNavigate();

  const [mode, setMode] = useState("file"); // "file" | "text"
  const [file, setFile] = useState(null); // .txt Datei
  const [text, setText] = useState(""); // Text-Eingabe

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const canAnalyze = useMemo(() => {
    if (mode === "file") return !!file;
    return text.trim().length >= 30;
  }, [mode, file, text]);

  async function onAnalyze() {
    if (!canAnalyze || loading) return;

    setLoading(true);
    setError(null);

    try {
      let data;

      if (mode === "text") {
        console.log("Sende TEXT an Backend:", text.trim().slice(0, 100));
        const res = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text.trim() }),
        });

        const json = await res.json().catch(() => ({}));
        console.log("Antwort von Backend (TEXT):", json);

        if (!res.ok)
          throw new Error(json.error || "Fehler bei der Analyse (Text).");
        data = json;
      } else {
        // mode === "file" (TXT)
        const fd = new FormData();
        fd.append("file", file); // Feldname MUSS "file" heißen

        console.log(
          "Sende TXT-Datei an Backend:",
          file?.name,
          file?.size,
          file?.type
        );

        const res = await fetch(`${API_BASE}/predict-file`, {
          method: "POST",
          body: fd,
        });

        const json = await res.json().catch(() => ({}));
        console.log("Antwort von Backend (TXT-Datei):", json);
        if (!res.ok)
          throw new Error(json.error || "Fehler bei der Analyse (Datei).");
        data = json;
      }

      const id = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());

      const item = {
        id,
        createdAt: new Date().toISOString(),
        inputType: mode,
        fileName: mode === "file" ? file?.name || null : null,
        preview: mode === "text" ? text.trim().slice(0, 120) : null,
        ...data,
      };

      addAnalysis(item);

      // Direkt zu Analysedetails springen
      nav(`/history/${id}`);
    } catch (e) {
      console.error("Analyse-Fehler:", e);
      setError(e?.message || "Unbekannter Fehler");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setText("");
    setError(null);
    setLoading(false);
  }

  return (
    <div className="card">
      <div className="card-inner">
        <h1 className="h1">Neue Analyse</h1>
        <p className="p">
          Lade eine <strong>.txt</strong>-Datei hoch oder füge Text ein. Die
          Analyse wird über das Backend aufgerufen.
        </p>

        <div className="spacer" />

        <div className="segmented">
          <button
            className={`seg-btn ${mode === "file" ? "seg-active" : ""}`}
            onClick={() => setMode("file")}
            type="button"
          >
            Datei
          </button>
          <button
            className={`seg-btn ${mode === "text" ? "seg-active" : ""}`}
            onClick={() => setMode("text")}
            type="button"
          >
            Text
          </button>
        </div>

        <div className="spacer" />

        {mode === "file" ? (
          <div className="upload-box">
            <div className="upload-title">TXT hochladen</div>
            <div className="upload-sub">
              Unterstützt: <strong>.txt</strong>
            </div>

            <div className="spacer" />

            <input
              className="file"
              type="file"
              accept=".txt"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />

            {file && (
              <div className="file-info">
                <span className="pill pill-strong">Ausgewählt</span>
                <span className="mono">{file.name}</span>
              </div>
            )}
          </div>
        ) : (
          <div className="text-input">
            <label className="label">Urteilstext</label>
            <textarea
              className="textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Text hier einfügen…"
            />
            <div className="muted" style={{ marginTop: 8 }}>
              Mindestlänge: 30 • Aktuell: {text.trim().length}
            </div>
          </div>
        )}

        <div className="spacer" />

        <div className="row row-center">
          <button
            className="btn"
            onClick={reset}
            type="button"
            disabled={loading}
          >
            Zurücksetzen
          </button>

          <button
            className="btn btn-primary"
            onClick={onAnalyze}
            disabled={!canAnalyze || loading}
            type="button"
            aria-disabled={!canAnalyze || loading}
          >
            {loading ? "Analysiere…" : "Analysieren"}
          </button>
        </div>

        {error && (
          <>
            <div className="spacer" />
            <div
              className="pill"
              style={{
                borderColor: "rgba(242,104,104,0.45)",
                background: "rgba(242,104,104,0.12)",
              }}
            >
              {error}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
