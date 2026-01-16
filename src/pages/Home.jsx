import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { addAnalysis } from "../lib/storage";

function fakePredict({ mode, fileName, text }) {
  // Dummy: später ersetzt ihr das durch Backend
  const base = text?.length ? Math.min(0.95, 0.55 + text.length / 2000) : 0.78;
  return {
    klasse: "Schadensersatz",
    entscheidung: "ja",
    betrag_eur: 23542.23,
    confidence: Number(base.toFixed(2)),
    meta: { eingabe: mode === "file" ? "Datei" : "Text", fileName: fileName || null }
  };
}

export default function Home() {
  const nav = useNavigate();
  const [mode, setMode] = useState("file"); // file | text
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const canAnalyze = useMemo(() => {
    if (mode === "file") return !!file;
    return text.trim().length >= 30;
  }, [mode, file, text]);

  async function onAnalyze() {
    if (!canAnalyze) return;
    setLoading(true);
    setResult(null);

    setTimeout(() => {
      const res = fakePredict({
        mode,
        fileName: file?.name,
        text: mode === "text" ? text.trim() : "",
      });

      const id = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
      const item = {
        id,
        createdAt: new Date().toISOString(),
        inputType: mode,
        fileName: file?.name || null,
        preview: mode === "text" ? text.trim().slice(0, 120) : null,
        ...res,
      };

      addAnalysis(item);
      setResult(item);
      setLoading(false);

      // Optional: direkt zur Detailseite springen
      nav(`/history/${id}`);
    }, 800);
  }

  function reset() {
    setFile(null);
    setText("");
    setResult(null);
    setLoading(false);
  }

  return (
    <div className="card">
      <div className="card-inner">
        <h1 className="h1">Neue Analyse</h1>
        <p className="p">Lade ein Urteil hoch oder füge Text ein. Die Analyse wird (vorerst) im Frontend simuliert.</p>

        

        <div className="spacer" />

        <div className="segmented">
          <button className={`seg-btn ${mode === "file" ? "seg-active" : ""}`} onClick={() => setMode("file")} type="button">
            Datei
          </button>
          <button className={`seg-btn ${mode === "text" ? "seg-active" : ""}`} onClick={() => setMode("text")} type="button">
            Text
          </button>
        </div>

        <div className="spacer" />

        {mode === "file" ? (
          <div className="upload-box">
            <div className="upload-title">Urteil hochladen</div>
            <div className="upload-sub">Unterstützt: <strong>.pdf</strong>, <strong>.txt</strong></div>

            <div className="spacer" />

            <input className="file" type="file" accept=".pdf,.txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />

            {file && (
              <div className="file-info">
                <span className="pill">Ausgewählt</span>
                <span className="mono">{file.name}</span>
              </div>
            )}
          </div>
        ) : (
          <div className="text-input">
            <label className="label">Urteilstext</label>
            <textarea className="textarea" value={text} onChange={(e) => setText(e.target.value)} placeholder="Text hier einfügen…" />
            <div className="muted" style={{ marginTop: 8 }}>Mindestlänge: 30 • Aktuell: {text.trim().length}</div>
          </div>
        )}

        <div className="spacer" />

        <div className="row">
          <button className="btn" onClick={reset} type="button">Zurücksetzen</button>
          <button className="btn btn-primary" onClick={onAnalyze} disabled={!canAnalyze || loading} type="button">
            {loading ? "Analysiere…" : "Analysieren"}
          </button>
        </div>


        {result && (
          <>
            <div className="spacer" />
            <div className="card">
              <div className="card-inner">
                <div className="result-head">
                  <div className="result-title">Letztes Ergebnis</div>
                  <span className="pill">Confidence: {Math.round(result.confidence * 100)}%</span>
                </div>
                <div className="result-grid">
                  <div className="kv"><div className="k">Klassifikation</div><div className="v">{result.klasse}</div></div>
                  <div className="kv"><div className="k">Entscheidung</div><div className="v">{result.entscheidung}</div></div>
                  <div className="kv"><div className="k">Betrag</div><div className="v">{result.betrag_eur.toLocaleString("de-DE")} €</div></div>
                  <div className="kv"><div className="k">Eingabe</div><div className="v">{result.inputType === "file" ? result.fileName : "Text"}</div></div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

