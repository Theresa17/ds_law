import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { loadAnalyses } from "../lib/storage";
import { allAnalysesCsv, downloadTextFile } from "../lib/csv";

function fmtDate(iso) {
  try {
    return new Date(iso).toLocaleString("de-DE");
  } catch {
    return iso;
  }
}

export default function History() {
  const [items, setItems] = useState([]);

  useEffect(() => {
    setItems(loadAnalyses());
  }, []);

  return (
    <div className="card">
      <div className="card-inner">
        <h1 className="h1">Verlauf</h1>
        <p className="p">
          Alle bisher durchgeführten Analysen (gespeichert im Browser).
        </p>

        <div className="spacer" />

        <div className="row">
          <button
            className="btn"
            type="button"
            disabled={items.length === 0}
            onClick={() => {
              const csv = allAnalysesCsv(items);
              downloadTextFile("analysen_verlauf.csv", csv);
            }}
          >
            Verlauf als CSV herunterladen
          </button>
        </div>

        <div className="spacer" />

        {items.length === 0 ? (
          <p className="p">
            Noch keine Analysen vorhanden. Starte eine Analyse auf der
            Home-Seite.
          </p>
        ) : (
          <div className="history-list">
            {items.map((a) => {
              const inputLabel =
                a.inputType === "form"
                  ? `Formular${a.preview ? ` · ${a.preview}` : ""}`
                  : `Text: ${a.preview || ""}`;

              return (
                <Link
                  key={a.id}
                  to={`/history/${a.id}`}
                  className="kv"
                  style={{ display: "block" }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      gap: 10,
                      flexWrap: "wrap",
                    }}
                  >
                    <div>
                      <strong>{a.klasse}</strong>{" "}
                      <span className="muted">· {a.entscheidung}</span>
                    </div>
                    <div className="muted">{fmtDate(a.createdAt)}</div>
                  </div>
                  <div className="muted" style={{ marginTop: 6 }}>
                    {inputLabel}
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
