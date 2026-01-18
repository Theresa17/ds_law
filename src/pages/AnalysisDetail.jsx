import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { getAnalysisById } from "../lib/storage";
import { singleAnalysisCsv, downloadTextFile } from "../lib/csv";

export default function AnalysisDetail() {
  const { id } = useParams();
  const item = useMemo(() => getAnalysisById(id), [id]);

  if (!item) {
    return (
      <div className="card">
        <div className="card-inner">
          <h1 className="h1">Nicht gefunden</h1>
          <p className="p">
            Diese Analyse existiert nicht (mehr) im localStorage.
          </p>
          <div className="spacer" />
          <Link className="btn" to="/history">
            Zurück zum Verlauf
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-inner">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h1 className="h1" style={{ margin: 0 }}>
            Analyse-Details
          </h1>

          <div className="row">
            <button
              className="btn"
              type="button"
              onClick={() => {
                const csv = singleAnalysisCsv(item);
                downloadTextFile(`analyse_${item.id}.csv`, csv);
              }}
            >
              CSV herunterladen
            </button>

            <Link className="btn" to="/history">
              Verlauf
            </Link>
          </div>
        </div>

        <div className="spacer" />

        <div className="result-grid">
          <div className="kv">
            <div className="k">Klassifikation</div>
            <div className="v">{item.klasse}</div>
          </div>
          <div className="kv">
            <div className="k">Entscheidung</div>
            <div className="v">{item.entscheidung}</div>
          </div>
          <div className="kv">
            <div className="k">Betrag</div>
            <div className="v">{item.betrag_eur.toLocaleString("de-DE")} €</div>
          </div>
          <div className="kv">
            <div className="k">Confidence</div>
            <div className="v">{Math.round(item.confidence * 100)}%</div>
          </div>
          <div className="kv">
            <div className="k">Eingabe</div>
            <div className="v">
              {item.inputType === "file" ? item.fileName : "Text"}
            </div>
          </div>
          <div className="kv">
            <div className="k">Zeitpunkt</div>
            <div className="v">
              {new Date(item.createdAt).toLocaleString("de-DE")}
            </div>
          </div>
        </div>

        {item.inputType === "text" && item.preview && (
          <>
            <div className="spacer" />
            <div className="kv">
              <div className="k">Text-Ausschnitt</div>
              <div className="v mono">{item.preview}…</div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
