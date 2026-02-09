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

  const inputLabel = item.inputType === "form" ? "Formular" : "Text";
  const displayText = item.fullText ?? item.preview ?? null;
  const rangeLabel = (() => {
    const k = String(item.klasse || "").toUpperCase();
    if (k === "LOW") return "< 10.000 €";
    if (k === "MID") return "10.000–20.000 €";
    if (k === "HIGH") return "> 20.000 €";
    if (k === "KEIN ANSPRUCH") return "Kein Anspruch";
    return "-";
  })();

  const formEntries = useMemo(() => {
    const data = item.formData || {};
    const fields = [
      "Dieselmotor_Typ",
      "Art_Abschalteinrichtung",
      "KBA_Rueckruf",
      "Update_Status",
      "Fahrzeugstatus",
      "Fahrzeugmodell_Baureihe",
      "Kilometerstand_Kauf",
      "Kilometerstand_Klageerhebung",
      "Erwartete_Gesamtlaufleistung",
      "Kaufdatum",
      "Uebergabedatum",
      "Datum_Klageerhebung",
      "Beklagten_Typ",
      "Kaufpreis",
      "Nacherfuellungsverlangen_Fristsetzung",
      "Klageziel",
      "Rechtsgrundlage",
    ];

    function formatValue(v) {
      if (v === null || v === undefined || v === "") return "-";
      if (typeof v === "boolean") return v ? "Ja" : "Nein";
      if (typeof v === "string") {
        const s = v.trim();
        if (s === "") return "-";
        if (s.toLowerCase() === "true") return "Ja";
        if (s.toLowerCase() === "false") return "Nein";
        return s;
      }
      return String(v);
    }

    return fields.map((k) => [k, formatValue(data[k])]);
  }, [item]);

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

        <div
          className="pill"
          style={{
            borderColor: "rgba(242, 118, 46, 0.45)",
            background: "rgba(242, 118, 46, 0.12)",
            color: "rgba(255,255,255,0.9)",
            marginBottom: 12,
          }}
        >
          Hinweis: Neue Urteile werden automatisch klassifiziert (Anspruch ja/nein
          und Schadenshöhe als Range LOW/MID/HIGH).
        </div>

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
            <div className="v">{rangeLabel}</div>
          </div>
          <div className="kv">
            <div className="k">Confidence</div>
            <div className="v">{Math.round(item.confidence * 100)}%</div>
          </div>
          <div className="kv">
            <div className="k">Eingabe</div>
            <div className="v">{inputLabel}</div>
          </div>
          <div className="kv">
            <div className="k">Zeitpunkt</div>
            <div className="v">
              {new Date(item.createdAt).toLocaleString("de-DE")}
            </div>
          </div>
        </div>

        {displayText && (item.inputType === "text" || item.inputType === "form") && (
          <>
            <div className="spacer" />
            <div className="kv">
              <div className="k">
                {item.inputType === "form" ? "Formular-Notiz" : "Fallbeschreibung"}
              </div>
              <div className="v mono">{displayText}</div>
            </div>
          </>
        )}

        {item.inputType === "form" && (
          <>
            <div className="spacer" />
            <h2 className="h2">Formular-Daten</h2>
            <div className="form-values">
              {formEntries.map(([k, v]) => (
                <div className="kv" key={k}>
                  <div className="k">{k}</div>
                  <div className="v mono">{v}</div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
