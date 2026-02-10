import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getAnalysisById } from "../lib/storage";
import { singleAnalysisTable, downloadTextFile } from "../lib/csv";

export default function AnalysisDetail() {
  const { id } = useParams();
  const item = useMemo(() => getAnalysisById(id), [id]);
  const [showConfidenceInfo, setShowConfidenceInfo] = useState(false);
  const confidenceRef = useRef(null);

  useEffect(() => {
    if (!showConfidenceInfo) return undefined;
    const onAnyClick = (e) => {
      if (confidenceRef.current && confidenceRef.current.contains(e.target)) {
        return;
      }
      setShowConfidenceInfo(false);
    };
    document.addEventListener("mousedown", onAnyClick, true);
    document.addEventListener("contextmenu", onAnyClick, true);
    return () => {
      document.removeEventListener("mousedown", onAnyClick, true);
      document.removeEventListener("contextmenu", onAnyClick, true);
    };
  }, [showConfidenceInfo]);

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

  const inputLabel =
    item.inputType === "form"
      ? "Formular"
      : item.inputType === "file"
      ? "Datei"
      : "Text";
  const displayText = item.fullText ?? item.preview ?? null;
  const rangeLabel = (() => {
    const k = String(item.klasse || "").toUpperCase();
    if (k === "LOW") return "< 10.000 €";
    if (k === "MID") return "10.000–20.000 €";
    if (k === "HIGH") return "> 20.000 €";
    if (k === "KEIN ANSPRUCH") return "Kein Anspruch";
    return "-";
  })();

  const explanation = (() => {
    const mode = item?.meta?.mode;
    if (mode === "rule") {
      const sig = item?.meta?.rule_signals || {};
      const parsed = item?.meta?.parsed || {};
      const factors = [];
      if (sig.kba) factors.push("KBA-Rückruf erwähnt");
      if (sig.abschalt) factors.push("Abschalteinrichtung genannt");
      if (sig.rechts) factors.push("§ 826/§ 31 BGB bzw. sittenwidrig erwähnt");
      if (sig.hersteller) factors.push("Herstellerverantwortung genannt");
      if (sig.claim) factors.push("Rückabwicklung/Schadensersatz gefordert");

      let rangeReason = null;
      if (item?.klasse && item.klasse !== "Kein Anspruch") {
        if (parsed?.kaufpreis) {
          rangeReason = `Schadenshöhe aus Kaufpreis (~${Math.round(parsed.kaufpreis)} EUR) und Nutzungsquote geschätzt.`;
        } else {
          rangeReason = "Schadenshöhe über die genannten Angaben geschätzt.";
        }
      }

      const lines = [];
      if (factors.length > 0) {
        lines.push(`Anspruch aus folgenden Faktoren abgeleitet: ${factors.join(", ")}.`);
      } else {
        lines.push("Anspruch aus den erkannten Angaben abgeleitet.");
      }
      if (rangeReason) lines.push(rangeReason);
      return lines;
    }

    return [
      "Die Bewertung basiert auf einer Gewichtung aus Textmerkmalen und strukturierten Angaben.",
      "Die Begründung ist indikativ und stellt keine juristische Würdigung dar.",
    ];
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
                const table = singleAnalysisTable(item);
                downloadTextFile(
                  `analyse_${item.id}.xls`,
                  table,
                  "application/vnd.ms-excel"
                );
              }}
            >
              Tabelle herunterladen
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
            <div className="k">
              Confidence
              <span ref={confidenceRef} style={{ position: "relative" }}>
                <button
                  type="button"
                  className="info-icon"
                  aria-label="Definition von Confidence"
                  aria-expanded={showConfidenceInfo}
                  onClick={() => setShowConfidenceInfo((v) => !v)}
                >
                  i
                </button>
                {showConfidenceInfo && (
                  <div className="info-popover">
                    Confidence bezeichnet die modellinterne Sicherheit der
                    Vorhersage. Sie gibt an, wie eindeutig das KI-Modell diesen
                    Fall einer Klasse (z. B. LOW/MID/HIGH) zuordnet, basierend auf
                    Mustern aus historischen Urteilen. Sie stellt keine rechtliche
                    Erfolgswahrscheinlichkeit dar.
                  </div>
                )}
              </span>
            </div>
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

        {explanation && explanation.length > 0 && (
          <>
            <div className="spacer" />
            <div
              style={{
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.04)",
                borderRadius: 14,
                padding: "12px 16px",
                maxWidth: 720,
                margin: "0 auto",
              }}
            >
              <div
                style={{
                  fontWeight: 600,
                  marginBottom: 6,
                  color: "rgba(255,255,255,0.92)",
                }}
              >
                Begründung
              </div>
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {explanation.map((line) => (
                  <li key={line} className="muted" style={{ marginBottom: 4 }}>
                    {line}
                  </li>
                ))}
              </ul>
            </div>
          </>
        )}

        {displayText && (item.inputType === "text" || item.inputType === "form" || item.inputType === "file") && (
          <>
            <div className="spacer" />
            <div className="kv">
              <div className="k">
                {item.inputType === "form"
                  ? "Formular-Notiz"
                  : item.inputType === "file"
                  ? "Datei"
                  : "Fallbeschreibung"}
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
