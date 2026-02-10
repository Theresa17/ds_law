import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { addAnalysis } from "../lib/storage";

const API_BASE = "http://127.0.0.1:5000";


export default function Home() {
  const nav = useNavigate();

  const [mode, setMode] = useState("text"); // "text" | "form"
  const [text, setText] = useState(""); // Text-Eingabe
  const [formText, setFormText] = useState(""); // Freitext im Formular
  const [openSelect, setOpenSelect] = useState(null);
  const [showDetailInfo, setShowDetailInfo] = useState(false);
  const detailInfoRef = useRef(null);
  const [file, setFile] = useState(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [form, setForm] = useState({
    Dieselmotor_Typ: "",
    Art_Abschalteinrichtung: "",
    KBA_Rueckruf: "",
    Update_Status: "",
    Fahrzeugstatus: "",
    Fahrzeugmodell_Baureihe: "",
    Kilometerstand_Kauf: "",
    Kilometerstand_Klageerhebung: "",
    Erwartete_Gesamtlaufleistung: "",
    Kaufdatum: "",
    Uebergabedatum: "",
    Datum_Klageerhebung: "",
    Beklagten_Typ: "",
    Kaufpreis: "",
    Nacherfuellungsverlangen_Fristsetzung: "",
    Klageziel: "",
    Rechtsgrundlage: "",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!openSelect) return undefined;
    const close = () => setOpenSelect(null);
    const onKey = (e) => {
      if (e.key === "Escape") setOpenSelect(null);
    };
    document.addEventListener("click", close);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("click", close);
      document.removeEventListener("keydown", onKey);
    };
  }, [openSelect]);

  useEffect(() => {
    if (!showDetailInfo) return undefined;
    const onAnyClick = (e) => {
      if (detailInfoRef.current && detailInfoRef.current.contains(e.target)) {
        return;
      }
      setShowDetailInfo(false);
    };
    document.addEventListener("mousedown", onAnyClick, true);
    document.addEventListener("contextmenu", onAnyClick, true);
    return () => {
      document.removeEventListener("mousedown", onAnyClick, true);
      document.removeEventListener("contextmenu", onAnyClick, true);
    };
  }, [showDetailInfo]);

  const canAnalyze = useMemo(() => {
    if (mode === "text") {
      if (file) return true;
      return text.trim().length >= 30;
    }
    const hasText = formText.trim().length > 0;
    return hasText;
  }, [mode, text, form, formText, file]);

  const textSignals = useMemo(() => {
    const t = text.toLowerCase();
    const hasRegex = (re) => re.test(t);
    const sentences = t.split(/(?<=[\.\!\?])\s+/);
    const sentenceHas = (a, b) =>
      sentences.some((s) => s.includes(a) && s.includes(b));
    const sentenceHasRegex = (ra, rb) =>
      sentences.some((s) => ra.test(s) && rb.test(s));
    return {
      kaufpreis: t.includes("kaufpreis"),
      neuwagen: t.includes("neuwagen"),
      gebrauchtwagen: t.includes("gebrauchtwagen"),
      km_kauf:
        sentenceHas("kilometerstand", "kauf") ||
        sentenceHas("kilometer", "kauf") ||
        sentenceHasRegex(/\bkm\b/, /\bkauf\b/) ||
        hasRegex(/\b(kilometerstand|kilometer|km)\b[\s\S]{0,80}\bkauf\b/) ||
        hasRegex(/\bkauf\b[\s\S]{0,80}\b(kilometerstand|kilometer|km)\b/),
      km_klage:
        sentenceHas("kilometerstand", "klage") ||
        sentenceHas("kilometer", "klage") ||
        sentenceHasRegex(/\bkm\b/, /\bklage\b/) ||
        hasRegex(/\b(kilometerstand|kilometer|km)\b[\s\S]{0,80}\bklage\b/) ||
        hasRegex(/\bklage\b[\s\S]{0,80}\b(kilometerstand|kilometer|km)\b/),
      kba: t.includes("kba") && (t.includes("rückruf") || t.includes("rueckruf")),
      beklagter:
        t.includes("beklagter") ||
        t.includes("hersteller") ||
        t.includes("händler") ||
        t.includes("haendler"),
      klageziel:
        t.includes("klageziel") ||
        t.includes("rückabwicklung") ||
        t.includes("rueckabwicklung") ||
        t.includes("schadensersatz"),
      fristsetzung:
        t.includes("fristsetzung") ||
        t.includes("nacherfüllung") ||
        t.includes("nacherfuellung") ||
        t.includes("entbehrlich"),
      abschalteinrichtung:
        t.includes("abschalt") || t.includes("thermofenster") || t.includes("umschaltlogik"),
      rechtsgrundlage:
        t.includes("§ 826") ||
        t.includes("826 bgb") ||
        t.includes("§ 31") ||
        t.includes("31 bgb") ||
        t.includes("sittenwidrig"),
      gesamtlaufleistung: t.includes("gesamtlaufleistung"),
      update:
        t.includes("update") ||
        t.includes("software-update") ||
        t.includes("software update"),
    };
  }, [text]);

  function updateForm(name, value) {
    setForm((prev) => ({ ...prev, [name]: value }));
  }

  function buildFeatures() {
    const out = {};
    for (const [k, v] of Object.entries(form)) {
      const val = typeof v === "string" ? v.trim() : v;
      if (val !== "" && val !== null && val !== undefined) out[k] = val;
    }
    return out;
  }

  function SelectField({
    name,
    label,
    value,
    options,
    placeholder = "Bitte auswählen",
    onChange,
  }) {
    const selected = options.find((o) => o.value === value) || null;
    const isOpen = openSelect === name;
    const display = selected ? selected.label : placeholder;

    return (
      <div className="field select-field" onClick={(e) => e.stopPropagation()}>
        <label className="label">{label}</label>
        <button
          type="button"
          className={`select-btn ${isOpen ? "open" : ""}`}
          onClick={() => setOpenSelect(isOpen ? null : name)}
        >
          <span className={`select-value ${selected ? "" : "muted"}`}>
            {display}
          </span>
          <span className="select-caret" />
        </button>
        {isOpen && (
          <div className="select-menu" role="listbox">
            <button
              type="button"
              className={`select-option ${!selected ? "active" : ""}`}
              onClick={() => {
                onChange("");
                setOpenSelect(null);
              }}
            >
              {placeholder}
            </button>
            {options.map((opt) => (
              <button
                type="button"
                key={opt.value}
                className={`select-option ${
                  selected?.value === opt.value ? "active" : ""
                }`}
                onClick={() => {
                  onChange(opt.value);
                  setOpenSelect(null);
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  const yesNoOptions = [
    { label: "Ja", value: "true" },
    { label: "Nein", value: "false" },
  ];

  async function onAnalyze() {
    if (loading) return;

    setError(null);

    try {
      let data;

      if (mode === "text") {
        setLoading(true);
        if (file) {
          const name = file.name || "";
          if (!name.toLowerCase().endsWith(".txt")) {
            throw new Error("Bitte nur .txt Dateien hochladen.");
          }
          const formData = new FormData();
          formData.append("file", file);
          const res = await fetch(`${API_BASE}/predict-file`, {
            method: "POST",
            body: formData,
          });
          const json = await res.json().catch(() => ({}));
          console.log("Antwort von Backend (FILE):", json);
          if (!res.ok)
            throw new Error(json.error || "Fehler bei der Analyse (Datei).");
          data = json;
        } else {
          if (text.trim().length < 30) {
            throw new Error("Bitte einen längeren Text eingeben (mind. 30 Zeichen).");
          }
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
        }
      } else {
        // mode === "form"
        const features = buildFeatures();
        const requiredKeys = [
          "Kaufpreis",
          "Kaufdatum",
          "Kilometerstand_Klageerhebung",
          "Fahrzeugstatus",
          "Beklagten_Typ",
        ];
        const missing = requiredKeys.filter(
          (k) => !String(form[k] ?? "").trim()
        );
        if (missing.length > 0 || formText.trim().length === 0) {
          throw new Error("Bitte Pflichtangaben eintragen.");
        }
        setLoading(true);
        const payload = {
          text: formText.trim(),
          features,
        };

        const res = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const json = await res.json().catch(() => ({}));
        console.log("Antwort von Backend (FORM):", json);
        if (!res.ok)
          throw new Error(json.error || "Fehler bei der Analyse (Formular).");
        data = json;
      }

      const id = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());

      const item = {
        id,
        createdAt: new Date().toISOString(),
        inputType: mode === "text" && file ? "file" : mode,
        fileName: mode === "text" && file ? file.name : null,
        preview:
          mode === "text" && file
            ? file.name
            : mode === "text"
            ? text.trim().slice(0, 120)
            : mode === "form"
            ? formText.trim().slice(0, 120) || null
            : null,
        fullText:
          mode === "text" && file
            ? null
            : mode === "text"
            ? text.trim()
            : mode === "form"
            ? formText.trim() || null
            : null,
        formData: mode === "form" ? buildFeatures() : null,
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
    setText("");
    setFormText("");
    setFile(null);
    setFileInputKey((k) => k + 1);
    setForm({
      Dieselmotor_Typ: "",
      Art_Abschalteinrichtung: "",
      KBA_Rueckruf: "",
      Update_Status: "",
      Fahrzeugstatus: "",
      Fahrzeugmodell_Baureihe: "",
      Kilometerstand_Kauf: "",
      Kilometerstand_Klageerhebung: "",
      Erwartete_Gesamtlaufleistung: "",
      Kaufdatum: "",
      Uebergabedatum: "",
      Datum_Klageerhebung: "",
      Beklagten_Typ: "",
      Kaufpreis: "",
      Nacherfuellungsverlangen_Fristsetzung: "",
      Klageziel: "",
      Rechtsgrundlage: "",
    });
    setError(null);
    setLoading(false);
  }


  return (
    <div className="card">
      <div className="card-inner">
        <h1 className="h1">Neue Analyse</h1>
       
        <div className="spacer" />

        <div className="segmented">
          <button
            className={`seg-btn ${mode === "text" ? "seg-active" : ""}`}
            onClick={() => setMode("text")}
            type="button"
          >
            Text
          </button>
          <button
            className={`seg-btn ${mode === "form" ? "seg-active" : ""}`}
            onClick={() => setMode("form")}
            type="button"
          >
            Formular
          </button>
        </div>

        <div className="spacer" />

        {mode === "text" ? (
          <div className="text-input">
            <label className="label">
              Fallbeschreibung
              <span className="info-inline" ref={detailInfoRef}>
                <button
                  type="button"
                  className="info-icon"
                  aria-label="Hinweis zur Detailtiefe"
                  aria-expanded={showDetailInfo}
                  onClick={() => setShowDetailInfo((v) => !v)}
                >
                  i
                </button>
                {showDetailInfo && (
                  <div className="info-popover">
                    Einflussfaktor Detailtiefe: Die Verlässlichkeit der Prognose
                    korreliert direkt mit der Substanz Ihrer Fallbeschreibung.
                    Unser Algorithmus scannt den Text nach juristischen und
                    technischen Signalwörtern (z. B. zu Motortyp,
                    Abschalteinrichtung oder Klageantrag), die in historischen
                    Urteilen maßgeblich für den Erfolg waren. Hinweis: Vage oder
                    unvollständige Sachverhalte führen statistisch zu einer
                    konservativeren (negativen) Einschätzung, da klare
                    Erfolgssignale fehlen.
                  </div>
                )}
              </span>
            </label>
            <div className="signal-row">
              <span className={textSignals.kaufpreis ? "signal on" : "signal"}>Kaufpreis</span>
              <span className={textSignals.neuwagen ? "signal on" : "signal"}>Neuwagen</span>
              <span className={textSignals.gebrauchtwagen ? "signal on" : "signal"}>Gebrauchtwagen</span>
              <span className={textSignals.km_kauf ? "signal on" : "signal"}>Kilometerstand Kauf</span>
              <span className={textSignals.km_klage ? "signal on" : "signal"}>Kilometerstand Klage</span>
              <span className={textSignals.kba ? "signal on" : "signal"}>KBA-Rückruf</span>
              <span className={textSignals.beklagter ? "signal on" : "signal"}>Beklagter</span>
              <span className={textSignals.klageziel ? "signal on" : "signal"}>Klageziel</span>
              <span className={textSignals.fristsetzung ? "signal on" : "signal"}>Fristsetzung</span>
              <span className={textSignals.abschalteinrichtung ? "signal on" : "signal"}>Abschalteinrichtung</span>
              <span className={textSignals.rechtsgrundlage ? "signal on" : "signal"}>Rechtsgrundlage</span>
              <span className={textSignals.gesamtlaufleistung ? "signal on" : "signal"}>Gesamtlaufleistung</span>
              <span className={textSignals.update ? "signal on" : "signal"}>Update</span>
            </div>
            <div className="field" style={{ marginBottom: 8 }}>
              <label className="label">TXT-Datei hochladen (optional)</label>
              <input
                key={fileInputKey}
                className="input"
                type="file"
                accept=".txt,.pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
                disabled={text.trim().length > 0}
                onChange={(e) => {
                  const picked = e.target.files?.[0] || null;
                  setFile(picked);
                  if (picked) setText("");
                }}
              />
              <div className="muted" style={{ marginTop: 6 }}>
                Wenn eine Datei gewählt ist, wird der Text aus der Datei analysiert.
              </div>
              {file && (
                <div className="row" style={{ marginTop: 6, gap: 8 }}>
                  <div className="muted">AusgewÃ¤hlt: {file.name}</div>
                  <button
                    className="btn"
                    type="button"
                    onClick={() => {
                      setFile(null);
                      setFileInputKey((k) => k + 1);
                    }}
                  >
                    Datei entfernen
                  </button>
                </div>
              )}
            </div>
            <textarea
              className="textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={!!file}
              placeholder="Text hier einfügen?"
            />
            <div className="muted" style={{ marginTop: 8 }}>
              Mindestlänge: 30 · Aktuell: {text.trim().length}
            </div>
          </div>
        ) : (
          <div className="form">
            <div className="spacer" />
            <div className="form-grid">
              <div className="field">
                <label className="label">Kaufpreis (EUR) *</label>
                <input
                  className="input"
                  type="text"
                  value={form.Kaufpreis}
                  onChange={(e) => updateForm("Kaufpreis", e.target.value)}
                  placeholder="z. B. 25900"
                />
              </div>

              <div className="field">
                <label className="label">Kaufdatum *</label>
                <input
                  className="input"
                  type="date"
                  value={form.Kaufdatum}
                  onChange={(e) => updateForm("Kaufdatum", e.target.value)}
                />
              </div>

              <div className="field">
                <label className="label">Kilometerstand Klage *</label>
                <input
                  className="input"
                  type="number"
                  value={form.Kilometerstand_Klageerhebung}
                  onChange={(e) =>
                    updateForm("Kilometerstand_Klageerhebung", e.target.value)
                  }
                  placeholder="z. B. 30000"
                />
              </div>

              <SelectField
                name="status"
                label="Fahrzeugstatus *"
                value={form.Fahrzeugstatus}
                placeholder="Bitte auswählen"
                onChange={(v) => updateForm("Fahrzeugstatus", v)}
                options={[
                  { label: "Neuwagen", value: "Neuwagen" },
                  { label: "Gebrauchtwagen", value: "Gebrauchtwagen" },
                ]}
              />

              <SelectField
                name="beklagte"
                label="Beklagten Typ *"
                value={form.Beklagten_Typ}
                placeholder="Bitte auswählen"
                onChange={(v) => updateForm("Beklagten_Typ", v)}
                options={[
                  { label: "Händler", value: "Händler" },
                  { label: "Hersteller", value: "Hersteller" },
                ]}
              />

              <div
                style={{
                  gridColumn: "1 / -1",
                  height: 1,
                  background: "rgba(255,255,255,0.08)",
                  margin: "12px 0 14px",
                }}
              />

              <SelectField
                name="diesel"
                label="Dieselmotor Typ"
                value={form.Dieselmotor_Typ}
                onChange={(v) => updateForm("Dieselmotor_Typ", v)}
                options={[
                  { label: "EA 189", value: "EA 189" },
                  { label: "EA 288", value: "EA 288" },
                  { label: "Sonstige", value: "Sonstige" },
                ]}
              />

              <SelectField
                name="abschalt"
                label="Abschalteinrichtung"
                value={form.Art_Abschalteinrichtung}
                onChange={(v) => updateForm("Art_Abschalteinrichtung", v)}
                options={[
                  { label: "Umschaltlogik", value: "Umschaltlogik" },
                  { label: "Thermofenster", value: "Thermofenster" },
                  { label: "Sonstige", value: "Sonstige" },
                ]}
              />

              <SelectField
                name="kba"
                label="KBA Rückruf"
                value={form.KBA_Rueckruf}
                onChange={(v) => updateForm("KBA_Rueckruf", v)}
                options={yesNoOptions}
              />

              <SelectField
                name="update"
                label="Update Status"
                value={form.Update_Status}
                onChange={(v) => updateForm("Update_Status", v)}
                options={yesNoOptions}
              />

              <div className="field">
                <label className="label">Fahrzeugmodell/Baureihe</label>
                <input
                  className="input"
                  value={form.Fahrzeugmodell_Baureihe}
                  onChange={(e) =>
                    updateForm("Fahrzeugmodell_Baureihe", e.target.value)
                  }
                  placeholder="z. B. VW Golf 2.0 TDI"
                />
              </div>

              <div className="field">
                <label className="label">Kilometerstand Kauf</label>
                <input
                  className="input"
                  type="number"
                  value={form.Kilometerstand_Kauf}
                  onChange={(e) =>
                    updateForm("Kilometerstand_Kauf", e.target.value)
                  }
                  placeholder="z. B. 12000"
                />
              </div>
              <div className="field">
                <label className="label">Erwartete Gesamtlaufleistung</label>
                <input
                  className="input"
                  type="number"
                  value={form.Erwartete_Gesamtlaufleistung}
                  onChange={(e) =>
                    updateForm("Erwartete_Gesamtlaufleistung", e.target.value)
                  }
                  placeholder="z. B. 200000"
                />
              </div>
              <div className="field">
                <label className="label">Übergabedatum</label>
                <input
                  className="input"
                  type="date"
                  value={form.Uebergabedatum}
                  onChange={(e) => updateForm("Uebergabedatum", e.target.value)}
                />
              </div>

              <div className="field">
                <label className="label">Datum Klageerhebung</label>
                <input
                  className="input"
                  type="date"
                  value={form.Datum_Klageerhebung}
                  onChange={(e) =>
                    updateForm("Datum_Klageerhebung", e.target.value)
                  }
                />
              </div>

              <SelectField
                name="nachfuell"
                label="Nacherfüllung / Fristsetzung"
                value={form.Nacherfuellungsverlangen_Fristsetzung}
                onChange={(v) =>
                  updateForm("Nacherfuellungsverlangen_Fristsetzung", v)
                }
                options={[
                  { label: "Ja", value: "Ja" },
                  { label: "Nein", value: "Nein" },
                  { label: "Entbehrlich", value: "Entbehrlich" },
                ]}
              />

            </div>

            <div className="spacer" />

            <div className="text-input">
              <label className="label">Freitext / Tatbestand</label>
              <textarea
                className="textarea"
                value={formText}
                onChange={(e) => setFormText(e.target.value)}
                placeholder="Zusätzliche Angaben hier einfügen?"
              />
              <div className="muted" style={{ marginTop: 8 }}>
                Optionaler Freitext · Aktuell: {formText.trim().length}
              </div>
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
            disabled={loading}
            type="button"
            aria-disabled={loading}
          >
            {loading ? "Analysiere…" : "Analysieren"}
          </button>
        </div>

        {error && (
          <>
            <div className="spacer" />
            <div
              style={{
                border: "1px solid rgba(242,104,104,0.5)",
                background: "rgba(242,104,104,0.12)",
                borderRadius: 14,
                padding: "10px 14px",
                maxWidth: 520,
                margin: "0 auto",
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontWeight: 600,
                  color: "rgba(255,255,255,0.95)",
                  marginBottom: 4,
                }}
              >
                Fehlende Pflichtangaben
              </div>
              <div className="muted">{error}</div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
