import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { addAnalysis } from "../lib/storage";

const API_BASE = "http://127.0.0.1:5000";

export default function Home() {
  const nav = useNavigate();

  const [mode, setMode] = useState("form"); // "text" | "form"
  const [text, setText] = useState(""); // Text-Eingabe
  const [formText, setFormText] = useState(""); // Freitext im Formular
  const [openSelect, setOpenSelect] = useState(null);
  const [form, setForm] = useState({
    Dieselmotor_Typ: "",
    Art_Abschalteinrichtung: "",
    KBA_Rueckruf: "",
    Fahrzeugstatus: "",
    Fahrzeugmodell_Baureihe: "",
    Kilometerstand_Kauf: "",
    Kilometerstand_Klageerhebung: "",
    Erwartete_Gesamtlaufleistung: "",
    Kaufdatum: "",
    Uebergabedatum: "",
    Beklagten_Typ: "",
    Kaufpreis: "",
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

  const canAnalyze = useMemo(() => {
    if (mode === "text") return text.trim().length >= 30;
    return true;
  }, [mode, text, form, formText]);

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
        if (text.trim().length < 30) {
          throw new Error("Bitte einen längeren Text eingeben (mind. 30 Zeichen).");
        }
        setLoading(true);
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
        if (missing.length > 0) {
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
        inputType: mode,
        fileName: null,
        preview:
          mode === "text"
            ? text.trim().slice(0, 120)
            : mode === "form"
            ? formText.trim().slice(0, 120) || null
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
    setForm({
      Dieselmotor_Typ: "",
      Art_Abschalteinrichtung: "",
      KBA_Rueckruf: "",
      Fahrzeugstatus: "",
      Fahrzeugmodell_Baureihe: "",
      Kilometerstand_Kauf: "",
      Kilometerstand_Klageerhebung: "",
      Erwartete_Gesamtlaufleistung: "",
      Kaufdatum: "",
      Uebergabedatum: "",
      Beklagten_Typ: "",
      Kaufpreis: "",
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
            className={`seg-btn ${mode === "form" ? "seg-active" : ""}`}
            onClick={() => setMode("form")}
            type="button"
          >
            Formular
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

        {mode === "text" ? (
          <div className="text-input">
            <label className="label">Urteilstext</label>
            <textarea
              className="textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
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
                <label className="label">Rechtsgrundlage</label>
                <input
                  className="input"
                  value={form.Rechtsgrundlage}
                  onChange={(e) => updateForm("Rechtsgrundlage", e.target.value)}
                  placeholder="z. B. § 826 BGB"
                />
              </div>
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
