function escapeCsv(value) {
  const s = value == null ? "" : String(value);
  // CSV-sicher: doppelte Anführungszeichen verdoppeln
  const escaped = s.replace(/"/g, '""');
  // wenn Komma/Zeilenumbruch/Quote drin: in Quotes setzen
  return /[",\n\r;]/.test(escaped) ? `"${escaped}"` : escaped;
}

function rangeLabelFromClass(klasse) {
  const k = String(klasse || "").toUpperCase();
  if (k === "LOW") return "< 10.000 €";
  if (k === "MID") return "10.000–20.000 €";
  if (k === "HIGH") return "> 20.000 €";
  if (k === "KEIN ANSPRUCH") return "Kein Anspruch";
  return "";
}

export function analysisToCsvRow(a) {
  const cols = [
    a.id,
    a.createdAt,
    a.inputType,
    a.fileName ?? "",
    a.klasse ?? "",
    a.entscheidung ?? "",
    rangeLabelFromClass(a.klasse),
    a.confidence ?? "",
    a.fullText ?? a.preview ?? "",
  ];
  return cols.map(escapeCsv).join(";");
}

export function singleAnalysisCsv(a) {
  const header = [
    "id",
    "createdAt",
    "inputType",
    "fileName",
    "klasse",
    "entscheidung",
    "betrag_eur",
    "confidence",
    "text",
  ].join(";");
  return `${header}\n${analysisToCsvRow(a)}`;
}

export function downloadTextFile(filename, content, mime = "text/csv;charset=utf-8") {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();

  URL.revokeObjectURL(url);
}

export function allAnalysesCsv(items) {
  const header = [
    "id",
    "createdAt",
    "inputType",
    "fileName",
    "klasse",
    "entscheidung",
    "betrag_eur",
    "confidence",
    "text",
  ].join(";");

  const rows = items.map(analysisToCsvRow);
  return [header, ...rows].join("\n");
}

