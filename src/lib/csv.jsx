function escapeCsv(value) {
  const s = value == null ? "" : String(value);
  // CSV-sicher: doppelte Anführungszeichen verdoppeln
  const escaped = s.replace(/"/g, '""');
  // wenn Komma/Zeilenumbruch/Quote drin: in Quotes setzen
  return /[",\n\r;]/.test(escaped) ? `"${escaped}"` : escaped;
}

export function analysisToCsvRow(a) {
  const cols = [
    a.id,
    a.createdAt,
    a.inputType,
    a.fileName ?? "",
    a.klasse ?? "",
    a.entscheidung ?? "",
    a.betrag_eur ?? "",
    a.confidence ?? "",
    a.preview ?? "",
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
    "preview",
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
    "preview",
  ].join(";");

  const rows = items.map(analysisToCsvRow);
  return [header, ...rows].join("\n");
}

