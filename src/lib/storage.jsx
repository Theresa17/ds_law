const KEY = "ds_law_analyses_v1";

export function loadAnalyses() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveAnalyses(items) {
  localStorage.setItem(KEY, JSON.stringify(items));
}

export function addAnalysis(item) {
  const items = loadAnalyses();
  items.unshift(item); // neueste zuerst
  saveAnalyses(items);
  return items;
}

export function getAnalysisById(id) {
  const items = loadAnalyses();
  return items.find((x) => x.id === id) || null;
}
