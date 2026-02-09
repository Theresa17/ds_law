"""
Inference pipeline for claim + range classification.

Mirrors the feature engineering steps from analyse_final.ipynb.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re

import numpy as np
import pandas as pd
import joblib
from gensim.models import Word2Vec

try:
    import spacy
except Exception:  # pragma: no cover - optional at import time
    spacy = None


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
FEATURES_CSV = Path(__file__).resolve().parents[1] / "jupyter_notebook" / "gemini_results_paid_complete.csv"


def split_judgment(text: str) -> dict[str, str]:
    """
    Teilt ein Urteil in Rubrum, Tenor, Tatbestand und Entscheidungsgründe auf.
    """
    segments = {
        "rubrum": "",
        "tenor": "",
        "tatbestand": "",
        "entscheidungsgruende": "",
    }

    m_tenor = re.search(r"\bTenor\b", text, re.IGNORECASE)
    m_tatbestand = re.search(r"\bTatbestand\b", text, re.IGNORECASE)
    m_gruende = re.search(
        r"\b(Entscheidungsgründe|Entscheidungsgruende|Gründe|Gruende)\b",
        text,
        re.IGNORECASE,
    )

    if m_tenor:
        segments["rubrum"] = text[: m_tenor.start()].strip()

        if m_tatbestand:
            segments["tenor"] = text[m_tenor.end() : m_tatbestand.start()].strip()

            if m_gruende:
                segments["tatbestand"] = text[
                    m_tatbestand.end() : m_gruende.start()
                ].strip()
                segments["entscheidungsgruende"] = text[m_gruende.end() :].strip()
            else:
                segments["tatbestand"] = text[m_tatbestand.end() :].strip()
        else:
            if m_gruende:
                segments["tenor"] = text[m_tenor.end() : m_gruende.start()].strip()
                segments["entscheidungsgruende"] = text[m_gruende.end() :].strip()
            else:
                segments["tenor"] = text[m_tenor.end() :].strip()

    return segments


# --- Preprocessing (from notebook) ---

LUECKENFUELLER = {
    "wobei",
    "somit",
    "jeweils",
    "hinsichtlich",
    "diverser",
    "nebst",
    "sowie",
    "zudem",
    "ferner",
    "daraufhin",
    "folgend",
    "folgende",
    "folge",
    "rahmen",
    "betreffen",
    "betroffen",
    "betreffend",
    "hierzu",
    "hierfür",
    "hiervon",
    "hierdurch",
    "insbesondere",
    "dafür",
    "wonach",
    "seinerzeit",
    "damalig",
    "ursprünglich",
    "inzwischen",
    "nunmehr",
    "lediglich",
    "einfach",
    "jedenfalls",
    "sodass",
    "darstellen",
    "geben",
    "ausreichend",
    "möglichst",
    "maximal",
    "bisherig",
    "zusammenfassend",
    "identisch",
    "entsprechend",
    "nämlich",
    "somit",
    "jeder",
    "allgemein",
    "gleichzeitig",
    "geeignet",
    "verbindlich",
    "unbekannt",
    "bzw",
    "ca",
    "u.a.",
    "iii",
    "ivm",
    "samt",
    "nebst",
    "ehemals",
    "jeweils",
    "anderer",
    "etwaig",
    "möglich",
    "soweit",
    "zeigen",
    "künftig",
    "identisch",
    "dar",
    "kommen",
    "weise",
    "bringen",
    "letzter",
    "erstens",
    "exakt",
    "anfang",
    "linie",
}

GENDER_FIX = {
    "klägerin": "kläger",
    "beklagte": "beklagter",
    "händlerin": "händler",
    "käuferin": "käufer",
    "verkäuferin": "verkäufer",
    "herstellerin": "hersteller",
    "leasinggeberin": "leasinggeber",
    "fahrzeughalterin": "fahrzeughalter",
    "anwältin": "anwalt",
    "prozessbevollmächtigte": "prozessbevollmächtigter",
}

MONATE = {
    "januar",
    "februar",
    "märz",
    "april",
    "mai",
    "juni",
    "juli",
    "august",
    "september",
    "oktober",
    "november",
    "dezember",
}


def _load_spacy() -> Any:
    if spacy is None:
        raise RuntimeError(
            "spaCy ist nicht installiert. Bitte `spacy` + Modell `de_core_news_lg` installieren."
        )
    return spacy.load("de_core_news_lg", disable=["ner", "parser"])


def legal_preprocess(text: str, nlp) -> str:
    if not isinstance(text, str) or not text or len(text) < 5:
        return ""
    text = text.lower()

    text = re.sub(r"\d{1,2}\.\d{1,2}\.(?:\d{2,4})?", " ", text)
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)

    doc = nlp(text)
    tokens: list[str] = []
    for t in doc:
        lemma = t.lemma_.lower()
        final = GENDER_FIX.get(lemma, lemma)

        if final in {"§", "abs", "sittenwidrig", "täuschung", "manipulation", "abschalteinrichtung", "thermofenster"}:
            tokens.append(final)
            continue

        if (
            not t.is_stop
            and final not in LUECKENFUELLER
            and final not in MONATE
            and not t.is_punct
            and not t.is_space
            and len(final) > 2
            and t.is_alpha
        ):
            tokens.append(final)

    return " ".join(tokens)


def get_weighted_doc_vector(doc: str, w2v_model, weights: dict[str, float]) -> np.ndarray:
    tokens = str(doc).split()
    valid_tokens = [t for t in tokens if t in w2v_model.wv and t in weights]

    if not valid_tokens:
        return np.zeros(w2v_model.vector_size)

    vectors = [w2v_model.wv[t] for t in valid_tokens]
    token_weights = [weights[t] for t in valid_tokens]

    return np.average(vectors, axis=0, weights=token_weights)


def clean_money_robust(val) -> float:
    if pd.isna(val) or str(val).strip().lower() in ["null", "none", "nan", ""]:
        return 0.0

    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    s = re.sub(r"[^\d.,-]", "", s)

    if "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        if "." in s:
            parts = s.split(".")
            if len(parts) > 1 and len(parts[-1]) == 3:
                s = s.replace(".", "")

    try:
        return float(s)
    except ValueError:
        return 0.0


def _normalize_feature_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        if v == "" or v.lower() in {"null", "none", "nan"}:
            return None
        return v
    return value


def normalize_features(features: dict | None) -> dict:
    if not features:
        return {}
    out: dict[str, Any] = {}
    for k, v in features.items():
        nv = _normalize_feature_value(v)
        if nv is None:
            continue
        out[k] = nv
    return out


def map_bool_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.lower()
        .str.strip()
        .map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "ja": 1,
                "nein": 0,
                "none": np.nan,
                "null": np.nan,
                "nan": np.nan,
            }
        )
    )


def prepare_features(df: pd.DataFrame, text_vectors: np.ndarray) -> pd.DataFrame:
    emb_cols = [f"emb_{i}" for i in range(text_vectors.shape[1])]
    df_emb = pd.DataFrame(text_vectors, columns=emb_cols, index=df.index)

    cat_cols = [
        "Dieselmotor_Typ",
        "Art_Abschalteinrichtung",
        "Fahrzeugstatus",
        "Fahrzeugmodell_Baureihe",
        "Beklagten_Typ",
        "Nacherfuellungsverlangen_Fristsetzung",
        "Klageziel",
        "Rechtsgrundlage",
    ]
    bool_cols = ["KBA_Rueckruf", "Update_Status"]
    num_cols = [
        "Kaufpreis_num",
        "Kilometerstand_Kauf",
        "Kilometerstand_Klageerhebung",
        "Erwartete_Gesamtlaufleistung",
    ]
    date_cols = ["Kaufdatum", "Uebergabedatum", "Datum_Klageerhebung", "Datum_Urteil"]

    available_cat = [c for c in cat_cols if c in df.columns]
    available_bool = [c for c in bool_cols if c in df.columns]
    available_num = [c for c in num_cols if c in df.columns]
    available_date = [c for c in date_cols if c in df.columns]

    df_struct = df.copy()

    for c in available_bool:
        df_struct[c] = map_bool_series(df_struct[c])

    for c in available_date:
        df_struct[c] = pd.to_datetime(df_struct[c], errors="coerce")

    if "Datum_Klageerhebung" in df_struct.columns and "Kaufdatum" in df_struct.columns:
        df_struct["tage_bis_klage"] = (
            df_struct["Datum_Klageerhebung"] - df_struct["Kaufdatum"]
        ).dt.days

    if "Datum_Urteil" in df_struct.columns and "Datum_Klageerhebung" in df_struct.columns:
        df_struct["tage_bis_urteil"] = (
            df_struct["Datum_Urteil"] - df_struct["Datum_Klageerhebung"]
        ).dt.days

    if "Datum_Klageerhebung" in df_struct.columns and "Uebergabedatum" in df_struct.columns:
        df_struct["tage_besitz_bis_klage"] = (
            df_struct["Datum_Klageerhebung"] - df_struct["Uebergabedatum"]
        ).dt.days

    delta_cols = [
        c
        for c in ["tage_bis_klage", "tage_bis_urteil", "tage_besitz_bis_klage"]
        if c in df_struct.columns
    ]

    df_cat = pd.get_dummies(df_struct[available_cat], columns=available_cat, dummy_na=True)
    df_num = df_struct[available_num + available_bool + delta_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    df_num = df_num.fillna(0)

    frames = []
    if not df_num.empty:
        frames.append(df_num)
    if not df_cat.empty:
        frames.append(df_cat)

    if frames:
        X_struct = pd.concat(frames, axis=1).fillna(0).astype(float)
    else:
        X_struct = pd.DataFrame(index=df.index)

    parts = []
    if not X_struct.empty or len(X_struct.columns) > 0:
        parts.append(X_struct)
    if not df_emb.empty or len(df_emb.columns) > 0:
        parts.append(df_emb)

    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def add_domain_ratios(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    def series_or_zero(col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0)
        return pd.Series([0.0] * len(df), index=df.index)

    kaufpreis = series_or_zero("Kaufpreis_num")
    km_kauf = series_or_zero("Kilometerstand_Kauf")
    km_klage = series_or_zero("Kilometerstand_Klageerhebung")
    gesamt_km = series_or_zero("Erwartete_Gesamtlaufleistung")

    def safe_ratio(num, denom, fill=0.0):
        return pd.Series(np.where(denom > 0, num / denom, fill), index=df.index).fillna(
            fill
        )

    km_gefahren = (km_klage - km_kauf).clip(lower=0)
    nutzungsquote = safe_ratio(km_klage, gesamt_km)
    X["nutzungsquote"] = nutzungsquote.values
    X["schadensproxy"] = (kaufpreis * (1 - nutzungsquote)).clip(lower=0).values
    X["km_gefahren"] = km_gefahren.values
    X["anteil_gefahren"] = safe_ratio(km_gefahren, gesamt_km).values
    X["preis_pro_km"] = safe_ratio(kaufpreis, gesamt_km).values
    X["wertminderung_pro_km"] = (safe_ratio(kaufpreis, gesamt_km) * km_gefahren).values
    X["log_kaufpreis"] = np.log1p(kaufpreis).values
    return X


RANGE_MIDPOINTS = {
    "LOW": 7500.0,
    "MID": 15000.0,
    "HIGH": 27500.0,
}


def amount_from_range(label: str | None) -> float:
    if not label:
        return 0.0
    key = str(label).strip().upper()
    return float(RANGE_MIDPOINTS.get(key, 0.0))


@dataclass
class ModelBundle:
    model: Any
    w2v: Any
    word_weights: dict[str, float]
    feature_columns: list[str]


class PredictionPipeline:
    def __init__(self) -> None:
        self._nlp = None
        self._claim: ModelBundle | None = None
        self._range: ModelBundle | None = None
        self._feature_repo: pd.DataFrame | None = None

    @staticmethod
    def _zero_features(bundle: ModelBundle) -> pd.DataFrame:
        if not bundle.feature_columns:
            return pd.DataFrame([{}])
        return pd.DataFrame(
            [[0.0] * len(bundle.feature_columns)], columns=bundle.feature_columns
        )

    def _ensure_nlp(self):
        if self._nlp is None:
            self._nlp = _load_spacy()

    def _load_bundle(self, prefix: str) -> ModelBundle:
        model_path = ARTIFACT_DIR / f"{prefix}_model.joblib"
        w2v_path = ARTIFACT_DIR / f"w2v_{prefix}.model"
        weights_path = ARTIFACT_DIR / f"word_weights_{prefix}.json"
        cols_path = ARTIFACT_DIR / f"{prefix}_feature_columns.json"

        missing = [
            str(p)
            for p in [model_path, w2v_path, weights_path, cols_path]
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Fehlende Modell-Artefakte: " + ", ".join(missing)
            )

        model = joblib.load(model_path)
        w2v = Word2Vec.load(str(w2v_path))
        word_weights = json.loads(weights_path.read_text(encoding="utf-8"))
        feature_columns = json.loads(cols_path.read_text(encoding="utf-8"))

        return ModelBundle(
            model=model, w2v=w2v, word_weights=word_weights, feature_columns=feature_columns
        )

    def _ensure_models(self):
        if self._claim is None:
            self._claim = self._load_bundle("claim")
        if self._range is None:
            self._range = self._load_bundle("range")

    def _load_feature_repo(self) -> pd.DataFrame:
        if self._feature_repo is None:
            if not FEATURES_CSV.exists():
                raise FileNotFoundError(
                    f"Feature-Repo nicht gefunden: {FEATURES_CSV}"
                )
            df = pd.read_csv(FEATURES_CSV)
            df["case_id"] = df["case_id"].astype(str)
            self._feature_repo = df
        return self._feature_repo

    def _get_case_row(self, case_id: str) -> pd.DataFrame:
        df = self._load_feature_repo()
        case_id = str(case_id).strip()
        row = df[df["case_id"] == case_id]
        if row.empty:
            raise ValueError(f"case_id nicht gefunden: {case_id}")

        row = row.copy().iloc[:1]
        if "Kaufpreis_num" not in row.columns and "Kaufpreis" in row.columns:
            row["Kaufpreis_num"] = row["Kaufpreis"].apply(clean_money_robust)
        return row

    def _build_features(self, text: str, bundle: ModelBundle) -> pd.DataFrame:
        self._ensure_nlp()

        segments = split_judgment(text or "")
        tatbestand = segments.get("tatbestand") or text or ""

        cleaned = legal_preprocess(tatbestand, self._nlp)
        text_vec = np.array([get_weighted_doc_vector(cleaned, bundle.w2v, bundle.word_weights)])

        # Minimal DF: only text features; structured features missing => filled with 0 later
        df = pd.DataFrame(
            [
                {
                    "tatbestand": tatbestand,
                    "cleaned_text": cleaned,
                }
            ]
        )

        try:
            X = prepare_features(df, text_vec)
            X = X.reindex(columns=bundle.feature_columns, fill_value=0)
        except ValueError:
            X = self._zero_features(bundle)
        return X

    def _build_range_features(self, text: str, bundle: ModelBundle) -> pd.DataFrame:
        X = self._build_features(text, bundle)
        # Range model expects domain ratios as in notebook
        df = pd.DataFrame(
            [
                {
                    "Kaufpreis_num": np.nan,
                    "Kilometerstand_Kauf": np.nan,
                    "Kilometerstand_Klageerhebung": np.nan,
                    "Erwartete_Gesamtlaufleistung": np.nan,
                }
            ]
        )
        X = add_domain_ratios(X, df)
        X = X.reindex(columns=bundle.feature_columns, fill_value=0)
        return X

    def _build_features_from_payload(
        self, text: str | None, features: dict | None, bundle: ModelBundle
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        payload = normalize_features(features)
        tatbestand = payload.get("tatbestand") or (text or "").strip()

        if tatbestand:
            self._ensure_nlp()
            cleaned = legal_preprocess(tatbestand, self._nlp)
            text_vec = np.array(
                [get_weighted_doc_vector(cleaned, bundle.w2v, bundle.word_weights)]
            )
        else:
            cleaned = ""
            text_vec = np.zeros((1, bundle.w2v.vector_size), dtype=float)

        row = dict(payload)
        if tatbestand:
            row.setdefault("tatbestand", tatbestand)
        if cleaned:
            row.setdefault("cleaned_text", cleaned)

        if "Kaufpreis_num" in row:
            row["Kaufpreis_num"] = clean_money_robust(row["Kaufpreis_num"])
        elif "Kaufpreis" in row:
            row["Kaufpreis_num"] = clean_money_robust(row["Kaufpreis"])

        df = pd.DataFrame([row])
        try:
            X = prepare_features(df, text_vec)
            X = X.reindex(columns=bundle.feature_columns, fill_value=0)
        except ValueError:
            X = self._zero_features(bundle)
        return X, df

    def _build_range_features_from_payload(
        self, text: str | None, features: dict | None, bundle: ModelBundle
    ) -> pd.DataFrame:
        X, df = self._build_features_from_payload(text, features, bundle)
        X = add_domain_ratios(X, df)
        X = X.reindex(columns=bundle.feature_columns, fill_value=0)
        return X

    def _build_features_from_case(self, case_id: str, bundle: ModelBundle) -> pd.DataFrame:
        row = self._get_case_row(case_id)
        vector_size = bundle.w2v.vector_size
        text_vec = np.zeros((1, vector_size), dtype=float)

        try:
            X = prepare_features(row, text_vec)
            X = X.reindex(columns=bundle.feature_columns, fill_value=0)
        except ValueError:
            X = self._zero_features(bundle)
        return X

    def _build_range_features_from_case(self, case_id: str, bundle: ModelBundle) -> pd.DataFrame:
        X = self._build_features_from_case(case_id, bundle)
        row = self._get_case_row(case_id)
        X = add_domain_ratios(X, row)
        X = X.reindex(columns=bundle.feature_columns, fill_value=0)
        return X

    def predict(self, text: str) -> dict:
        return self.predict_from_payload(text, None)

    def predict_from_case_id(self, case_id: str) -> dict:
        case_id = str(case_id).strip()
        if not case_id:
            raise ValueError("case_id fehlt.")

        self._ensure_models()

        assert self._claim is not None
        assert self._range is not None

        X_claim = self._build_features_from_case(case_id, self._claim)
        claim_proba = float(self._claim.model.predict_proba(X_claim)[0][1])
        claim_label = 1 if claim_proba >= 0.5 else 0

        range_label = None
        range_conf = None
        if claim_label == 1:
            X_range = self._build_range_features_from_case(case_id, self._range)
            probs = self._range.model.predict_proba(X_range)[0]
            range_idx = int(np.argmax(probs))
            range_label = str(self._range.model.classes_[range_idx])
            range_conf = float(probs[range_idx])

        betrag_eur = amount_from_range(range_label) if claim_label == 1 else 0.0

        return {
            "klasse": range_label or "Kein Anspruch",
            "entscheidung": "ja" if claim_label == 1 else "nein",
            "betrag_eur": betrag_eur,
            "confidence": round(claim_proba, 2),
            "meta": {
                "mode": "model",
                "eingabe": "case_id",
                "case_id": case_id,
                "claim_proba": round(claim_proba, 4),
                "range_klasse": range_label,
                "range_confidence": round(range_conf, 4) if range_conf is not None else None,
            },
        }

    def predict_from_payload(self, text: str | None, features: dict | None) -> dict:
        text = (text or "").strip()
        features = normalize_features(features)

        if not text and not features:
            raise ValueError("Keine Eingabe vorhanden (Text oder Features fehlen).")

        if text and len(text) < 10 and not features:
            raise ValueError("Der Text ist zu kurz fuer eine Analyse.")

        self._ensure_models()

        assert self._claim is not None
        assert self._range is not None

        X_claim, _ = self._build_features_from_payload(text, features, self._claim)
        claim_proba = float(self._claim.model.predict_proba(X_claim)[0][1])
        claim_label = 1 if claim_proba >= 0.5 else 0

        range_label = None
        range_conf = None
        if claim_label == 1:
            X_range = self._build_range_features_from_payload(
                text, features, self._range
            )
            probs = self._range.model.predict_proba(X_range)[0]
            range_idx = int(np.argmax(probs))
            range_label = str(self._range.model.classes_[range_idx])
            range_conf = float(probs[range_idx])

        betrag_eur = amount_from_range(range_label) if claim_label == 1 else 0.0

        return {
            "klasse": range_label or "Kein Anspruch",
            "entscheidung": "ja" if claim_label == 1 else "nein",
            "betrag_eur": betrag_eur,
            "confidence": round(claim_proba, 2),
            "meta": {
                "mode": "model",
                "eingabe": "payload",
                "features_keys": sorted(features.keys()),
                "claim_proba": round(claim_proba, 4),
                "range_klasse": range_label,
                "range_confidence": round(range_conf, 4) if range_conf is not None else None,
            },
        }
