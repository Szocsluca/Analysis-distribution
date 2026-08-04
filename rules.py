"""
Exclusion rules for analysis distributions.
Each rule returns the set of CNPs to EXCLUDE for a given test when building the distribution.
Add new rules here as we add more analyses.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Path to CSV that maps (Diagnostic, Analysis) -> "DA" = exclude that result for that analysis
DIAGNOSTICE_CSV_PATH = Path(__file__).parent / "Diagnostice.csv"

# Test names used in rules (must match CSV column "Test")
GLICEMIE_TEST = "Glucoza serica (glicemie)"
HEMOGLOBINA_GLICATA_TEST = "Hemoglobina glicata (Hb A1c)"
CREATININA_TEST = "Creatinina serica"
# Markeri tumorali (MT folder): CA 125, CA 15.3, CA 19.9
MT_TESTS = {"CA 125", "CA 15.3", "CA 19.9"}
SODIU_TEST = "Sodiu in ser (Na)"
POTASIU_TEST = "Potasiu in ser (K)"
FOSFOR_TEST = "Fosfor in ser (P)"
AMILAZA_TEST = "Amilaza serica"
FIER_TEST = "Fier seric (sideremie)"

# Column names (must match prepared dataframe)
CNP = "CNP"
TEST = "Test"
REZULTAT = "Rezultat"
RATA_FILTRARII = "Rata filtrarii glomerolare"
DIAGNOSTIC = "Diagnostic"


def _normalize_diagnostic_value(value: str) -> str:
    """Normalize diagnostic labels so matching is stable with or without numeric prefixes."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"^\d+\s*-\s*", "", text).strip()


def _load_diagnostice_exclusion_map() -> dict[str, set[str]]:
    """
    Load Diagnostice.csv: first column = Diagnostic, other columns = analysis names.
    Returns dict[analysis_column_stripped, set(diagnostic strings with DA)].
    """
    if not DIAGNOSTICE_CSV_PATH.exists():
        return {}
    df = pd.read_csv(DIAGNOSTICE_CSV_PATH, encoding="utf-8")
    if df.empty or "Diagnostic" not in df.columns:
        return {}
    df["Diagnostic"] = df["Diagnostic"].map(_normalize_diagnostic_value)
    out = {}
    for col in df.columns:
        if col == "Diagnostic":
            continue
        col_stripped = col.strip()
        # Rows where this analysis column has "DA" (case-insensitive, stripped)
        mask = df[col].astype(str).str.strip().str.upper() == "DA"
        out[col_stripped] = set(df.loc[mask, "Diagnostic"].dropna().astype(str).str.strip())
    return out


def _diagnostics_to_exclude_for_test(selected_test: str) -> set[str]:
    """Return set of diagnostic strings to exclude for this analysis (from Diagnostice.csv)."""
    mapping = _load_diagnostice_exclusion_map()
    if not mapping:
        return set()
    # Match selected_test to a column: column name contained in selected_test or equal; use longest match
    selected = selected_test.strip()
    best = None
    for col_name in mapping:
        if col_name in selected or selected.startswith(col_name) or selected == col_name:
            if best is None or len(col_name) > len(best):
                best = col_name
    return mapping.get(best, set()) if best else set()


def _diagnostics_to_exclude_for_columns(column_names: list[str]) -> set[str]:
    """Return diagnostics marked DA for one or more explicit analysis columns from Diagnostice.csv."""
    mapping = _load_diagnostice_exclusion_map()
    if not mapping:
        return set()
    out: set[str] = set()
    for name in column_names:
        target = name.strip().lower()
        if not target:
            continue
        # Prefer exact column name match.
        exact = next((k for k in mapping.keys() if k.strip().lower() == target), None)
        if exact is not None:
            out |= mapping.get(exact, set())
            continue
        # Fallback: loose contains matching for slight naming differences.
        for col_name, values in mapping.items():
            c = col_name.strip().lower()
            if target in c or c in target:
                out |= values
    return out


def rule_glicemie_exclude_cnp_if_hba1c_gt_6(df: pd.DataFrame, selected_test: str) -> set:
    """
    For Glicemie: exclude a CNP if that CNP has any Hemoglobina glicata (Hb A1c) result > 6.
    """
    if selected_test != GLICEMIE_TEST:
        return set()
    if TEST not in df.columns or REZULTAT not in df.columns or CNP not in df.columns:
        return set()
    hb = df[(df[TEST] == HEMOGLOBINA_GLICATA_TEST) & (df[REZULTAT].notna())]
    if hb.empty:
        return set()
    rezultat_num = pd.to_numeric(hb[REZULTAT].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    cnps_with_hba1c_gt_6 = set(hb.loc[rezultat_num > 6, CNP].dropna().astype(str).str.strip())
    return cnps_with_hba1c_gt_6


# List of all exclusion rules: (test this rule applies to, human-readable name, function).
# Each function(df, selected_test) -> set of CNPs to exclude.
EXCLUSION_RULES = [
    (GLICEMIE_TEST, "Exclude CNP dacă Hemoglobina glicată (Hb A1c) > 6", rule_glicemie_exclude_cnp_if_hba1c_gt_6),
]


def apply_exclusion_rules(df: pd.DataFrame, selected_test: str) -> set:
    """
    Run all exclusion rules for the selected test. Returns the set of CNPs to exclude.
    """
    exclude_cnps = set()
    for rule_test, _name, rule_fn in EXCLUSION_RULES:
        if rule_test == selected_test:
            exclude_cnps |= rule_fn(df, selected_test)
    return exclude_cnps


def get_active_rule_names(selected_test: str) -> list[str]:
    """Return human-readable names of rules that apply to this test."""
    return [name for rule_test, name, _fn in EXCLUSION_RULES if rule_test == selected_test]


# --- Row filters: exclude rows by column values ---
# Each function(df, selected_test) -> pd.Series of bool (True = keep row).
# Each entry: (test this filter applies to, or None for all tests, description, function).


def row_filter_rata_min_90(df: pd.DataFrame, selected_test: str) -> pd.Series:
    """
    Keep only rows where Rata filtrarii glomerolare >= 90 or missing.
    Rows with Rata < 90 are excluded from statistics.
    """
    if RATA_FILTRARII not in df.columns:
        return pd.Series(True, index=df.index)
    s = df[RATA_FILTRARII].astype(str).str.strip().str.replace(",", ".", regex=False)
    num = pd.to_numeric(s, errors="coerce")
    return (num >= 90) | num.isna()


def row_filter_hba1c_exclude_leq_44(df: pd.DataFrame, selected_test: str) -> pd.Series:
    """
    For Hemoglobina glicata (Hb A1c): exclude results <= 4.4 (keep only Rezultat > 4.4).
    """
    if selected_test != HEMOGLOBINA_GLICATA_TEST:
        return pd.Series(True, index=df.index)
    if REZULTAT not in df.columns:
        return pd.Series(True, index=df.index)
    s = df[REZULTAT].astype(str).str.strip().str.replace(",", ".", regex=False)
    num = pd.to_numeric(s, errors="coerce")
    return num > 4.4


def row_filter_exclude_diagnostics_from_csv(df: pd.DataFrame, selected_test: str) -> pd.Series:
    """
    For analysis selected_test, exclude rows whose Diagnostic is marked DA in Diagnostice.csv
    (column = analysis, row = diagnostic, cell = DA).
    """
    # Custom diagnostic rules handle these tests explicitly.
    if selected_test in {SODIU_TEST, POTASIU_TEST, FOSFOR_TEST, AMILAZA_TEST, FIER_TEST}:
        return pd.Series(True, index=df.index)
    if DIAGNOSTIC not in df.columns:
        return pd.Series(True, index=df.index)
    exclude = _diagnostics_to_exclude_for_test(selected_test)
    if not exclude:
        return pd.Series(True, index=df.index)
    diag_normalized = df[DIAGNOSTIC].map(_normalize_diagnostic_value)
    return ~diag_normalized.isin(exclude)


def row_filter_mt_exclude_tumor_diagnostics(df: pd.DataFrame, selected_test: str) -> pd.Series:
    """
    For MT (markeri tumorali: CA 125, CA 15.3, CA 19.9), exclude rows whose Diagnostic
    contains 'tumora' or 'tumori' (case-insensitive), so statistics are not skewed by known tumor cases.
    """
    if selected_test not in MT_TESTS:
        return pd.Series(True, index=df.index)
    if DIAGNOSTIC not in df.columns:
        return pd.Series(True, index=df.index)
    diag = df[DIAGNOSTIC].astype(str).str.strip().str.lower()
    has_tumor = diag.str.contains("tumora|tumori", case=False, na=False, regex=True)
    return ~has_tumor


def row_filter_fosfor_correlate_with_calciu_magneziu(df: pd.DataFrame, selected_test: str) -> pd.Series:
    """
    For Fosfor: exclude diagnostics marked DA for Calciu and Magneziu columns.
    PTH is intentionally skipped until PTH statistics are available.
    """
    if selected_test != FOSFOR_TEST:
        return pd.Series(True, index=df.index)
    if DIAGNOSTIC not in df.columns:
        return pd.Series(True, index=df.index)
    exclude = _diagnostics_to_exclude_for_columns(["Calciu", "Magneziu", "PTH"])
    if not exclude:
        return pd.Series(True, index=df.index)
    diag_normalized = df[DIAGNOSTIC].map(_normalize_diagnostic_value)
    return ~diag_normalized.isin(exclude)


def row_filter_amilaza_correlate_with_tgo_tgp_and_keywords(df: pd.DataFrame, selected_test: str) -> pd.Series:
    """
    For Amilaza: exclude diagnostics marked DA for TGO/TGP and diagnostics containing
    pancreas/pancreatita keywords.
    """
    if selected_test != AMILAZA_TEST:
        return pd.Series(True, index=df.index)
    if DIAGNOSTIC not in df.columns:
        return pd.Series(True, index=df.index)
    exclude = _diagnostics_to_exclude_for_columns(["TGO", "TGP"])
    diag_normalized = df[DIAGNOSTIC].map(_normalize_diagnostic_value)
    diag_text = df[DIAGNOSTIC].astype(str).str.strip().str.lower()
    has_pancreas_terms = diag_text.str.contains("pancreas|pancreatita", case=False, na=False, regex=True)
    if exclude:
        return (~diag_normalized.isin(exclude)) & (~has_pancreas_terms)
    return ~has_pancreas_terms


def row_filter_fier_exclude_anemia_keywords(df: pd.DataFrame, selected_test: str) -> pd.Series:
    """For Fier: exclude diagnostics containing anemie/anemic/feripriva keywords."""
    if selected_test != FIER_TEST:
        return pd.Series(True, index=df.index)
    if DIAGNOSTIC not in df.columns:
        return pd.Series(True, index=df.index)
    diag_text = df[DIAGNOSTIC].astype(str).str.strip().str.lower()
    has_iron_deficiency_terms = diag_text.str.contains("anemie|anemia|anemic|feripriv", case=False, na=False, regex=True)
    return ~has_iron_deficiency_terms


# List of row filters: (test name or None for all, description, function).
# Only filters whose test matches selected_test (or test is None) are applied.
ROW_FILTERS = [
    (CREATININA_TEST, "Rata filtrarii glomerolare >= 90 (exclude < 90)", row_filter_rata_min_90),
    (HEMOGLOBINA_GLICATA_TEST, "Hemoglobina glicată: exclude rezultate <= 4.4", row_filter_hba1c_exclude_leq_44),
    (None, "Diagnostic exclus conform Diagnostice.csv (DA)", row_filter_exclude_diagnostics_from_csv),
    (FOSFOR_TEST, "Fosfor: corelat cu Calciu/Magneziu (PTH indisponibil)", row_filter_fosfor_correlate_with_calciu_magneziu),
    (AMILAZA_TEST, "Amilaza: corelat cu TGO/TGP + pancreas/pancreatita", row_filter_amilaza_correlate_with_tgo_tgp_and_keywords),
    (FIER_TEST, "Fier: exclude diagnostic anemie/anemic/feripriva", row_filter_fier_exclude_anemia_keywords),
    # MT: exclude rows with diagnostic containing tumora/tumori (one entry per MT test)
    ("CA 125", "MT: exclude diagnostic tumora/tumori", row_filter_mt_exclude_tumor_diagnostics),
    ("CA 15.3", "MT: exclude diagnostic tumora/tumori", row_filter_mt_exclude_tumor_diagnostics),
    ("CA 19.9", "MT: exclude diagnostic tumora/tumori", row_filter_mt_exclude_tumor_diagnostics),
]


def apply_row_filters(df: pd.DataFrame, selected_test: str) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """
    Apply row filters that apply to selected_test.
    Returns (filtered dataframe, list of (filter_name, n_excluded) for filters that excluded rows).
    """
    applied: list[tuple[str, int]] = []
    out = df
    for rule_test, name, rule_fn in ROW_FILTERS:
        if rule_test is not None and rule_test != selected_test:
            continue
        keep = rule_fn(out, selected_test)
        n_before = len(out)
        out = out.loc[keep]
        n_after = len(out)
        n_excluded = n_before - n_after
        if n_excluded > 0:
            applied.append((name, n_excluded))
    return out, applied
