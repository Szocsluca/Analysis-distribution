"""
Exclusion rules for analysis distributions.
Each rule returns the set of CNPs to EXCLUDE for a given test when building the distribution.
Add new rules here as we add more analyses.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Path to CSV that maps (Diagnostic, Analysis) -> "DA" = exclude that result for that analysis
DIAGNOSTICE_CSV_PATH = Path(__file__).parent / "Diagnostice.csv"

# Test names used in rules (must match CSV column "Test")
GLICEMIE_TEST = "Glucoza serica (glicemie)"
HEMOGLOBINA_GLICATA_TEST = "Hemoglobina glicata (Hb A1c)"
CREATININA_TEST = "Creatinina serica"

# Column names (must match prepared dataframe)
CNP = "CNP"
TEST = "Test"
REZULTAT = "Rezultat"
RATA_FILTRARII = "Rata filtrarii glomerolare"
DIAGNOSTIC = "Diagnostic"


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
    df["Diagnostic"] = df["Diagnostic"].astype(str).str.strip()
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
    if DIAGNOSTIC not in df.columns:
        return pd.Series(True, index=df.index)
    exclude = _diagnostics_to_exclude_for_test(selected_test)
    if not exclude:
        return pd.Series(True, index=df.index)
    diag_stripped = df[DIAGNOSTIC].astype(str).str.strip()
    return ~diag_stripped.isin(exclude)


# List of row filters: (test name or None for all, description, function).
# Only filters whose test matches selected_test (or test is None) are applied.
ROW_FILTERS = [
    (CREATININA_TEST, "Rata filtrarii glomerolare >= 90 (exclude < 90)", row_filter_rata_min_90),
    (HEMOGLOBINA_GLICATA_TEST, "Hemoglobina glicată: exclude rezultate <= 4.4", row_filter_hba1c_exclude_leq_44),
    (None, "Diagnostic exclus conform Diagnostice.csv (DA)", row_filter_exclude_diagnostics_from_csv),
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
