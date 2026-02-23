"""
Medical Results Distribution App
Loads lab CSV, filters by analysis and conditions (sex, age, city), shows Rezultat distribution.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rules import apply_exclusion_rules, apply_row_filters, get_active_rule_names

# Folders containing CSV files for all years. All CSVs in these folders are loaded and concatenated.
# Each folder can contain one or more analysis types (Test column: e.g. TGO, TGP in "TGO & TGP").
ANALYSIS_FOLDERS = ["Creatinina", "Hemoglobina", "Glucoza", "TGO & TGP", "ALP & GGT", "MT"]


def load_all_csvs_from_folders(base_path: Path | None = None) -> pd.DataFrame:
    """Load all *.csv (and *.csv.encrypted) from each analysis folder. Encrypted takes precedence over plain if both exist."""
    base = base_path or Path(__file__).parent
    all_dfs = []
    load_errors = []
    try:
        from security_utils import get_encryption_key
        has_key = bool(get_encryption_key())
    except Exception:
        has_key = False
    for folder_name in ANALYSIS_FOLDERS:
        folder = base / folder_name
        if not folder.is_dir():
            continue
        plain = {p.name: p for p in sorted(folder.glob("*.csv")) if p.suffix == ".csv"}
        encrypted = {p.name.removesuffix(".encrypted"): p for p in sorted(folder.glob("*.csv.encrypted"))}
        for name in sorted(set(plain) | set(encrypted)):
            path = encrypted.get(name) or plain.get(name)
            if path is None:
                continue
            try:
                all_dfs.append(load_csv(path))
            except Exception as e:
                load_errors.append(f"{path.name}: {e!s}")
    if not all_dfs and load_errors:
        raise RuntimeError(
            "Fișierele există dar nu s-au putut încărca. Posibil cheie de criptare incorectă sau fișiere deteriorate. "
            f"Prima eroare: {load_errors[0]}"
        )
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def get_data_folder_status(base_path: Path | None = None) -> tuple[Path, list[tuple[str, bool, int]]]:
    """Return (base_path, [(folder_name, exists, n_csv_or_encrypted)])."""
    base = base_path or Path(__file__).parent
    status = []
    for folder_name in ANALYSIS_FOLDERS:
        folder = base / folder_name
        exists = folder.is_dir()
        n = 0
        if exists:
            n = len(list(folder.glob("*.csv"))) + len(list(folder.glob("*.csv.encrypted")))
        status.append((folder_name, exists, n))
    return base, status

# Column name mapping after strip (header may have trailing spaces)
COL_NR_CRT = "Nr. crt."
COL_ID_CERERE = "IDCerere"
COL_DATA = "Data"
COL_VARSTA = "Varsta"
COL_CNP = "CNP"
COL_SEX = "Sex"
COL_LOCALITATE = "Localitate"
COL_TEST = "Test"
COL_REZULTAT = "Rezultat"
COL_UM = "UM"
COL_INTERVAL_REF = "Interval de referinta"
COL_PROBA = "Proba"
COL_METODA = "Metoda asociata"
COL_ECHIPAMENT = "Echipament"
COL_DIAGNOSTIC = "Diagnostic"
COL_RATA = "Rata filtrarii glomerolare"

# Analysis-specific histogram intervals (fallback when CSV not used). (low, high) or (low, high, left_open_inclusive). None = unbounded.
ANALYSIS_INTERVALS = {
    "Creatinina serica": [
        (None, 0.2),
        (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0),
        (1.0, 1.1), (1.1, 1.2), (1.2, 1.3),
        (1.3, None),
    ],
    "Hemoglobina glicata (Hb A1c)": [
        (None, 4.4, True),
        (4.4, 4.5), (4.5, 4.6), (4.6, 4.7), (4.7, 4.8), (4.8, 4.9), (4.9, 5.0),
        (5.0, 5.1), (5.1, 5.2), (5.2, 5.3), (5.3, 5.4), (5.4, 5.5), (5.5, 5.6),
        (5.7, 5.8), (5.8, 5.9), (5.9, 6.0), (6.0, 6.1),
        (6.1, None, True),  # >=6.1
    ],
    "Glucoza serica (glicemie)": [
        (None, 40),
        (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80),
        (80, 85), (85, 90), (90, 95), (95, 100), (100, 105), (105, 110), (110, 115), (115, 120),
        (120, 125),
        (125, None),
    ],
}

# Map data test names to CSV "Denumire" (exact or prefix match in get_csv_interval_sets_for_test)
INTERVALS_CSV_TEST_ALIASES = {"ALP": "Fosfataza alcalina", "GGT (gama glutamiltransferaza)": "GGT"}

INTERVALS_CSV_PATH = Path(__file__).parent / "Intervale de statistica.csv"


def _parse_interval_label(label: str) -> tuple | None:
    """Parse one interval label from Intervale CSV to (low, high) or (low, high, inclusive). Returns None if unparseable."""
    s = str(label).strip()
    if not s:
        return None
    # < 10, <10, <= 10
    m = re.match(r"^[<≤]\s*([\d.,]+)$", s, re.IGNORECASE)
    if m:
        high = float(m.group(1).replace(",", "."))
        return (None, high)
    # > 75, >75, >= 75
    m = re.match(r"^[>≥]\s*([\d.,]+)$", s, re.IGNORECASE)
    if m:
        low = float(m.group(1).replace(",", "."))
        return (low, None, True)  # right_inclusive for >=
    # 10-15, 30-40
    m = re.match(r"^([\d.,]+)\s*-\s*([\d.,]+)$", s)
    if m:
        low = float(m.group(1).replace(",", "."))
        high = float(m.group(2).replace(",", "."))
        return (low, high)
    return None


def load_intervals_from_csv(path: Path | None = None) -> list[dict]:
    """
    Load Intervale de statistica.csv. Returns list of:
    { "test": str, "condition_label": str, "age": "copii"|"adulti"|None, "equipment": "VITROS"|"ALTE ECHIPAMENTE"|None, "intervals": list of (low, high) tuples }
    """
    p = path or INTERVALS_CSV_PATH
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p, encoding="utf-8", dtype=str)
    except Exception:
        return []
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    col_names = list(df.columns)
    idx_obs = next((i for i, c in enumerate(col_names) if "Observatii" in str(c)), len(col_names))
    # Interval columns by index (2 to idx_obs-1) to avoid duplicate column names in CSV
    interval_start, interval_end = 2, idx_obs
    if interval_end <= interval_start:
        return []

    denumire_col = "Denumire"
    observatii_col = "Observatii"
    equip_col = col_names[idx_obs + 1] if idx_obs + 1 < len(col_names) else None
    # Build list of (test, condition_label, age, equipment, interval_labels). Propagate equipment from previous row when empty (same test block).
    result = []
    last_equipment = None
    last_denumire = None
    for _, row in df.iterrows():
        denumire = str(row.iloc[1]) if len(row) > 1 else ""
        denumire = denumire.strip() if isinstance(denumire, str) else ""
        if not denumire or denumire.lower() == "nan":
            continue
        if denumire != last_denumire:
            last_equipment = None
            last_denumire = denumire
        labels = []
        for j in range(interval_start, interval_end):
            if j >= len(row):
                break
            v = str(row.iloc[j]).strip()
            if v and v.lower() != "nan":
                labels.append(v)
        if not labels:
            continue
        obs = str(row.iloc[idx_obs]).strip() if idx_obs < len(row) else ""
        equip = str(row.iloc[idx_obs + 1]).strip() if equip_col is not None and idx_obs + 1 < len(row) else ""
        if equip and equip.lower() == "nan":
            equip = ""
        if "VITROS" in equip.upper():
            equipment = "VITROS"
            last_equipment = "VITROS"
        elif "ALTE" in equip.upper() or "ECHIPAMENTE" in equip.upper():
            equipment = "ALTE ECHIPAMENTE"
            last_equipment = "ALTE ECHIPAMENTE"
        else:
            equipment = last_equipment  # same block as previous row (e.g. Adulti after Copii VITROS)
        # Parse Observatii: "Copii: < 18 ani", "Adulti: >= 18 ani", "Indiferent de gen (M/F)"
        age = None
        if "Copii" in obs or "< 18" in obs:
            age = "copii"
        elif "Adulti" in obs or ">= 18" in obs or "≥ 18" in obs:
            age = "adulti"
        condition_parts = [obs] if obs else []
        if equipment:
            condition_parts.append(equipment)
        condition_label = " | ".join(condition_parts) if condition_parts else denumire
        parsed = []
        for lb in labels:
            t = _parse_interval_label(lb)
            if t is not None:
                parsed.append(t)
        if not parsed:
            continue
        result.append({
            "test": denumire,
            "condition_label": condition_label,
            "age": age,
            "equipment": equipment,
            "intervals": parsed,
        })
    return result


def get_csv_interval_sets_for_test(test_name: str, base_path: Path | None = None) -> list[dict]:
    """Return all interval sets from Intervale CSV for this test (with conditions). Empty if none."""
    sets_cache = getattr(load_intervals_from_csv, "_cache", None)
    if sets_cache is None:
        load_intervals_from_csv._cache = load_intervals_from_csv(base_path)
        sets_cache = load_intervals_from_csv._cache
    test_stripped = test_name.strip()
    canonical = INTERVALS_CSV_TEST_ALIASES.get(test_stripped, test_stripped)
    # Exact match first
    out = [s for s in sets_cache if s["test"] == canonical]
    if out:
        return out
    # Else match by prefix: CSV "GGT" matches data "GGT (gama glutamiltransferaza)"
    for s in sets_cache:
        csv_test = s["test"]
        if test_stripped.startswith(csv_test) or csv_test in test_stripped:
            out.append(s)
    return out


def get_intervals_for_test(test_name: str, interval_set_index: int | None = None, csv_interval_sets: list[dict] | None = None) -> list | None:
    """
    Return predefined bin edges for this test.
    If csv_interval_sets is provided and interval_set_index is not None, use that CSV set.
    Otherwise use ANALYSIS_INTERVALS (hardcoded) or first CSV set for this test.
    """
    if csv_interval_sets is not None and len(csv_interval_sets) > 0:
        idx = interval_set_index if interval_set_index is not None else 0
        if 0 <= idx < len(csv_interval_sets):
            return csv_interval_sets[idx]["intervals"]
    csv_sets = get_csv_interval_sets_for_test(test_name)
    if csv_sets:
        idx = interval_set_index if interval_set_index is not None else 0
        if 0 <= idx < len(csv_sets):
            return csv_sets[idx]["intervals"]
    return ANALYSIS_INTERVALS.get(test_name)


def bin_values_by_intervals(values: pd.Series, intervals: list) -> pd.DataFrame:
    """Assign each value to an interval. Intervals: (low, high) or (low, high, left_open_inclusive). None = unbounded."""
    rows = []
    for item in intervals:
        low, high = item[0], item[1]
        left_open_inclusive = item[2] if len(item) > 2 else False
        if low is None and high is not None:
            label = f"<={high}" if left_open_inclusive else f"<{high}"
            mask = values <= high if left_open_inclusive else values < high
        elif high is None and low is not None:
            right_inclusive = item[2] if len(item) > 2 else False
            label = f">={low}" if right_inclusive else f">{low}"
            mask = values >= low if right_inclusive else values > low
        elif low is not None and high is not None:
            label = f"{low} - {high}"
            mask = (values >= low) & (values < high)
        else:
            continue
        subset = values[mask]
        n = mask.sum()
        actual_min = float(subset.min()) if n > 0 else np.nan
        actual_max = float(subset.max()) if n > 0 else np.nan
        rows.append((label, n, actual_min, actual_max))
    return pd.DataFrame(rows, columns=["Interval", "Frecvență", "Min real", "Max real"])


def _read_csv_content(path_or_buffer) -> str:
    """Read CSV file or buffer to string. Supports optional decryption for .encrypted files."""
    import io
    if hasattr(path_or_buffer, "read"):
        content = path_or_buffer.read()
        return content.decode("utf-8") if isinstance(content, bytes) else content
    path = Path(path_or_buffer)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        from security_utils import get_encryption_key, read_file_content
        key = get_encryption_key()
        return read_file_content(path, key)
    except ImportError:
        return path.read_text(encoding="utf-8")
    except Exception:
        raise
    return path.read_text(encoding="utf-8")


def load_csv(path_or_buffer) -> pd.DataFrame:
    """Load CSV: find header row (contains 'Rezultat' or 'Nr. crt.'), strip column names. Supports .encrypted files if key is set."""
    import io
    if hasattr(path_or_buffer, "read"):
        content = path_or_buffer.read()
        content = content.decode("utf-8") if isinstance(content, bytes) else content
    else:
        content = _read_csv_content(path_or_buffer)
    lines = content.splitlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if "Rezultat" in line or "Nr. crt." in line:
            header_idx = i
            break
    content_from_header = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(content_from_header), encoding="utf-8", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_rezultat(series: pd.Series) -> pd.Series:
    """Convert Rezultat to numeric; handle European decimal comma."""
    s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def parse_varsta_to_age(series: pd.Series) -> pd.Series:
    """Parse 'X ani Y luni' to numeric age (years + months/12)."""
    def parse(s):
        if pd.isna(s) or not isinstance(s, str):
            return np.nan
        s = s.strip()
        # e.g. "57 ani 6 luni", "3 ani 2 luni", "85 ani"
        m_ani = re.search(r"(\d+)\s*ani", s)
        m_luni = re.search(r"(\d+)\s*luni", s)
        years = float(m_ani.group(1)) if m_ani else 0.0
        months = float(m_luni.group(1)) if m_luni else 0.0
        return years + months / 12.0

    return series.map(parse)


def parse_reference_interval(series: pd.Series):
    """Parse '65 - 110' style interval; return (low, high) or (None, None)."""
    if series.isna().all() or series.empty:
        return None, None
    s = series.dropna().iloc[0]
    if not isinstance(s, str):
        return None, None
    m = re.search(r"([\d,.\s]+)\s*-\s*([\d,.\s]+)", s)
    if not m:
        return None, None
    try:
        low = float(m.group(1).strip().replace(",", "."))
        high = float(m.group(2).strip().replace(",", "."))
        return low, high
    except ValueError:
        return None, None


def parse_year_from_data(series: pd.Series) -> pd.Series:
    """Parse year from Data column (e.g. '13.01.2025 7:08:00' or '08.01.2025 8:53:00')."""
    def parse(s):
        if pd.isna(s) or not isinstance(s, str):
            return np.nan
        s = s.strip()
        # DD.MM.YYYY or DD.MM.YYYY HH:MM:SS
        m = re.search(r"\d{2}\.\d{2}\.(\d{4})", s)
        return int(m.group(1)) if m else np.nan

    return series.map(parse)


@st.cache_data
def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean headers, normalize Rezultat, parse Varsta; drop rows with invalid Rezultat."""
    df = raw_df.copy()
    df[COL_REZULTAT] = normalize_rezultat(df[COL_REZULTAT])
    df["age"] = parse_varsta_to_age(df[COL_VARSTA])
    if COL_DATA in df.columns:
        df["year"] = parse_year_from_data(df[COL_DATA])
    if COL_SEX in df.columns:
        df[COL_SEX] = df[COL_SEX].astype(str).str.strip()
    if COL_LOCALITATE in df.columns:
        df[COL_LOCALITATE] = df[COL_LOCALITATE].astype(str).str.strip()
    if COL_TEST in df.columns:
        df[COL_TEST] = df[COL_TEST].astype(str).str.strip()
    if COL_ECHIPAMENT in df.columns:
        df[COL_ECHIPAMENT] = df[COL_ECHIPAMENT].astype(str).str.strip()
    # Keep rows with valid numeric Rezultat for distribution
    df = df.dropna(subset=[COL_REZULTAT])
    return df


def apply_filters(
    df: pd.DataFrame,
    test: str | None,
    sex: str,
    age_min: float | None,
    age_max: float | None,
    localitati: list[str] | None,
    echipamente: list[str] | None = None,
    years: list[int] | None = None,
    cnp: str | None = None,
) -> pd.DataFrame:
    """Apply Test, Sex, Age, Localitate, Echipament, Year, CNP filters."""
    out = df.copy()
    if test:
        out = out[out[COL_TEST] == test]
    if sex and sex != "Toate":
        out = out[out[COL_SEX] == sex]
    if age_min is not None and not np.isnan(age_min):
        out = out[out["age"] >= age_min]
    if age_max is not None and not np.isnan(age_max):
        out = out[out["age"] <= age_max]
    if localitati:
        out = out[out[COL_LOCALITATE].isin(localitati)]
    if echipamente and COL_ECHIPAMENT in out.columns:
        out = out[out[COL_ECHIPAMENT].isin(echipamente)]
    if years and "year" in out.columns:
        out = out[out["year"].isin(years)]
    if cnp and (cnp := cnp.strip()) and COL_CNP in out.columns:
        out = out[out[COL_CNP].astype(str).str.strip() == cnp]
    return out


def main():
    st.set_page_config(page_title="Distribuție rezultate analize", layout="wide")
    st.title("Distribuția rezultatelor analizelor medicale")

    # --- Optional app password (st.secrets app_password or env APP_PASSWORD) ---
    try:
        from security_utils import get_app_password
        app_password = get_app_password()
    except Exception:
        app_password = None
    if app_password:
        if st.session_state.get("_auth_ok"):
            pass
        else:
            p = st.text_input("Parolă aplicație", type="password", key="app_pwd")
            if p and p.strip() == app_password:
                st.session_state["_auth_ok"] = True
                st.rerun()
            elif p:
                st.error("Parolă incorectă.")
                st.stop()
            else:
                st.stop()

    # --- Data source: default files or upload ---
    use_upload = st.sidebar.checkbox("Încarcă alt fișier(e) CSV", value=False)
    if use_upload:
        uploaded = st.sidebar.file_uploader("Fișier(e) CSV", type=["csv"], accept_multiple_files=True)
        if uploaded:
            raw = pd.concat([load_csv(f) for f in uploaded], ignore_index=True)
        else:
            st.info("Încarcă unul sau mai multe fișiere CSV sau debifează pentru fișierele implicite.")
            return
    else:
        try:
            raw = load_all_csvs_from_folders()
        except RuntimeError as e:
            st.error(str(e))
            with st.expander("Detalii: unde caută aplicația și ce a găsit"):
                base, status = get_data_folder_status()
                st.markdown(f"**Unde caută:** `{base.resolve()}`\n\n" + "\n".join(
                    f"- **{name}**: există, {n} fișier(e)" if ex else f"- **{name}**: nu există"
                    for name, ex, n in status
                ))
            return
        if raw.empty:
            base, status = get_data_folder_status()
            lines = [f"**Unde caută aplicația:** `{base.resolve()}`", ""]
            for name, exists, n in status:
                if exists:
                    lines.append(f"- **{name}**: există, {n} fișier(e) .csv sau .csv.encrypted")
                else:
                    lines.append(f"- **{name}**: folderul nu există")
            st.error(
                "Niciun fișier CSV găsit în folderele de analize. "
                "Asigură-te că folderele există în același director cu app.py și conțin fișiere .csv sau .csv.encrypted."
            )
            with st.expander("Detalii: unde caută aplicația și ce a găsit"):
                st.markdown("\n".join(lines))
            return

    df = prepare_data(raw)
    if df.empty:
        st.warning("Nu există date cu Rezultat numeric valid.")
        return

    # --- Sidebar: Analysis and filters ---
    st.sidebar.header("Filtre")
    tests = sorted(df[COL_TEST].dropna().unique().tolist())
    if not tests:
        st.warning("Nu există coloana Test sau nu are valori.")
        return
    selected_test = st.sidebar.selectbox("Analiză (Test)", options=tests, index=0)

    cnp_search = st.sidebar.text_input("CNP (un singur pacient)", placeholder="ex: 2670704284393", help="Lasă gol pentru statistici pe toată populația.")

    sex_options = ["Toate", "F", "M"]
    existing_sex = df[COL_SEX].dropna().unique()
    selected_sex = st.sidebar.selectbox("Sex", options=sex_options, index=0)

    age_min_global = float(df["age"].min()) if df["age"].notna().any() else 0
    age_max_global = float(df["age"].max()) if df["age"].notna().any() else 120
    age_min = st.sidebar.number_input("Vârstă minimă (ani)", value=age_min_global, min_value=0.0, max_value=120.0, step=1.0)
    age_max = st.sidebar.number_input("Vârstă maximă (ani)", value=age_max_global, min_value=0.0, max_value=120.0, step=1.0)

    localitati_all = sorted(df[COL_LOCALITATE].dropna().unique().tolist())
    selected_localitati = st.sidebar.multiselect("Localitate (lasă gol = toate)", options=localitati_all, default=[])

    echipamente_all = sorted(df[COL_ECHIPAMENT].dropna().replace("", np.nan).dropna().unique().tolist()) if COL_ECHIPAMENT in df.columns else []
    selected_echipamente = st.sidebar.multiselect("Echipament (lasă gol = toate)", options=echipamente_all, default=[])

    years_all = sorted(df["year"].dropna().astype(int).unique().tolist()) if "year" in df.columns and df["year"].notna().any() else []
    selected_years = st.sidebar.multiselect("An (lasă gol = toți anii)", options=years_all, default=[], format_func=lambda x: str(int(x)))

    unique_cnp = st.sidebar.checkbox("Doar CNP unice (un rezultat per persoană)", value=False)

    filtered = apply_filters(
        df, selected_test,
        selected_sex if selected_sex != "Toate" else None,
        age_min, age_max,
        selected_localitati if selected_localitati else None,
        selected_echipamente if selected_echipamente else None,
        selected_years if selected_years else None,
        cnp_search.strip() or None,
    )

    # Apply exclusion rules (e.g. Glicemie: exclude CNP if Hb A1c > 6)
    exclude_cnps = apply_exclusion_rules(df, selected_test)
    if exclude_cnps:
        cnp_stripped = filtered[COL_CNP].astype(str).str.strip()
        before = len(filtered)
        filtered = filtered[~cnp_stripped.isin(exclude_cnps)]
        n_excluded = before - len(filtered)
        active_rules = get_active_rule_names(selected_test)
        if active_rules and n_excluded > 0:
            st.sidebar.caption(f"Reguli aplicate: {len(active_rules)}. Excluse: {n_excluded} rezultate.")

    # Apply row filters (e.g. Rata filtrarii glomerolare >= 90)
    before_row = len(filtered)
    filtered, applied_row_filters = apply_row_filters(filtered, selected_test)
    if applied_row_filters:
        lines = [f"**{name}:** {n:,} rânduri excluse" for name, n in applied_row_filters]
        st.sidebar.caption("Filtre pe rând:\n\n" + "\n\n".join(lines))

    if unique_cnp and COL_CNP in filtered.columns:
        before_dedup = len(filtered)
        # Keep most recent result per CNP (by Data); sort descending so keep="first" = latest
        if COL_DATA in filtered.columns:
            try:
                dt = pd.to_datetime(filtered[COL_DATA], format="%d.%m.%Y %H:%M:%S", errors="coerce")
                if dt.notna().any():
                    filtered = filtered.assign(_sort_date=dt).sort_values("_sort_date", ascending=False).drop(columns=["_sort_date"])
            except Exception:
                pass
        filtered = filtered.drop_duplicates(subset=[COL_CNP], keep="first")
        if before_dedup > len(filtered):
            st.sidebar.caption(f"**CNP unice:** {before_dedup:,} → {len(filtered):,} rezultate (unul per persoană).")

    values = filtered[COL_REZULTAT].dropna()

    if values.empty:
        st.warning("Nicio înregistrare după filtre. Relaxează filtrele.")
        return

    # --- Intervals: CSV for TGO, TGP, ALP, GGT. For ALP only: show each set as a tab (no sidebar selector). For others: use first set, no sidebar. ---
    csv_interval_sets = get_csv_interval_sets_for_test(selected_test)
    is_alp_tabs = selected_test in ("Fosfataza alcalina", "ALP") and len(csv_interval_sets) > 1
    interval_set_index = 0 if csv_interval_sets and not is_alp_tabs else None

    intervals = get_intervals_for_test(selected_test, interval_set_index=interval_set_index, csv_interval_sets=csv_interval_sets)
    use_custom_bins = intervals is not None
    if use_custom_bins:
        show_kde = False
    else:
        n_bins = st.sidebar.slider("Număr intervale histogramă", min_value=10, max_value=150, value=40)
        show_kde = st.sidebar.checkbox("Afișează curbă densitate (KDE)", value=True)

    ref_low, ref_high = None, None
    if COL_INTERVAL_REF in filtered.columns:
        ref_low, ref_high = parse_reference_interval(filtered[COL_INTERVAL_REF])

    # Optional: two vertical lines (hidden for ALP tabs to avoid key/context confusion)
    v_min, v_max = float(values.min()), float(values.max())
    if not is_alp_tabs:
        st.sidebar.subheader("Interval personalizat (linii verticale)")
        line1 = st.sidebar.number_input("Punct 1 (pe OX)", value=None, min_value=v_min, max_value=v_max, step=(v_max - v_min) / 100 if v_max > v_min else 0.1, format="%.2f", key="line1")
        line2 = st.sidebar.number_input("Punct 2 (pe OX)", value=None, min_value=v_min, max_value=v_max, step=(v_max - v_min) / 100 if v_max > v_min else 0.1, format="%.2f", key="line2")
    else:
        line1, line2 = None, None
    show_interval_lines = line1 is not None and line2 is not None

    if is_alp_tabs:
        # ALP only: one tab per interval set (Copii VITROS, Adulti VITROS, etc.); data filtered by age + equipment per tab
        tab_labels = [s["condition_label"] for s in csv_interval_sets]
        tabs = st.tabs(tab_labels)
        for tab, ival_set in zip(tabs, csv_interval_sets):
            with tab:
                # Subset filtered by this set's age and equipment
                mask_age = pd.Series(True, index=filtered.index)
                if "age" in filtered.columns:
                    if ival_set["age"] == "copii":
                        mask_age = filtered["age"] < 18
                    elif ival_set["age"] == "adulti":
                        mask_age = filtered["age"] >= 18
                mask_eq = pd.Series(True, index=filtered.index)
                if COL_ECHIPAMENT in filtered.columns and ival_set["equipment"]:
                    eq = filtered[COL_ECHIPAMENT].astype(str).str.upper()
                    if ival_set["equipment"] == "VITROS":
                        mask_eq = eq.str.contains("VITROS", na=False)
                    else:
                        mask_eq = ~eq.str.contains("VITROS", na=False)
                tab_df = filtered.loc[mask_age & mask_eq]
                values_tab = tab_df[COL_REZULTAT].dropna()
                if values_tab.empty:
                    st.info("Nicio înregistrare pentru acest set de filtre (vârstă + echipament).")
                    continue
                intervals_tab = ival_set["intervals"]
                binned_tab = bin_values_by_intervals(values_tab, intervals_tab)
                min_str = binned_tab["Min real"].apply(lambda x: str(x) if pd.notna(x) else "—")
                max_str = binned_tab["Max real"].apply(lambda x: str(x) if pd.notna(x) else "—")
                fig_tab = go.Figure()
                fig_tab.add_trace(go.Bar(
                    x=binned_tab["Interval"],
                    y=binned_tab["Frecvență"],
                    name="Frecvență",
                    opacity=0.7,
                    customdata=np.column_stack([min_str, max_str]),
                    hovertemplate="%{x}<br>Frecvență: %{y:,}<br>Min real: %{customdata[0]}<br>Max real: %{customdata[1]}<extra></extra>",
                ))
                fig_tab.update_layout(
                    title=f"Fosfataza alcalina – {ival_set['condition_label']} (n={len(values_tab):,})",
                    xaxis_title="Interval Rezultat",
                    yaxis_title="Frecvență",
                    xaxis_tickangle=-45,
                    showlegend=False,
                    height=450,
                )
                st.plotly_chart(fig_tab, width="stretch")
                st.subheader("Intervale: min–max real în date")
                display_tab = binned_tab.copy()
                display_tab["Min real"] = display_tab["Min real"].apply(lambda x: str(x) if pd.notna(x) else "—")
                display_tab["Max real"] = display_tab["Max real"].apply(lambda x: str(x) if pd.notna(x) else "—")
                n_rows = len(display_tab)
                table_height = min(36 * n_rows + 52, 1200)
                st.dataframe(display_tab[["Interval", "Frecvență", "Min real", "Max real"]], width="stretch", hide_index=True, height=table_height)
                unit_tab = tab_df[COL_UM].dropna().iloc[0] if COL_UM in tab_df.columns and tab_df[COL_UM].notna().any() else ""
                n_tab = len(values_tab)
                n_children_tab = int((tab_df.loc[values_tab.index, "age"] < 18).sum()) if "age" in tab_df.columns else 0
                pct_children_tab = (100.0 * n_children_tab / n_tab) if n_tab else 0
                st.subheader("Statistici descriptive")
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.metric("N", f"{n_tab:,}")
                with c2:
                    st.metric("Medie" + (f" ({unit_tab})" if unit_tab else ""), f"{values_tab.mean():.2f}")
                with c3:
                    st.metric("Mediană" + (f" ({unit_tab})" if unit_tab else ""), f"{values_tab.median():.2f}")
                with c4:
                    st.metric("Std.dev.", f"{values_tab.std():.2f}")
                with c5:
                    st.metric("IQR", f"{values_tab.quantile(0.75) - values_tab.quantile(0.25):.2f}")
                st.markdown(f"### Copii (vârstă < 18 ani) în populație: **{n_children_tab:,}** ({pct_children_tab:.1f}%)")
        return

    # Single chart (non-ALP or ALP with one set)
    fig = go.Figure()
    binned = None
    if use_custom_bins:
        binned = bin_values_by_intervals(values, intervals)
        # Format actual min/max for hover (empty interval → "—")
        min_str = binned["Min real"].apply(lambda x: str(x) if pd.notna(x) else "—")
        max_str = binned["Max real"].apply(lambda x: str(x) if pd.notna(x) else "—")
        hover_text = [
            f"Interval: {row['Interval']}<br>Frecvență: {row['Frecvență']:,}<br>Min real: {min_str.iloc[i]}<br>Max real: {max_str.iloc[i]}"
            for i, row in binned.iterrows()
        ]
        fig.add_trace(go.Bar(
            x=binned["Interval"],
            y=binned["Frecvență"],
            name="Frecvență",
            opacity=0.7,
            customdata=np.column_stack([min_str, max_str]),
            hovertemplate="%{x}<br>Frecvență: %{y:,}<br>Min real: %{customdata[0]}<br>Max real: %{customdata[1]}<extra></extra>",
        ))
        if show_interval_lines:
            interval_left = min(line1, line2)
            interval_right = max(line1, line2)
            # Shade bars that overlap [interval_left, interval_right] using Min/Max real
            for i, row in binned.iterrows():
                r_min, r_max = row["Min real"], row["Max real"]
                if pd.isna(r_min) and pd.isna(r_max):
                    continue
                lo = r_min if pd.notna(r_min) else -np.inf
                hi = r_max if pd.notna(r_max) else np.inf
                if interval_left < hi and interval_right > lo:
                    fig.add_vrect(x0=i - 0.5, x1=i + 0.5, fillcolor="blue", opacity=0.25, line_width=0, layer="below")
        fig.update_layout(
            title=f"Distribuție Rezultat – {selected_test} (n={len(values):,})",
            xaxis_title="Interval Rezultat",
            yaxis_title="Frecvență",
            xaxis_tickangle=-45,
            showlegend=False,
            height=450,
        )
    else:
        fig.add_trace(go.Histogram(x=values, nbinsx=n_bins, name="Frecvență", opacity=0.7))
        if show_kde and len(values) >= 2:
            from scipy import stats as scipy_stats
            try:
                kde = scipy_stats.gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 200)
                fig.add_trace(go.Scatter(x=x_range, y=kde(x_range) * len(values) * (values.max() - values.min()) / n_bins, mode="lines", name="KDE", line=dict(width=2)))
            except Exception:
                pass
        if ref_low is not None and ref_high is not None:
            fig.add_vrect(x0=ref_low, x1=ref_high, fillcolor="green", opacity=0.15, line_width=0)
            fig.add_vline(x=ref_low, line_dash="dash", line_color="green")
            fig.add_vline(x=ref_high, line_dash="dash", line_color="green")
        if show_interval_lines:
            interval_left = min(line1, line2)
            interval_right = max(line1, line2)
            fig.add_vrect(x0=interval_left, x1=interval_right, fillcolor="blue", opacity=0.2, line_width=0)
            fig.add_vline(x=interval_left, line_dash="solid", line_color="blue", line_width=2)
            fig.add_vline(x=interval_right, line_dash="solid", line_color="blue", line_width=2)
        fig.update_layout(
            title=f"Distribuție Rezultat – {selected_test} (n={len(values):,})",
            xaxis_title="Rezultat",
            yaxis_title="Frecvență",
            showlegend=True,
            height=450,
        )
    st.plotly_chart(fig, width="stretch")

    if show_interval_lines:
        interval_left = min(line1, line2)
        interval_right = max(line1, line2)
        in_interval = (values >= interval_left) & (values <= interval_right)
        count_interval = int(in_interval.sum())
        pct_interval = (100.0 * count_interval / len(values)) if len(values) else 0
        st.success(
            f"**Interval ales (OX):** {interval_left:.2f} — {interval_right:.2f}  \n"
            f"**Populație între cele 2 linii:** *n* = {count_interval:,} (**{pct_interval:.1f}%** din total)"
        )

    if use_custom_bins and binned is not None:
        st.subheader("Intervale: min–max real în date")
        display = binned.copy()
        display["Min real"] = display["Min real"].apply(lambda x: str(x) if pd.notna(x) else "—")
        display["Max real"] = display["Max real"].apply(lambda x: str(x) if pd.notna(x) else "—")
        n_rows = len(display)
        table_height = min(36 * n_rows + 52, 1200)
        st.dataframe(display[["Interval", "Frecvență", "Min real", "Max real"]], width="stretch", hide_index=True, height=table_height)

    # Summary stats
    unit = filtered[COL_UM].dropna().iloc[0] if COL_UM in filtered.columns and filtered[COL_UM].notna().any() else ""
    n_total = len(values)
    n_children = 0
    if "age" in filtered.columns:
        pop_ages = filtered.loc[values.index, "age"]
        n_children = int((pop_ages < 18).sum())
    pct_children = (100.0 * n_children / n_total) if n_total else 0

    st.subheader("Statistici descriptive")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("N", f"{n_total:,}")
    with c2:
        st.metric("Medie" + (f" ({unit})" if unit else ""), f"{values.mean():.2f}")
    with c3:
        st.metric("Mediană" + (f" ({unit})" if unit else ""), f"{values.median():.2f}")
    with c4:
        st.metric("Std.dev.", f"{values.std():.2f}")
    with c5:
        st.metric("IQR", f"{values.quantile(0.75) - values.quantile(0.25):.2f}")

    st.markdown(f"### Copii (vârstă < 18 ani) în populație: **{n_children:,}** ({pct_children:.1f}%)")


if __name__ == "__main__":
    main()
