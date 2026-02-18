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

# Folders (one per analysis) containing CSV files for all years. All CSVs in these folders are loaded and concatenated.
ANALYSIS_FOLDERS = ["Creatinina", "Hemoglobina", "Glucoza"]


def load_all_csvs_from_folders(base_path: Path | None = None) -> pd.DataFrame:
    """Load all *.csv files from each analysis folder and concatenate. Returns one raw DataFrame."""
    base = base_path or Path(__file__).parent
    all_dfs = []
    for folder_name in ANALYSIS_FOLDERS:
        folder = base / folder_name
        if not folder.is_dir():
            continue
        for csv_path in sorted(folder.glob("*.csv")):
            try:
                all_dfs.append(load_csv(csv_path))
            except Exception:
                continue
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

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

# Analysis-specific histogram intervals. (low, high) or (low, high, left_open_inclusive). None = unbounded.
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


def get_intervals_for_test(test_name: str) -> list | None:
    """Return predefined bin edges for this test, or None to use automatic bins."""
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


def load_csv(path_or_buffer) -> pd.DataFrame:
    """Load CSV: find header row (contains 'Rezultat' or 'Nr. crt.'), strip column names."""
    import io
    if hasattr(path_or_buffer, "read"):
        content = path_or_buffer.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        lines = content.splitlines()
    else:
        with open(path_or_buffer, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n\r") for line in f.readlines()]
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
        raw = load_all_csvs_from_folders()
        if raw.empty:
            st.error(
                f"Niciun fișier CSV găsit în folderele: {ANALYSIS_FOLDERS}. "
                "Asigură-te că folderele există în același director cu app.py și conțin fișiere .csv."
            )
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

    # --- Main: histogram and stats ---
    intervals = get_intervals_for_test(selected_test)
    use_custom_bins = intervals is not None
    if use_custom_bins:
        show_kde = False
    else:
        n_bins = st.sidebar.slider("Număr intervale histogramă", min_value=10, max_value=150, value=40)
        show_kde = st.sidebar.checkbox("Afișează curbă densitate (KDE)", value=True)

    ref_low, ref_high = None, None
    if COL_INTERVAL_REF in filtered.columns:
        ref_low, ref_high = parse_reference_interval(filtered[COL_INTERVAL_REF])

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
        fig.update_layout(
            title=f"Distribuție Rezultat – {selected_test} (n={len(values):,})",
            xaxis_title="Rezultat",
            yaxis_title="Frecvență",
            showlegend=True,
            height=450,
        )
    st.plotly_chart(fig, width="stretch")

    if use_custom_bins and binned is not None:
        st.subheader("Intervale: min–max real în date")
        display = binned.copy()
        display["Min real"] = display["Min real"].apply(lambda x: str(x) if pd.notna(x) else "—")
        display["Max real"] = display["Max real"].apply(lambda x: str(x) if pd.notna(x) else "—")
        st.dataframe(display[["Interval", "Frecvență", "Min real", "Max real"]], width="stretch", hide_index=True)

    # Summary stats
    unit = filtered[COL_UM].dropna().iloc[0] if COL_UM in filtered.columns and filtered[COL_UM].notna().any() else ""
    st.subheader("Statistici descriptive")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("N", f"{len(values):,}")
    with c2:
        st.metric("Medie" + (f" ({unit})" if unit else ""), f"{values.mean():.2f}")
    with c3:
        st.metric("Mediană" + (f" ({unit})" if unit else ""), f"{values.median():.2f}")
    with c4:
        st.metric("Std.dev.", f"{values.std():.2f}")
    with c5:
        st.metric("IQR", f"{values.quantile(0.75) - values.quantile(0.25):.2f}")

    # Modal interval: interval with highest frequency ("where most of the population is")
    n_total = len(values)
    if use_custom_bins and binned is not None:
        idx_max = binned["Frecvență"].idxmax()
        modal_row = binned.loc[idx_max]
        modal_interval = modal_row["Interval"]
        modal_count = int(modal_row["Frecvență"])
    else:
        counts, edges = np.histogram(values, bins=n_bins)
        i_max = np.argmax(counts)
        modal_count = int(counts[i_max])
        modal_interval = f"{edges[i_max]:.2f} - {edges[i_max + 1]:.2f}"
    pct = (100.0 * modal_count / n_total) if n_total else 0
    st.info(
        f"**Intervalul în care se încadrează cea mai mare parte a populației (mod):** {modal_interval} "
        f"— *n* = {modal_count:,} ({pct:.1f}%)"
    )

    with st.expander("Percentile"):
        p = [1, 5, 25, 50, 75, 95, 99]
        perc = values.quantile([x / 100.0 for x in p])
        st.dataframe(pd.DataFrame({"Percentil": p, "Rezultat": perc.values}), width="stretch")


if __name__ == "__main__":
    main()
