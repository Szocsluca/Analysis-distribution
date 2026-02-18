# Distribuția rezultatelor analizelor medicale

Aplicație Streamlit care încarcă un export CSV de rezultate analize medicale, permite selectarea unui tip de analiză (Test) și afișează distribuția coloanei **Rezultat**, cu filtre opționale (sex, vârstă, localitate).

## Cerințe

- Python 3.10+
- Pachete din `requirements.txt`

## Instalare

```bash
pip install -r requirements.txt
```

## Rulare

```bash
streamlit run app.py
```

Aplicația va deschide în browser. Implicit se încarcă fișierul CSV din același folder cu `app.py`:

- `Statistica rezultate analize medicale 639066141340226722.csv`

Pentru alt fișier: în bara laterală bifează **Încarcă alt fișier CSV** și alege un fișier `.csv` cu aceeași structură (coloane: Test, Rezultat, Varsta, Sex, Localitate, Interval de referinta, UM, etc.).

## Funcționalități

- **Analiză (Test):** dropdown cu tipurile de analiză din CSV; afișarea distribuției se face doar pentru analiza selectată.
- **Filtre condiționale:**
  - **Sex:** Toate / F / M
  - **Vârstă:** minim și maxim (ani), pe baza coloanei Varsta (parsată din „X ani Y luni”).
  - **Localitate:** multi-select; dacă nu alegi niciuna, se afișează toate localitățile.
- **Histogramă** a Rezultatului, cu număr de intervale reglabil.
- **Curba KDE** (densitate) opțională.
- **Interval de referință** (dacă e parseabil în coloana „Interval de referinta”, ex. „65 - 110”) afișat ca bandă pe grafic.
- **Statistici:** N, medie, mediană, abatere standard, IQR și percentile.

## Structură CSV așteptată

- Prima linie poate fi goală; a doua linie = header.
- Coloane relevante: `Test`, `Rezultat` (numeric, eventual cu virgulă zecimală), `Varsta` (ex. „57 ani 6 luni”), `Sex` (M/F), `Localitate`, `UM`, `Interval de referinta`.
