# Securitate – date medicale

Datele din CSV (CNP, rezultate, diagnostice) sunt sensibile. Recomandări:

## 1. Rulare locală

- Rulați aplicația doar pe calculatorul dumneavoastră: `streamlit run app.py`
- În `.streamlit/config.toml` puteți seta `server.address = "127.0.0.1"` ca să asculte doar pe localhost (nu pe rețea).

## 2. Excludere date din Git

- Nu comitați folderele cu CSV-uri sau fișierele cu date. În `.gitignore` sunt incluse (sau pot fi decomentate) folderele de date și `Diagnostice.csv`.
- Verificați: `Creatinina/`, `Hemoglobina/`, `Glucoza/`, `TGO & TGP/`, `ALP & GGT/`, `MT/`, `Intervale de statistica.csv`, `Diagnostice.csv`.

## 3. Parolă aplicație (opțional)

- Puteți cere o parolă înainte de a accesa aplicația.
- Creați `.streamlit/secrets.toml` (nu se comite!) și adăugați:
  ```toml
  app_password = "parola-dorita"
  ```
  sau setați variabila de mediu `APP_PASSWORD`.

## 4. Criptare CSV la rest (opțional)

- Puteți stoca fișierele CSV criptate; aplicația le citește doar dacă este setată cheia.
- Pași:
  1. Instalați: `pip install cryptography`
  2. Rulați: `python encrypt_data.py` (va cere o parolă și va crea fișiere `.csv.encrypted`)
  3. Salvați cheia afișată la final și puneți-o în `.streamlit/secrets.toml`:
     ```toml
     encryption_key = "cheia-afisata-base64"
     ```
     sau în variabila de mediu `CSV_ENCRYPTION_KEY`.
  4. Opțional: ștergeți fișierele `.csv` necriptate după ce ați verificat că `.csv.encrypted` funcționează.
- Aplicația încarcă automat fișierele `.csv.encrypted` (dacă există cheia); dacă există atât `.csv` cât și `.csv.encrypted`, se folosește varianta criptată.

## 5. Permisiuni fișiere

- Restrângeți accesul la folderele cu date: doar utilizatorul care rulează aplicația ar trebui să aibă drept de citire (pe Windows: drepturi pe folder; pe Linux/macOS: `chmod 700` pe folderele de date).

## 6. Fără jurnalare sensibile

- Aplicația nu scrie CNP, rezultate sau diagnostice în fișiere de log. Nu hardcodați parole sau chei în cod.
