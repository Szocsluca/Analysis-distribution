"""
Encrypt CSV files in the analysis folders for storage at rest.
Run once to produce .csv.encrypted files; keep the key in st.secrets or env (CSV_ENCRYPTION_KEY).
Usage: python encrypt_data.py [--password "your-pass"] [--folders "Creatinina,Hemoglobina,..."]
Without --password, reads from env CSV_ENCRYPTION_PASSWORD (used to derive key) or prompts.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from pathlib import Path

# Folders to encrypt (default: same as app)
DEFAULT_FOLDERS = ["Creatinina", "Hemoglobina", "Glucoza", "TGO & TGP", "ALP & GGT", "MT"]


def derive_key(password: str) -> bytes:
    """Derive a Fernet key from password (32 url-safe base64 bytes)."""
    from cryptography.fernet import Fernet
    h = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(h)


def encrypt_file(path: Path, key: bytes, out_path: Path | None = None) -> None:
    """Encrypt file at path; write to out_path or path + '.encrypted'."""
    from cryptography.fernet import Fernet
    raw = path.read_bytes()
    f = Fernet(key)
    encrypted = f.encrypt(raw)
    target = out_path or Path(str(path) + ".encrypted")
    target.write_bytes(encrypted)
    print(f"  {path.name} -> {target.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt CSV files in data folders.")
    parser.add_argument("--password", "-p", help="Password to derive encryption key (or set CSV_ENCRYPTION_PASSWORD)")
    parser.add_argument("--folders", "-f", default=",".join(DEFAULT_FOLDERS), help="Comma-separated folder names")
    parser.add_argument("--dry-run", action="store_true", help="List files that would be encrypted")
    args = parser.parse_args()
    password = args.password or __import__("os").environ.get("CSV_ENCRYPTION_PASSWORD", "").strip()
    if not password:
        import getpass
        password = getpass.getpass("Parolă pentru criptare (va deriva cheia): ")
    if not password:
        print("Eroare: parola nu poate fi goală.", file=sys.stderr)
        sys.exit(1)
    try:
        key = derive_key(password)
    except ImportError:
        print("Instalează: pip install cryptography", file=sys.stderr)
        sys.exit(1)
    base = Path(__file__).parent
    folders = [s.strip() for s in args.folders.split(",") if s.strip()]
    for folder_name in folders:
        folder = base / folder_name
        if not folder.is_dir():
            print(f"Folder negăsit: {folder}")
            continue
        print(folder_name)
        for path in sorted(folder.glob("*.csv")):
            if path.name.endswith(".encrypted"):
                continue
            if args.dry_run:
                print(f"  (dry-run) {path.name} -> {path.name}.encrypted")
            else:
                encrypt_file(path, key)
    if not args.dry_run:
        print("Cheia (base64) pentru st.secrets sau CSV_ENCRYPTION_KEY:")
        print(key.decode())


if __name__ == "__main__":
    main()
