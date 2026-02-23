"""
Optional security: app password and CSV decryption.
Use Streamlit secrets or environment variables; no secrets in code.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path


def _normalize_fernet_key(key_raw: str | bytes) -> bytes:
    """Strip whitespace/BOM and return valid 32-byte base64 Fernet key as bytes. Raises if invalid."""
    if isinstance(key_raw, bytes):
        key_raw = key_raw.decode("utf-8", errors="replace")
    key_str = key_raw.strip().strip("\"'").replace("\r", "").replace("\n", "").strip()
    # Keep only valid base64url characters (A-Za-z0-9_-=)
    key_str = "".join(c for c in key_str if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=")
    if len(key_str) != 44:
        raise ValueError(
            f"Cheia Fernet trebuie să aibă exact 44 de caractere (base64). Ai {len(key_str)}. "
            "Copiază din nou cheia afișată la final când rulezi: python encrypt_data.py"
        )
    key_bytes = key_str.encode("ascii")
    try:
        decoded = base64.urlsafe_b64decode(key_bytes)
    except Exception as e:
        raise ValueError(
            "Cheia de criptare nu este base64 validă. Copiază exact cheia afișată de encrypt_data.py."
        ) from e
    if len(decoded) != 32:
        raise ValueError(
            f"Cheia decodată trebuie să fie 32 bytes (ai {len(decoded)}). Folosește cheia de la encrypt_data.py."
        )
    return key_bytes


def get_app_password() -> str | None:
    """Return app password if set (st.secrets or env APP_PASSWORD). None = no password required."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            p = st.secrets.get("app_password") or st.secrets.get("APP_PASSWORD")
            if p:
                return str(p).strip()
    except Exception:
        pass
    return os.environ.get("APP_PASSWORD", "").strip() or None


def get_encryption_key() -> bytes | None:
    """Return Fernet key if set (st.secrets, .streamlit/secrets.toml, or env CSV_ENCRYPTION_KEY). None = no decryption."""
    def try_key(k):
        if not k:
            return None
        if isinstance(k, bytes):
            k = k.decode("utf-8", errors="replace")
        k = k.strip().strip("\"'").replace("\r", "").replace("\n", "").strip()
        if not k:
            return None
        try:
            return _normalize_fernet_key(k)
        except ValueError:
            return None
    # 1) Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            k = st.secrets.get("encryption_key") or st.secrets.get("CSV_ENCRYPTION_KEY")
            out = try_key(k)
            if out:
                return out
    except Exception:
        pass
    # 2) Env
    raw = os.environ.get("CSV_ENCRYPTION_KEY", "").strip()
    out = try_key(raw)
    if out:
        return out
    # 3) Fallback: read .streamlit/secrets.toml from app directory
    try:
        app_dir = Path(__file__).resolve().parent
        secrets_file = app_dir / ".streamlit" / "secrets.toml"
        if secrets_file.exists():
            text = secrets_file.read_text(encoding="utf-8-sig")  # BOM strip
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("encryption_key") or line.startswith("CSV_ENCRYPTION_KEY"):
                    if "=" in line:
                        val = line.split("=", 1)[1].strip().strip('"').strip("'").replace("\r", "").replace("\n", "").strip()
                        if val:
                            return _normalize_fernet_key(val)
    except ValueError:
        raise
    except Exception:
        pass
    return None


def decrypt_content(raw: bytes, key: bytes) -> bytes:
    """Decrypt bytes with Fernet. Raises if cryptography not installed or key invalid."""
    try:
        from cryptography.fernet import Fernet
    except ImportError as e:
        raise ImportError(
            "Pentru fișiere criptate instalați: pip install cryptography"
        ) from e
    try:
        f = Fernet(key)
        return f.decrypt(raw)
    except Exception as e:
        err = str(e).lower()
        if "32 url-safe base64" in err or "fernet key" in err:
            raise ValueError(
                "Cheia din .streamlit/secrets.toml nu este validă pentru Fernet. "
                "Rulează din nou: python encrypt_data.py (cu aceeași parolă ca la criptare), "
                "copiază exact ultima linie afișată și pune-o în secrets.toml: encryption_key = \"cheia_ta\" (între ghilimele)."
            ) from e
        if "invalid" in err or "token" in err:
            raise ValueError(
                "Decriptare eșuată: cheia nu se potrivește cu fișierele (parolă greșită sau cheie schimbată). "
                "Folosește cheia generată când ai criptat aceste fișiere."
            ) from e
        raise


def read_file_content(path: Path, encryption_key: bytes | None = None) -> str:
    """
    Read file as text. If path ends with .encrypted and encryption_key is set, decrypt first.
    Otherwise read as UTF-8.
    """
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".encrypted" or path.name.endswith(".encrypted"):
        if encryption_key:
            raw = decrypt_content(raw, encryption_key)
        else:
            raise ValueError("Fișier criptat găsit dar nu e setată cheia (encryption_key / CSV_ENCRYPTION_KEY).")
    return raw.decode("utf-8")
