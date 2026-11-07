"""API keys: typed in the Settings page, kept in memory for the session.

Optional: tick "Remember on this computer" to save them to .saathi_keys.env in this folder
(plain text, readable by anyone who can open the folder: do not tick it on a shared PC).
Environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY,
OPENROUTER_API_KEY, ELEVENLABS_API_KEY, SARVAM_API_KEY) are also picked up.
"""
from __future__ import annotations

import os

from . import config as C

NAMES = ["LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID",
         "ELEVENLABS_MODEL", "SARVAM_API_KEY", "OPENAI_STT_KEY", "TTS_PROVIDER", "STT_PROVIDER"]


def _secrets() -> dict:
    """Values from Streamlit secrets (.streamlit/secrets.toml locally, or the app's Secrets box on
    Streamlit Community Cloud). Anything put here is shared by every visitor of the app."""
    try:
        import streamlit as st
        return {k: str(v) for k, v in st.secrets.items() if k in NAMES or k.startswith("ELEVENLABS_VOICE_ID_")}
    except Exception:
        return {}


def load() -> dict:
    out = {}
    if C.KEYS_FILE.exists() and not C.CLOUD:
        for line in C.KEYS_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    for k, v in _secrets().items():
        out.setdefault(k, v)
    for env in ("ELEVENLABS_API_KEY", "SARVAM_API_KEY"):
        if os.environ.get(env) and not out.get(env):
            out[env] = os.environ[env]
    if not out.get("LLM_API_KEY"):
        for name, p in C.PROVIDERS.items():
            if p["key_env"] and os.environ.get(p["key_env"]):
                out.setdefault("LLM_PROVIDER", name)
                out["LLM_API_KEY"] = os.environ[p["key_env"]]
                break
    return out


def save(values: dict):
    if C.CLOUD:
        raise RuntimeError("Keys are never saved on a shared server")
    lines = ["# FIRST Saathi keys - saved because 'Remember on this computer' was ticked. Delete this file to forget them."]
    lines += [f"{k}={values.get(k, '')}" for k in NAMES if values.get(k)]
    lines += [f"{k}={v}" for k, v in values.items() if k.startswith("ELEVENLABS_VOICE_ID_") and v]
    C.KEYS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def forget():
    if C.KEYS_FILE.exists():
        C.KEYS_FILE.unlink()


def mask(v: str) -> str:
    return (v[:4] + "..." + v[-4:]) if v and len(v) > 10 else ("set" if v else "not set")
