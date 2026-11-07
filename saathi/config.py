"""FIRST Saathi - configuration.

Tags used in comments:  SOURCED (public source)  ILLUSTRATIVE (our placeholder, replace
with bank data)  SYNTHETIC (demo data only).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KB_DIR = ROOT / "kb"
LOGS = ROOT / "logs"
AUDIO = ROOT / "audio_cache"
for p in (DATA, LOGS, AUDIO):
    p.mkdir(exist_ok=True)

# Running on a shared server (Streamlit Community Cloud mounts apps under /mount/src)?
# Then keys are never written to disk: one visitor's key must not reach the next visitor.
CLOUD = bool(os.environ.get("SAATHI_CLOUD")) or ROOT.as_posix().startswith("/mount/src")

KEYS_FILE = ROOT / ".saathi_keys.env"        # only written if the user ticks "remember on this computer"
TODAY = "2026-11-02"                          # the demo's "today"
SEED = 404

# ---------------------------------------------------------------------------
# LLM providers. Every one except Anthropic uses the OpenAI-compatible chat API.
# Model names change often: use "List models" in Settings to pick a live one.
# ---------------------------------------------------------------------------
PROVIDERS = {
    "Anthropic (Claude)": dict(kind="anthropic", base_url=None, default_model="claude-sonnet-5",
                               key_env="ANTHROPIC_API_KEY", key_url="https://console.anthropic.com"),
    "OpenAI": dict(kind="openai", base_url=None, default_model="gpt-4o-mini",
                   key_env="OPENAI_API_KEY", key_url="https://platform.openai.com/api-keys"),
    "Google Gemini (OpenAI-compatible)": dict(kind="openai", base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                                              default_model="gemini-2.5-flash", key_env="GEMINI_API_KEY",
                                              key_url="https://aistudio.google.com/apikey"),
    "Groq (OpenAI-compatible)": dict(kind="openai", base_url="https://api.groq.com/openai/v1",
                                     default_model="llama-3.3-70b-versatile", key_env="GROQ_API_KEY",
                                     key_url="https://console.groq.com/keys"),
    "OpenRouter (OpenAI-compatible)": dict(kind="openai", base_url="https://openrouter.ai/api/v1",
                                           default_model="meta-llama/llama-3.3-70b-instruct", key_env="OPENROUTER_API_KEY",
                                           key_url="https://openrouter.ai/keys"),
    "Ollama on this computer (no key)": dict(kind="openai", base_url="http://localhost:11434/v1",
                                             default_model="llama3.1", key_env="", key_url="https://ollama.com"),
    "Offline (no LLM)": dict(kind="offline", base_url=None, default_model="rules", key_env="", key_url=""),
}

LANGS = {   # code: (name, script, edge-tts voice, sarvam code)
    "en": ("English", "Latin", "en-IN-PrabhatNeural", "en-IN"),
    "hi": ("Hindi", "Devanagari or Roman Hinglish, mirror the connector", "hi-IN-MadhurNeural", "hi-IN"),
    "mr": ("Marathi", "Devanagari or Roman, mirror the connector", "mr-IN-ManoharNeural", "mr-IN"),
    "ta": ("Tamil", "Tamil", "ta-IN-ValluvarNeural", "ta-IN"),
    "te": ("Telugu", "Telugu", "te-IN-MohanNeural", "te-IN"),
    "gu": ("Gujarati", "Gujarati", "gu-IN-NiranjanNeural", "gu-IN"),
    "kn": ("Kannada", "Kannada", "kn-IN-GaganNeural", "kn-IN"),
    "bn": ("Bengali", "Bengali", "bn-IN-BashkarNeural", "bn-IN"),
}
FEMALE_EDGE = {"en": "en-IN-NeerjaNeural", "hi": "hi-IN-SwaraNeural", "mr": "mr-IN-AarohiNeural", "ta": "ta-IN-PallaviNeural",
               "te": "te-IN-ShrutiNeural", "gu": "gu-IN-DhwaniNeural", "kn": "kn-IN-SapnaNeural", "bn": "bn-IN-TanishaaNeural"}

ELEVEN_MODELS = ["eleven_multilingual_v2", "eleven_v3", "eleven_flash_v2_5"]
SARVAM_TTS_MODEL = "bulbul:v3"
SARVAM_STT_MODEL = "saaras:v4"

# ---------------------------------------------------------------------------
# Business rules (ILLUSTRATIVE unless marked)
# ---------------------------------------------------------------------------
HANDOFF_AMOUNT = 5_000_000          # leads above Rs 50 lakh bring the human RM in
NUDGE_CAP_PER_WEEK = 3
QUIET_HOURS = (21, 9)               # no nudges 9 pm - 9 am
PAYOUT_RATES = {                    # ILLUSTRATIVE payout as % of disbursed amount (replace with bank grid)
    "Personal Loan": 1.5, "Business Loan": 1.25, "Professional Loan": 1.0, "Home Loan": 0.4,
    "Loan Against Property": 0.6, "Used Car Loan": 1.0, "Two-Wheeler Loan": 1.5, "Education Loan": 0.75,
    "Credit Card": None,
}
CARD_FLAT_PAYOUT = 500              # ILLUSTRATIVE Rs per issued card
TIERS = [(0, "Bronze", 1.00), (2, "Silver", 1.05), (5, "Gold", 1.10), (10, "Platinum", 1.15)]  # ILLUSTRATIVE: disbursals in 90 days

SOURCES = {
    "MYFIRST": "https://www.idfcfirst.bank.in/myfirst-partner",
    "COMPLAINTS_FY26": "https://www.idfcfirst.bank.in/content/dam/idfcfirstbank/grievance-redressal/Complaints-Data-FY-2025-26.pdf",
    "DLD2025": "https://www.lawrbit.com/article/reserve-bank-of-india-digital-lending-directions-2025/",
    "FREEAI": "https://chambers.com/articles/rbi-committee-report-on-responsible-ai-in-the-financial-sector-free-ai-framework",
    "MRM2026": "https://www.rbi.org.in/scripts/bs_viewcontent.aspx?Id=5089",
    "DPDP": "https://yourstory.com/2025/11/digital-personal-data-protection-dpdp-rules-2025-rollout-over-18-months",
    "IOS2021": "https://www.idfcfirst.bank.in/content/dam/idfcfirstbank/pdf/ombudsman/Salient-Features-of-RB-Integrated-Ombudsman-Scheme-RBI-IOS-2021-English.pdf",
}
