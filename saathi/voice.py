"""Voice in and voice out.

TTS (the Digital RM speaks):
  - ElevenLabs: the RM's own cloned voice (instant voice clone from consented samples).
    Keys are typed by you in Settings, never hard-coded.
  - Sarvam Bulbul: Indian-language neural voices (11 languages), needs a Sarvam key.
  - edge-tts: free Microsoft neural voices for Indian languages, no key (needs internet).
STT (the connector sends a voice note):
  - Sarvam Saaras: 22+ Indian languages with auto-detect.
  - OpenAI transcription (whisper-1 / gpt-4o-transcribe).
Every generated voice note is labelled as AI-generated in the UI.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import threading

import requests

from . import config as C

TIMEOUT = 60


class VoiceError(RuntimeError):
    pass


def _cache_path(*parts, ext="mp3"):
    h = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]
    return C.AUDIO / f"{h}.{ext}"


# ------------------------------------------------------------------ TTS
def tts(text: str, lang: str, provider: str, keys: dict, gender: str = "M", sarvam_speaker: str = "") -> tuple[bytes, str]:
    text = text.strip()[:1400]
    if not text:
        raise VoiceError("Nothing to speak")
    if provider == "ElevenLabs (RM's cloned voice)":
        return tts_elevenlabs(text, keys.get("ELEVENLABS_API_KEY", ""), keys.get("ELEVENLABS_VOICE_ID", ""),
                              keys.get("ELEVENLABS_MODEL") or C.ELEVEN_MODELS[0]), "audio/mpeg"
    if provider == "Sarvam (Indian voices)":
        return tts_sarvam(text, lang, keys.get("SARVAM_API_KEY", ""), sarvam_speaker), "audio/wav"
    if provider == "edge-tts (free, no key)":
        voice = (C.LANGS.get(lang, C.LANGS["en"])[2] if gender == "M" else C.FEMALE_EDGE.get(lang, C.FEMALE_EDGE["en"]))
        return tts_edge(text, voice), "audio/mpeg"
    raise VoiceError("Voice is off")


def tts_elevenlabs(text: str, key: str, voice_id: str, model_id: str) -> bytes:
    if not key or not voice_id:
        raise VoiceError("ElevenLabs needs an API key and a voice ID (create the RM's clone in Persona studio)")
    p = _cache_path("el", voice_id, model_id, text)
    if p.exists():
        return p.read_bytes()
    r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                      headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                      json={"text": text, "model_id": model_id,
                            "voice_settings": {"stability": 0.5, "similarity_boost": 0.85, "style": 0.2, "use_speaker_boost": True}},
                      timeout=TIMEOUT)
    if r.status_code != 200:
        raise VoiceError(f"ElevenLabs error {r.status_code}: {r.text[:200]}")
    p.write_bytes(r.content)
    return r.content


def clone_voice_elevenlabs(key: str, name: str, files: list[tuple[str, bytes]], description: str = "") -> str:
    """Instant voice clone. Call ONLY after the RM has given written consent (Persona studio enforces this)."""
    if not key:
        raise VoiceError("ElevenLabs API key is empty")
    form = {"name": name, "description": description or "FIRST Saathi RM voice (consented)", "remove_background_noise": "true"}
    r = None
    for field in ("files", "files[]"):          # docs show both spellings; try the plain one first
        multipart = [(field, (fn, data, "audio/mpeg")) for fn, data in files]
        r = requests.post("https://api.elevenlabs.io/v1/voices/add", headers={"xi-api-key": key},
                          data=form, files=multipart, timeout=120)
        if r.status_code == 200:
            break
    if r.status_code != 200:
        raise VoiceError(f"ElevenLabs clone error {r.status_code}: {r.text[:300]}")
    return r.json()["voice_id"]


def tts_sarvam(text: str, lang: str, key: str, speaker: str = "") -> bytes:
    if not key:
        raise VoiceError("Sarvam API key is empty")
    code = C.LANGS.get(lang, C.LANGS["en"])[3]
    p = _cache_path("sv", code, speaker, text, ext="wav")
    if p.exists():
        return p.read_bytes()
    body = {"text": text, "language_code": code, "model": C.SARVAM_TTS_MODEL, "pace": 1.0, "speech_sample_rate": 22050}
    if speaker:
        body["speaker"] = speaker
    r = requests.post("https://api.sarvam.ai/text-to-speech", headers={"api-subscription-key": key, "Content-Type": "application/json"},
                      json=body, timeout=TIMEOUT)
    if r.status_code != 200:
        raise VoiceError(f"Sarvam TTS error {r.status_code}: {r.text[:200]}")
    audio = base64.b64decode(r.json()["audios"][0])
    p.write_bytes(audio)
    return audio


def tts_edge(text: str, voice: str) -> bytes:
    p = _cache_path("edge", voice, text)
    if p.exists():
        return p.read_bytes()
    try:
        import edge_tts
    except ImportError as e:
        raise VoiceError("Package 'edge-tts' missing: run SETUP_AND_RUN.bat again") from e

    err = []

    def run():
        try:
            asyncio.run(edge_tts.Communicate(text, voice).save(str(p)))
        except Exception as ex:  # network blocked, voice name changed, ...
            err.append(ex)

    t = threading.Thread(target=run)           # Streamlit already runs an event loop in some versions
    t.start()
    t.join(TIMEOUT)
    if err or not p.exists():
        raise VoiceError(f"edge-tts failed (needs internet): {err[0] if err else 'timeout'}")
    return p.read_bytes()


# ------------------------------------------------------------------ STT
def stt(audio: bytes, filename: str, provider: str, keys: dict, lang: str = "unknown") -> str:
    if provider == "Sarvam (Indian languages)":
        key = keys.get("SARVAM_API_KEY", "")
        if not key:
            raise VoiceError("Sarvam API key is empty")
        r = requests.post("https://api.sarvam.ai/speech-to-text", headers={"api-subscription-key": key},
                          files={"file": (filename, audio)},
                          data={"model": C.SARVAM_STT_MODEL, "language_code": C.LANGS.get(lang, (0, 0, 0, "unknown"))[3] if lang != "unknown" else "unknown"},
                          timeout=TIMEOUT)
        if r.status_code != 200:
            raise VoiceError(f"Sarvam STT error {r.status_code}: {r.text[:200]}")
        return r.json().get("transcript", "")
    if provider == "OpenAI transcription":
        key = keys.get("OPENAI_STT_KEY") or (keys.get("LLM_API_KEY") if keys.get("LLM_PROVIDER") == "OpenAI" else "")
        if not key:
            raise VoiceError("OpenAI key for transcription is empty")
        r = requests.post("https://api.openai.com/v1/audio/transcriptions", headers={"Authorization": f"Bearer {key}"},
                          files={"file": (filename, audio)}, data={"model": "whisper-1"}, timeout=TIMEOUT)
        if r.status_code != 200:
            raise VoiceError(f"OpenAI transcription error {r.status_code}: {r.text[:200]}")
        return r.json().get("text", "")
    raise VoiceError("Voice input is off: pick a speech-to-text provider in Settings")
