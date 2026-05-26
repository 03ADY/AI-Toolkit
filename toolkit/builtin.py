"""In-process AI services — runs inside Streamlit (no FastAPI required)."""

from __future__ import annotations

import io
import re
from collections import Counter
from typing import Any

import streamlit as st

from toolkit.scenarios import DEMO_SAMPLES


def should_use_builtin() -> bool:
    from toolkit.cloud import is_streamlit_cloud

    if is_streamlit_cloud():
        return True
    return bool(st.session_state.get("builtin_mode", False))


@st.cache_resource(show_spinner="Loading built-in NLP (first use may take ~10s)…")
def _textblob_ready():
    import nltk
    from textblob import TextBlob

    for pkg in ("punkt", "punkt_tab", "brown"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return TextBlob


def _sentiment(text: str) -> dict:
    TextBlob = _textblob_ready()
    blob = TextBlob(text[:4000])
    pol = blob.sentiment.polarity
    label = "POSITIVE" if pol >= 0.05 else "NEGATIVE" if pol <= -0.05 else "NEUTRAL"
    if label == "NEUTRAL":
        label = "POSITIVE" if pol >= 0 else "NEGATIVE"
    score = min(0.99, max(0.51, 0.5 + abs(pol) / 2))
    return {"label": label, "score": float(score)}


def _summarize(text: str) -> dict:
    TextBlob = _textblob_ready()
    blob = TextBlob(text[:8000])
    sentences = [str(s).strip() for s in blob.sentences if str(s).strip()]
    if len(sentences) <= 2:
        return {"summary_text": text[:500]}
    words = re.findall(r"\w+", text.lower())
    freq = Counter(w for w in words if len(w) > 3)
    scored = []
    for i, sent in enumerate(sentences):
        score = sum(freq.get(w.lower(), 0) for w in re.findall(r"\w+", sent))
        scored.append((score / (i + 1) ** 0.3, sent))
    top = [s for _, s in sorted(scored, reverse=True)[:3]]
    top.sort(key=lambda s: sentences.index(s) if s in sentences else 0)
    return {"summary_text": " ".join(top)}


def _translate(text: str) -> dict:
    from deep_translator import GoogleTranslator

    translated = GoogleTranslator(source="en", target="fr").translate(text[:4500])
    return {"original_text": text, "translated_text": translated or ""}


def _generate(text: str) -> dict:
    prompt = text.strip()
    lower = prompt.lower()
    if "thank" in lower or "email" in lower:
        body = (
            "Thank you for your partnership. We appreciate the successful pilot and "
            "look forward to expanding the engagement next quarter."
        )
    elif "?" in prompt:
        body = (
            "This toolkit provides sentiment, summarization, translation, Q&A, image captioning, "
            "and speech services through a unified Streamlit interface."
        )
    else:
        body = f"Following up on your request: {prompt[:120]} — our team will respond with next steps shortly."
    return {"generated_text": body}


def _qa(payload: dict) -> dict:
    question = (payload or {}).get("question", "")
    context = (payload or {}).get("context", "")
    if not context.strip():
        return {
            "answer": "Add context text for extractive question answering.",
            "confidence": 0.0,
            "sources": [],
        }
    chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", context) if c.strip()]
    if not chunks:
        chunks = [context.strip()]
    q_words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 2}
    best, best_score = chunks[0], -1
    for c in chunks:
        c_words = set(re.findall(r"\w+", c.lower()))
        score = len(q_words & c_words)
        if score > best_score:
            best, best_score = c, score
    conf = min(0.95, 0.45 + best_score * 0.12) if best_score > 0 else 0.35
    return {"answer": best, "confidence": conf, "sources": ["Provided Context"]}


def _caption(image_bytes: bytes) -> dict:
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    thumb = img.resize((32, 32))
    pixels = list(thumb.getdata())
    avg = tuple(sum(c[i] for c in pixels) // len(pixels) for i in range(3))
    tone = "dark" if sum(avg) < 380 else "bright"
    return {
        "generated_text": (
            f"A {tone} {w}×{h} image (RGB average ~{avg}). "
            "Built-in vision uses lightweight analysis on Streamlit Cloud."
        ),
    }


def _stt(_payload: dict | None, file_bytes: bytes | None = None) -> dict:
    if not file_bytes:
        return {"transcribed_text": "Upload a short .wav file for built-in transcription."}
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(file_bytes)) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return {"transcribed_text": text}
    except Exception:
        return {
            "transcribed_text": (
                "Built-in STT could not decode this file. Try a clear English .wav under 30 seconds."
            ),
        }


def _tts(text: str) -> bytes:
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text[:500], lang="en").write_to_fp(buf)
    return buf.getvalue()


def invoke(service_id: str, payload: dict | None = None, *, file_bytes: bytes | None = None) -> Any:
    payload = payload or {}
    text = payload.get("text", "")

    if service_id == "sentiment":
        return _sentiment(text or DEMO_SAMPLES["sentiment"])
    if service_id == "summarization":
        return _summarize(text or DEMO_SAMPLES["summarization"])
    if service_id == "translation":
        return _translate(text or DEMO_SAMPLES["translation"])
    if service_id == "generation":
        return _generate(text or DEMO_SAMPLES["generation"])
    if service_id == "chatbot":
        return _generate(text or DEMO_SAMPLES["chatbot"])
    if service_id == "qa":
        return _qa(payload)
    if service_id == "caption":
        return _caption(file_bytes or b"")
    if service_id == "stt":
        return _stt(payload, file_bytes)
    if service_id == "tts":
        return _tts(text or "Hello from AI Toolkit built-in speech.")
    raise ValueError(f"Unknown built-in service: {service_id}")
