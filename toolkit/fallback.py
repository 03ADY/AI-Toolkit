"""Offline demo responses when backend is unavailable."""

MOCK = {
    "sentiment": {"label": "POSITIVE", "score": 0.94},
    "summarization": {"summary_text": "Enterprise AI toolkits unify NLP, speech, and vision behind one API gateway for faster delivery."},
    "generation": {"generated_text": "Thank you for your partnership. We look forward to expanding the pilot."},
    "translation": {"original_text": "", "translated_text": "Notre plateforme offre des services IA sécurisés et évolutifs."},
    "qa": {"answer": "Reduced latency and improved data privacy.", "confidence": 0.91, "sources": ["Provided Context"]},
    "caption": {"generated_text": "A demo industrial workshop with blue equipment and control panels."},
    "stt": {"transcribed_text": "The predictive maintenance system detected elevated vibration levels."},
    "tts": b"",  # empty bytes placeholder
}


def mock_response(service_id: str, payload: dict | None = None) -> dict:
    base = MOCK.get(service_id, {"message": "Demo mock response"})
    if service_id == "translation" and payload:
        base = {**base, "original_text": payload.get("text", "")}
    return base
