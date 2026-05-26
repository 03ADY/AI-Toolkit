"""Demo cost estimator for API usage."""

# USD per 1k calls (illustrative demo pricing)
RATE_CARD = {
    "Sentiment Analysis": 0.15,
    "Text Summarization": 0.45,
    "Creative Text Generation": 0.35,
    "Image Captioning": 0.55,
    "Language Translation": 0.25,
    "Text to Speech": 0.60,
    "Speech to Text": 0.70,
    "AI Chatbot": 0.35,
    "Question Answering": 0.30,
}


def estimate_session_cost(history: list[dict]) -> dict:
    total = 0.0
    breakdown = {}
    for h in history:
        svc = h.get("service", "Unknown")
        rate = RATE_CARD.get(svc, 0.20)
        breakdown[svc] = breakdown.get(svc, 0) + 1
        total += rate / 1000
    return {"total_usd": round(total, 4), "breakdown": breakdown, "calls": len(history)}
