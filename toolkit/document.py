"""Multi-service document analysis."""

from toolkit.helpers import call_api


def analyze_document(text: str, *, offline: bool = False) -> dict:
    results = {"length": len(text), "word_count": len(text.split())}
    ok, sent, err, ms1 = call_api("/sentiment/analyze", {"text": text[:512]}, offline=offline)
    results["sentiment"] = sent if ok else {"error": err}
    results["sentiment_ms"] = ms1

    if len(text) > 80:
        ok2, summ, err2, ms2 = call_api(
            "/summarization/summarize",
            {"text": text},
            offline=offline,
            timeout=120,
        )
        results["summary"] = summ if ok2 else {"error": err2}
        results["summary_ms"] = ms2
    else:
        results["summary"] = {"summary_text": "Text too short to summarize."}
        results["summary_ms"] = 0

    words = [w.lower() for w in text.split() if len(w) > 4]
    top = sorted(set(words), key=words.count, reverse=True)[:8]
    results["keywords"] = top
    return results
