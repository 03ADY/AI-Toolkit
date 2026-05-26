"""Demo samples for presenter flow."""

DEMO_SAMPLES = {
    "sentiment": "The product launch exceeded expectations and customer feedback has been overwhelmingly positive.",
    "summarization": """
    Artificial intelligence is transforming enterprise software. Teams deploy models for document understanding,
    customer support automation, and predictive analytics. Self-hosted inference reduces latency and keeps data
    on-premises. A unified toolkit lets developers experiment with NLP, speech, and vision APIs behind one gateway.
    """.strip(),
    "generation": "Write a professional email thanking a client for a successful pilot project.",
    "translation": "Our platform delivers secure, scalable AI services for modern businesses.",
    "chatbot": "Hello! Can you help me understand what services this toolkit provides?",
    "qa": {
        "question": "What is the main benefit of self-hosted AI?",
        "context": "Self-hosted AI inference reduces latency, improves privacy, and gives teams control over model versions and costs.",
    },
}

MENU_TO_SAMPLE = {
    "Sentiment Analysis": "sentiment",
    "Text Summarization": "summarization",
    "Creative Text Generation": "generation",
    "Language Translation": "translation",
    "AI Chatbot": "chatbot",
    "Question Answering": "qa",
    "Document Analyzer": "summarization",
    "Service Compare": "sentiment",
}


def get_sample(key: str):
    return DEMO_SAMPLES.get(key, "")
