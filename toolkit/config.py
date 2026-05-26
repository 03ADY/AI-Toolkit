import os

APP_NAME = "AI Toolkit Enterprise"

API_ROOT = os.getenv("AI_TOOLKIT_API_ROOT", "http://127.0.0.1:8000").rstrip("/")
API_BASE = os.getenv("AI_TOOLKIT_API_BASE", f"{API_ROOT}/api")

SERVICES = [
    {"id": "sentiment", "name": "Sentiment Analysis", "endpoint": "/sentiment/analyze", "icon": "🎭"},
    {"id": "summarization", "name": "Text Summarization", "endpoint": "/summarization/summarize", "icon": "📄"},
    {"id": "generation", "name": "Text Generation", "endpoint": "/generation/generate", "icon": "✍️"},
    {"id": "captioning", "name": "Image Captioning", "endpoint": "/image/caption", "icon": "🖼️"},
    {"id": "translation", "name": "Translation", "endpoint": "/translation/translate", "icon": "🌍"},
    {"id": "tts", "name": "Text to Speech", "endpoint": "/tts", "icon": "🔊"},
    {"id": "stt", "name": "Speech to Text", "endpoint": "/stt", "icon": "🎤"},
    {"id": "chatbot", "name": "Chatbot", "endpoint": "/generation/generate", "icon": "💬"},
    {"id": "qa", "name": "Question Answering", "endpoint": "/qa/answer", "icon": "❓"},
]
