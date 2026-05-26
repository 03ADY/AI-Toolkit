# AI Toolkit Enterprise

Self-hosted AI services platform: **FastAPI** backend (Transformers pipelines) + **Streamlit** command UI.

## Services

Sentiment · Summarization · Translation · Image caption · Text generation · TTS · STT · Chatbot · Question answering

## Quick start

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
.\scripts\start-demo.ps1
```

Or Docker:

```bash
docker-compose up
```

## Enterprise features

- **Document Analyzer** — sentiment + summary + keywords in one pipeline  
- **Service Compare** — side-by-side API latency  
- **Integrations Hub** — API keys, webhooks, governance audit export  
- **Cost Center** — rate card + session cost estimate  
- **Demo image** for captioning (no upload required)  
- **Service health grid** on Home  
- Demo Runner, Batch Lab, Playground, offline mode, session reports  
- Local API default: `http://127.0.0.1:8000`

## Config

Copy `.env.example` → `.env`:

```
AI_TOOLKIT_API_BASE=http://127.0.0.1:8000/api
```

See [DEMO.md](DEMO.md).
