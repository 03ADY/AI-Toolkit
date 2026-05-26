# AI Toolkit Enterprise — Demo

**Streamlit Cloud:** runs **built-in AI** inside the app — no API URL, no `127.0.0.1`, no separate server. Open any service from the sidebar and click the action button.

For **full Hugging Face models** (heavier), run `start-demo.ps1` locally or deploy `merged_backend.py` and use **Advanced: external FastAPI API** in the sidebar (local only).


```powershell
.\scripts\start-demo.ps1
```

API **:8000** · UI **:8506**

## Full tour (3 min)

1. Sidebar → **Backend ready** or **Offline demo mode**  
2. **Home** → service health grid + session cost  
3. **Demo Runner** → full NLP suite  
4. **Document Analyzer** → sentiment + summary + keywords  
5. **Service Compare** → same text, two APIs  
6. **Image Captioning** → **Use demo image** (no upload needed)  
7. **Integrations Hub** → API keys, webhooks, governance export  
8. **Cost Center** → rate card + session estimate  
9. **Session Report** → download HTML  

## Offline mode

Toggle in sidebar — runs full UI without GPU/backend (mock responses).
