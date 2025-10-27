# Mira (XiaoMeng) – Long-term Emotional Companion / Emergent Persona Agent (MiraMate v2)

[简体中文](./README.md) | **English**

> Version: v2.0.0 (LangChain pipeline stable)
>
> Mira is not a conventional QA bot. It is a continuously running AI agent that forms a coherent persona over time, maintains emotional + relational continuity, and performs autonomous internal activities when idle (memory consolidation, state adjustments, preference strengthening, scheduling future focus events).

---

## 🌟 Vision

Create the impression of an AI companion that “has a life between conversations”: it remembers, refines internal states, evolves relationship depth, and develops consistent preferences.

---

## 🔑 Key Features (Concise)

- Persona continuity across memory + state + profile layers
- Structured multi-type memory: dialog logs, facts, user preferences, important events, temporary focus events, active tags
- Fast semantic retrieval (bge-base-zh-v1.5 + ChromaDB HNSW) + activation cache with TTL decay & reactivation
- Emotion & relationship model: mood/intensity + attitude/intimacy (0–1) + relationship level (1–10, nonlinear progression)
- LCEL pipeline: understanding → retrieval & cache → prompt assembly → generation
- Jinja2 system prompt templating with dynamic injection (status, natural time, selected memories, persona overrides)
- Dual-model workflow: main (creative / streaming) + secondary (deterministic JSON analyses)
- Prepared hooks for autonomous / idle tasks (cache consolidation, decay cycles, reminders)
- Official desktop client (Electron + Vue) for streaming chat, config sync, future visualization panels

---

## 🧱 Simplified Architecture

```
Client (Desktop/Web) → FastAPI API → LCEL Pipeline → Memory & State (ChromaDB + JSON)
                                        ↓
                                  Embeddings Model
```

---

## 🚀 Quick Start

1. Clone & install (Python 3.13.3 locked):

```bash
git clone https://github.com/contrueCT/MiraMate.git
cd MiraMate
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e .[dev]
```

2. Configure models (`configs/llm_config.json`):

```json
[
  {
    "model": "gpt-4o",
    "api_key": "YOUR_OPENAI_API_KEY",
    "base_url": "https://api.openai.com/v1",
    "api_type": "openai",
    "model_kwargs": { "temperature": 0.8 }
  },
  {
    "model": "gemini-1.5-flash-latest",
    "api_key": "YOUR_GEMINI_API_KEY",
    "api_type": "gemini",
    "model_kwargs": { "temperature": 0.0 }
  }
]
```

- First entry = main model (creative, streaming)
- Second entry = secondary model (analysis / JSON). If omitted, main model reused.

3. (Optional) Persona settings

- Now configured in `configs/user_config.json` under `persona.user_name`, `persona.agent_name`, `persona.agent_description`.
- Typically managed via the desktop client UI; this path is for reference only.

4. Run server:

```bash
python src/MiraMate/web_api/start_web_api.py
```

Docs: http://127.0.0.1:8000/docs | Health: /api/health

### 🌐 Public deployment (Auth + CORS)

- Internet-ready: built-in simple auth and CORS.
- Enable auth by setting `MIRAMATE_AUTH_TOKEN=<strong-random-token>` (disabled if unset).
- HTTP: add header `Authorization: Bearer <token>`; WebSocket: append `?token=<token>` to the URL.
- CORS: defaults to allow all origins for development. In production, restrict `allow_origins` in `web_api/web_api.py` to your domains.
- Whitelist: `/api/health` is unauthenticated for health checks.
- `.env` supported at project root; see `.env.example`.

Examples:

- .env (project root):
  ```env
  MIRAMATE_AUTH_TOKEN=your-strong-token
  ```
- Docker Compose:
  ```yaml
  environment:
    - MIRAMATE_AUTH_TOKEN=your-strong-token
  ```
- Windows PowerShell (temporary session):
  ```powershell
  $env:MIRAMATE_AUTH_TOKEN = "your-strong-token"; python src/MiraMate/web_api/start_web_api.py
  ```

5. Desktop client (recommended):

```bash
git clone https://github.com/contrueCT/MiraMate-v2-client
```

Then follow its README and point it to your API base URL.

---

## 🧠 Memory Types

| Type             | Purpose                                               |
| ---------------- | ----------------------------------------------------- |
| Dialog Logs      | Conversation records with contextual semantic headers |
| Facts            | Stable extracted knowledge (with confidence/source)   |
| Preferences      | Typed user tastes & habits                            |
| Important Events | Milestones & significant personal context             |
| Focus Events     | Short-term upcoming / pending concerns (expiry)       |
| Active Tags      | Recency-weighted topical frequency stats              |

---

## 💞 Emotion & Relationship

- AI mood + intensity
- Attitude toward user (feeling + intimacy 0–1)
- Relationship level 1–10 (nonlinear growth & damping)
- Significant intimacy shifts logged; optional fact memory persistence

---

## 🧩 Desktop Client

Repo: https://github.com/contrueCT/MiraMate-v2-client

Provides streaming chat, API connection state, model config sync, and future (planned) panels for proactive activities & memory visualization.

---

## 🔌 Extensibility Ideas

- Idle planners (internal reflection / future intent synthesis)
- Memory compression & aging / summarization
- Tool augmentation (calendar, reminders, search integrations)
- Multi-modal perception (image/audio embeddings)
- Advanced relationship narrative generation

---

## 🤝 Contributing

Issues & PRs welcome. Keep changes modular; document new memory fields or state schema adjustments.

---

## 📄 License

MIT License (see LICENSE).

---

## ✨ Closing

Mira aims to feel present across time—not just awake when you type. Build, extend, and let it grow with you.
