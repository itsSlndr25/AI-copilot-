# AI 客服 Copilot — Product Case Study

A product prototype demonstrating how an AI-powered Copilot can help customer service agents respond faster and more consistently, without replacing human judgment.

**Live Demo → [your-streamlit-url-here]**

---

## Product Design

This prototype is part of a full product case study covering:
- User research & problem definition
- Feature prioritization (2x2 matrix)
- Product roadmap (MVPv1.0 → v2.0 → v3.0)
- Grounded answering from a knowledge base (simplified RAG: the whole FAQ fits in the prompt)
- Testing across 3 boundary conditions
- Risk assessment & success metrics (OKR framework)

**Key Design Decisions:**
- **Human-in-the-loop in v1.0**: All AI suggestions require agent confirmation before sending — builds trust before automating
- **Anti-hallucination guardrail**: When the knowledge base has no answer, AI must say so rather than guess; cited FAQ IDs are checked in code against the knowledge base
- **Structured output**: The model must return JSON matching a schema, validated with Pydantic before display
- **Knowledge base layering (design only, not implemented)**: Customer-facing FAQ and internal SOP should be separated to prevent data leakage

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Frontend | Streamlit |
| AI Model | Google Gemini (free tier) via `google-genai` SDK |
| Knowledge | 15 mock FAQ entries placed in the system instruction |
| Deployment | Streamlit Community Cloud |

---

## Run Locally

**Step 1: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Set up API key**

Edit `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_key_here"
```

**Step 3: Run**
```bash
streamlit run app.py
```

**Tests (offline, mocked API):**
```bash
python test_app.py
```

---

## Project Structure

```
copilot_app/
├── app.py                  # Main application
├── test_app.py             # UI/flow tests with a mocked Gemini client
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── secrets.toml        # API key (not committed to GitHub)
├── .gitignore
└── README.md
```

---

*Built by I-Hang CHEN*
