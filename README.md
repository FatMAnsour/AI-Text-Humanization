# AI Text Humanizer

A **Django web app** that rewrites AI-generated text so it reads more like human writing and is less likely to be flagged by AI detectors. It uses multiple LLM providers, detector-aware prompts, optional dataset-driven style targets, and light post-processing to improve **perplexity** and **burstiness** so output is classified as human written.

---

## What it does

- **Humanize** pasted or pasted-in text: same meaning, similar length, more natural rhythm and word choice.
- **Support multiple providers**: Groq (default), or a stacked pipeline (Groq → Mistral → Cohere) for maximum variation.
- **One or two passes**: Single-pass quick rewrite, or two-pass (including optional “key-point rewrite”: extract phrases → rewrite from phrases) to break original structure.
- **Optional dataset tuning**: Compute sentence-length and burstiness stats from a Hugging Face dataset (e.g. Reddit human essays) so the humanizer targets more human-like stats.
- **API status page**: Check that Groq, Mistral, and Cohere API keys are valid and reachable.

---

## How it works

1. **Detector-aware prompts**  
   The system prompt tells the model that the output will be checked by AI detectors and must be classified as human. It enforces: varied sentence length (short fragments + longer sentences), contractions, active voice, no em dashes/semicolons, and a long list of banned formal/AI-sounding phrases.

2. **Providers**
   - **Groq (default)**  
     Uses Groq’s API. Supports 1 pass or 2 passes. With 2 passes you can use “key-point rewrite”: first pass extracts 5–8 short phrases, second pass writes one paragraph from those phrases only, which helps break the original structure.
   - **Stack4**  
     Runs the text through Groq, then Mistral, then Cohere. Each step uses the same anti-detection rules; later steps use higher temperature and penalties for more variation.

3. **Style stats (optional)**  
   A management command reads a Hugging Face dataset, computes human-writing stats (mean/std sentence length, % short/long sentences), and saves them to `humanizer/data/human_style_stats.json`. The humanizer uses these in the prompt for burstiness targets. If the file is missing or stats are rejected, built-in evasion-friendly defaults are used.

4. **Post-processing**  
   After the LLM rewrite: strip editor notes and markdown, replace formal phrases, remove em dashes and semicolons, then apply light “noise” (occasional filler words, dropped punctuation, optional comma drop, sentence merges) to reduce uniformity.

---

## Tech stack

- **Backend**: Django 5+
- **LLM APIs**: OpenAI-compatible clients for [Groq](https://groq.com), [Mistral](https://mistral.ai), [Cohere](https://cohere.com)
- **Config**: `python-dotenv` for `.env`; optional `datasets` + `huggingface_hub` for dataset-based stats

---

## Setup

1. **Clone and install**
   ```bash
   cd ai_humanize
   pip install -r requirements.txt
   ```

2. **Environment**
   Create a `.env` in the project root with at least:
   ```env
   GROQ_API_KEY=your_groq_key
   ```
   For stack4 (Groq + Mistral + Cohere):
   ```env
   GROQ_API_KEY=...
   MISTRAL_API_KEY=...
   COHERE_API_KEY=...
   ```

3. **Run**
   ```bash
   python manage.py runserver
   ```
   Open `http://127.0.0.1:8000/` for the humanizer UI and `http://127.0.0.1:8000/check-apis/` to verify API keys.

---

## Configuration (env)

| Variable | Description |
|----------|-------------|
| `HUMANIZE_PROVIDER` | `groq` (default) or `stack4` |
| `HUMANIZE_PASSES` | `1` or `2` (only for provider `groq`). Default `2`. |
| `HUMANIZE_KEYPOINT_REWRITE` | `true` (default) or `false`. When `true` and passes=2, use key-point rewrite (extract phrases → rewrite from phrases). |

---

## Usage

- **Web UI**  
  Paste text, choose style (casual, professional), audience, and purpose. Submit to get humanized text and word counts. Use the detector links to check the result on ZeroGPT/GPTZero.

- **Optional: human style stats from a dataset**  
  To tune burstiness from real human text (ex: Reddit):
  ```bash
  python manage.py compute_human_stats \
    --dataset jonathanli/human-essays-reddit \
    --split train --human-only --text-column top_comment \
    --max-samples 15000
  ```
  See `humanizer/data/RECOMMENDED_DATASETS.md` for dataset options and caveats.

---

## Project layout

```
ai_humanize/
├── manage.py
├── requirements.txt
├── .env                    
├── ai_humanize/            
│   ├── settings.py       
│   └── urls.py
└── humanizer/             
    ├── views.py           
    ├── style_stats.py    
    ├── data/
    │   ├── human_style_stats.json  
    │   └── RECOMMENDED_DATASETS.md
    ├── management/commands/
    │   └── compute_human_stats.py  # Dataset → human style stats
    └── templates/humanizer/
        ├── humanize.html
        └── api_status.html
```

---
