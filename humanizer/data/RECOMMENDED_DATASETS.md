# Recommended datasets for human-style stats

Use `python manage.py compute_human_stats` only if you want to **override** the built-in burstiness targets. The app uses a **detector-aware pipeline**: explicit “must be classified as human” instructions, evasion-friendly defaults (20% short, 22% long), in-context human-style examples, high temperature (0.96–0.98 on final pass), key-point rewrite (extract phrases → rewrite from phrases), and light post-process noise (fillers, comma drops, sentence merges). You can skip the dataset step and humanize directly.

**More samples (`--max-samples`) = more stable stats:** Using 10k–20k samples (default is 15k) gives a better estimate of real human sentence-length distribution, which can make the humanized output more consistently human-like. Going much beyond 20k usually has diminishing returns.

**Optional:** If you see *"You are sending unauthenticated requests to the HF Hub"*, you can set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in your `.env` (create a token at https://huggingface.co/settings/tokens) for higher rate limits and faster downloads. The command still works without it.

---

## Use these (they load and give usable stats)

### 1. **Reddit creative writing (human-only)** — best for human-like stats

**Dataset:** `jonathanli/human-essays-reddit`  
- **38,885** human essays from r/WritingPrompts. All human; no AI. Strong for natural sentence variation.  
- **You must use `--text-column top_comment`** (the text column is `top_comment`, not `response`).

```bash
python manage.py compute_human_stats \
  --dataset jonathanli/human-essays-reddit \
  --split train \
  --human-only \
  --text-column top_comment \
  --max-samples 15000
```

---

### 2. **Social media human vs AI (AIGTBench)**

**Dataset:** `tarryzhang/AIGTBench`  
- **845k** samples from Reddit, Quora, Medium. Label **0** = human, **1** = AI.  
- If your humanized text still scores badly after this, use the **built-in defaults** instead (delete or don’t run `compute_human_stats` and rely on the app’s defaults).

```bash
python manage.py compute_human_stats \
  --dataset tarryzhang/AIGTBench \
  --split train \
  --human-label 0 \
  --max-samples 15000
```

---

### 3. **Labeled human/AI (inokusan)**

**Dataset:** `inokusan/human_ai_text_classification`  
- **113k** samples. Label **0** = human, **1** = AI.  
- Can be formal; if stats are rejected (formal profile), the app will write evasion defaults and warn you.

```bash
python manage.py compute_human_stats \
  --dataset inokusan/human_ai_text_classification \
  --split train \
  --human-label 0 \
  --max-samples 15000
```

---

## Datasets that don’t work or give worse results

- **barilan/blog_authorship_corpus** — uses a loading script that is **no longer supported** by Hugging Face `datasets`. The command will fail or suggest Parquet fallback; use one of the datasets above instead.
- **silentone0725/ai-human-text-detection-v1** — often produces **formal, long-heavy** stats (e.g. 40% long, 11% short). The app **rejects** these and uses built-in defaults; humanization can get worse if you force such stats.
- Other **detection-benchmark** mixes (e.g. merged academic/professional corpora) — same issue: formal human text → worse detection scores. Prefer **Reddit** or **AIGTBench** for casual stats.

---

## If nothing works or accuracy is bad

1. **Don’t run `compute_human_stats`** — the app uses built-in **evasion-friendly defaults** (20% short, 22% long, mean ~14 words). Many users get better, more consistent results with these than with dataset-derived stats.
2. Use **Reddit** with `--text-column top_comment` for creative-style stats.
3. After humanizing, test on ZeroGPT/GPTZero; if scores are still bad, rely on defaults and tune **prompts/settings** (provider, passes) rather than more datasets.
