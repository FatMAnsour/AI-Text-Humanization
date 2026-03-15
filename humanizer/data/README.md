# Dataset-driven human style stats

- **`human_style_stats.json`** – Target sentence-length (burstiness) stats used by the humanizer to match human writing distribution. Defaults are from research; you can overwrite with stats from a real dataset.

- **Compute stats from a Hugging Face dataset** (optional):
  ```bash
  pip install datasets
  cd /path/to/ai_humanize
  python manage.py compute_human_stats
  ```
  This loads a dataset of human-written text, computes mean/std and % short/long sentences, and saves to `human_style_stats.json`. The humanizer then uses these values in the anti-detection prompt.

- **Example with a different dataset or label:**
  ```bash
  python manage.py compute_human_stats --dataset dmitva/human_ai_generated_text --human-label 0 --max-samples 3000
  ```
