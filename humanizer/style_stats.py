import json
import os

# Evasion-friendly: more short sentences, moderate long (detectors flag uniform/long-heavy text).
DEFAULT_STATS = {
    "mean_sentence_length": 14,
    "std_sentence_length": 8,
    "pct_short_sentences": 0.20,
    "pct_long_sentences": 0.22,
    "short_max_words": 5,
    "long_min_words": 20,
}

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_STATS_PATH = os.path.join(_DATA_DIR, "human_style_stats.json")


def load_human_style_stats():
    """Load stats from JSON if present and valid, else return defaults.
    Rejects 'formal' profiles (too few short / too many long sentences) that worsen AI detection."""
    if os.path.isfile(_STATS_PATH):
        try:
            with open(_STATS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = {**DEFAULT_STATS, **data}
            mean_len = merged.get("mean_sentence_length", 14)
            pct_short = merged.get("pct_short_sentences", 0.20) or 0.20
            pct_long = merged.get("pct_long_sentences", 0.22) or 0.22
            # Reject nonsensical mean
            if mean_len is not None and (mean_len < 3 or mean_len > 150):
                return DEFAULT_STATS.copy()
            # Reject formal/long-heavy stats — evasion works better with more short sentences, fewer long
            if pct_short < 0.15 or pct_long > 0.28:
                return DEFAULT_STATS.copy()
            return merged
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_STATS.copy()


def get_style_target_prompt():
    """
    Build a short prompt hint from loaded stats (burstiness targets) for inclusion in system prompts.
    Uses human_style_stats.json from compute_human_stats (or research defaults).
    """
    s = load_human_style_stats()
    pct_short = int((s.get("pct_short_sentences", 0.20) or 0.20) * 100)
    pct_long = int((s.get("pct_long_sentences", 0.22) or 0.22) * 100)
    short_max = s.get("short_max_words", 5)
    long_min = s.get("long_min_words", 20)
    mean_len = s.get("mean_sentence_length", 14)
    return (
        f"Target burstiness: about {pct_short}% of sentences 2–{short_max} words, "
        f"about {pct_long}% with {long_min}+ words; average ~{mean_len} words. Vary rhythm (e.g. medium, short, long, short)."
    )
