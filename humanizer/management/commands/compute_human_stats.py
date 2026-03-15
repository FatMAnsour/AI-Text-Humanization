"""
Compute human-writing style stats from a Hugging Face dataset and save to human_style_stats.json.
Run: python manage.py compute_human_stats

Requires: pip install datasets
Uses dataset with human/AI labels and keeps only human text, then computes sentence-length
distribution (mean, std, % short, % long) for burstiness targets.
"""
import json
import os
import re

from django.core.management.base import BaseCommand


def _sentence_lengths(text):
    """Return list of word counts per sentence."""
    if not (text and isinstance(text, str)):
        return []
    sentences = re.split(r"[.!?]+", text)
    return [len(s.split()) for s in sentences if len(s.split()) > 0]


class Command(BaseCommand):
    help = "Compute human style stats from Hugging Face dataset and save to humanizer/data/human_style_stats.json"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dataset",
            default="silentone0725/ai-human-text-detection-v1",
            help="Hugging Face dataset name (must have 'human' label or similar)",
        )
        parser.add_argument(
            "--split",
            default="train",
            help="Dataset split to use",
        )
        parser.add_argument(
            "--max-samples",
            type=int,
            default=15000,
            help="Max human samples to use for stats (default 15000). More samples = more stable, human-like burstiness targets; 10k–20k is a good range.",
        )
        parser.add_argument(
            "--human-label",
            default="human",
            help="Label value for human text (e.g. human, 0). Ignored if --human-only is set.",
        )
        parser.add_argument(
            "--human-only",
            action="store_true",
            help="Dataset has only human text; use all rows (no label filter). Use with jonathanli/human-essays-reddit.",
        )
        parser.add_argument(
            "--text-column",
            default="",
            help="Use this column for text (e.g. response, human_answer). Overrides auto-detection.",
        )

    def handle(self, *args, **options):
        try:
            from datasets import load_dataset
        except ImportError:
            self.stderr.write(
                self.style.ERROR("Install Hugging Face datasets: pip install datasets")
            )
            return

        dataset_name = options["dataset"]
        split = options["split"]
        max_samples = options["max_samples"]
        human_label = options["human_label"]
        human_only = options["human_only"]
        text_column = (options["text_column"] or "").strip()

        self.stdout.write(f"Loading dataset {dataset_name} (split={split})...")
        try:
            ds = load_dataset(dataset_name, split=split)
        except Exception as e1:
            err_msg = str(e1)
            # Dataset loading scripts are deprecated in datasets>=4.0; try Parquet export if available
            if "script" in err_msg.lower() or "no longer supported" in err_msg.lower() or "trust_remote_code" in err_msg.lower():
                ds = None
                try:
                    from huggingface_hub import list_repo_files
                    prefix = f"refs/convert/parquet/{split}"
                    files = list_repo_files(dataset_name, repo_type="dataset")
                    parquet_files = [f for f in files if f.startswith(prefix) and f.endswith(".parquet")]
                    if parquet_files:
                        data_files = [f"hf://datasets/{dataset_name}/{f}" for f in parquet_files]
                        full = load_dataset("parquet", data_files=data_files)
                        if split in full:
                            ds = full[split]
                        elif "train" in full:
                            ds = full["train"]
                        else:
                            ds = next(iter(full.values()))
                except Exception:
                    pass
                if ds is None:
                    self.stderr.write(
                        self.style.ERROR(
                            f"Failed to load dataset: {err_msg}\n\n"
                            "This dataset uses a loading script, which is no longer supported. "
                            "Use a Parquet-based dataset instead, e.g.:\n"
                            "  --dataset tarryzhang/AIGTBench --human-label 0\n"
                            "  --dataset jonathanli/human-essays-reddit --human-only --text-column top_comment\n"
                            "  --dataset inokusan/human_ai_text_classification --human-label 0\n"
                            "Or skip compute_human_stats; the app uses built-in evasion-friendly defaults."
                        )
                    )
                    return
            else:
                self.stderr.write(self.style.ERROR(f"Failed to load dataset: {err_msg}"))
                return

        # Text column: explicit, or infer
        if text_column and text_column in ds.column_names:
            text_col = text_column
        else:
            text_col = None
            for c in ds.column_names:
                if c.lower() in ("text", "content", "sentence", "paragraph", "response", "human_answer", "answer", "top_comment", "body"):
                    text_col = c
                    break
            if not text_col:
                text_col = ds.column_names[0]

        # Label column: only used when not human_only and no text_column override for human-only datasets
        label_col = None if human_only else None
        if not human_only:
            for c in ds.column_names:
                if c.lower() in ("label", "labels", "class", "generated", "source"):
                    label_col = c
                    break
            if not label_col and "label" in ds.column_names:
                label_col = "label"

        lengths = []
        count = 0
        for row in ds:
            if count >= max_samples:
                break
            if not human_only and label_col is not None:
                val = row.get(label_col)
                if val is not None:
                    val_str = str(val).strip().lower()
                    label_str = str(human_label).strip().lower()
                    try:
                        if val_str != label_str and int(val) != int(human_label):
                            continue
                    except (ValueError, TypeError):
                        if val_str != label_str:
                            continue
            text = row.get(text_col) or row.get("text", "")
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            for n in _sentence_lengths(str(text)):
                lengths.append(n)
            count += 1

        if not lengths:
            self.stderr.write(
                self.style.ERROR(
                    "No human text found. Try --human-label 0 or --human-label human, or --human-only with --text-column."
                )
            )
            return

        import statistics
        mean_len = statistics.mean(lengths)
        std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0
        short_max = 5
        long_min = 20
        pct_short = sum(1 for n in lengths if n <= short_max) / len(lengths)
        pct_long = sum(1 for n in lengths if n >= long_min) / len(lengths)

        # Reject nonsensical stats (wrong column or broken sentence split)
        if mean_len < 3 or mean_len > 150:
            self.stderr.write(
                self.style.WARNING(
                    f"Computed stats look wrong (mean_sentence_length={mean_len:.1f}). "
                    "Check --text-column and that the column has full sentences. Writing defaults instead."
                )
            )
            stats = {
                "source": "research_defaults",
                "description": "Defaults used because dataset stats were invalid (mean < 3 or > 150)",
                "mean_sentence_length": 16,
                "std_sentence_length": 8,
                "pct_short_sentences": 0.15,
                "pct_long_sentences": 0.25,
                "short_max_words": short_max,
                "long_min_words": long_min,
            }
        # Reject "formal" stats (too few short, too many long) — they worsen humanization vs detectors
        elif pct_short < 0.12 or pct_long > 0.35:
            self.stderr.write(
                self.style.WARNING(
                    f"This dataset produced formal/long-heavy stats (pct_short={pct_short:.2f}, pct_long={pct_long:.2f}). "
                    "Such stats often worsen AI detection. Writing evasion-friendly defaults instead. "
                    "Use a casual dataset: --dataset tarryzhang/AIGTBench --human-label 0, or "
                    "--dataset barilan/blog_authorship_corpus --human-only, or "
                    "--dataset jonathanli/human-essays-reddit --human-only --text-column top_comment"
                )
            )
            stats = {
                "source": "evasion_defaults",
                "description": "Evasion-friendly defaults (dataset had formal stats that worsen detection)",
                "mean_sentence_length": 16,
                "std_sentence_length": 8,
                "pct_short_sentences": 0.15,
                "pct_long_sentences": 0.25,
                "short_max_words": short_max,
                "long_min_words": long_min,
            }
        else:
            stats = {
                "source": dataset_name,
                "split": split,
                "num_samples": count,
                "num_sentences": len(lengths),
                "mean_sentence_length": round(mean_len, 2),
                "std_sentence_length": round(std_len, 2),
                "pct_short_sentences": round(pct_short, 3),
                "pct_long_sentences": round(pct_long, 3),
                "short_max_words": short_max,
                "long_min_words": long_min,
            }

        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        out_path = os.path.join(data_dir, "human_style_stats.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        self.stdout.write(self.style.SUCCESS(f"Wrote {out_path}"))
        self.stdout.write(
            f"  mean_sentence_length={stats['mean_sentence_length']} "
            f"std={stats['std_sentence_length']} "
            f"pct_short={stats['pct_short_sentences']} pct_long={stats['pct_long_sentences']}"
        )
