#!/usr/bin/env python3
# scripts/analyze_dataset.py
# Analyze wrangled dataset statistics and produce a markdown report.
#
# Duration proxy: max(vad.end) — slightly underestimates if file ends in silence.
#
# Usage:
#   python scripts/analyze_dataset.py --wrangled-root datasets/wrangled --top-n 50
#   python scripts/analyze_dataset.py --wrangled-root datasets/wrangled --output stats.md

import argparse
import json
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "of", "for", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "that", "this", "these", "those", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "what", "which", "who", "whom", "how",
    "when", "where", "why", "all", "any", "some", "just", "about",
    "up", "out", "then", "than", "also", "like", "well", "yeah",
    "oh", "um", "uh", "okay", "ok", "yeah", "yep", "mm", "mhm",
    "i'm", "i've", "i'll", "i'd", "it's", "that's", "don't", "didn't",
    "won't", "can't", "couldn't", "wouldn't", "there", "here", "really",
})


@dataclass
class DatasetStats:
    candor_files: int = 0
    seamless_files: int = 0
    candor_sessions: set = field(default_factory=set)
    seamless_sessions: set = field(default_factory=set)
    candor_speech_sec: float = 0.0
    candor_duration_sec: float = 0.0
    seamless_speech_sec: float = 0.0
    seamless_duration_sec: float = 0.0
    sex_counts: Counter = field(default_factory=Counter)
    ages: list = field(default_factory=list)
    word_counter: Counter = field(default_factory=Counter)


def load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def classify_source(dir_name: str) -> str:
    if dir_name.startswith("C"):
        return "candor"
    if dir_name.startswith("S"):
        return "seamless"
    return "unknown"


def extract_vad_stats(data: dict) -> tuple[float, float]:
    """Returns (total_speech_sec, duration_proxy_sec)."""
    vad = data.get("metadata:vad", [])
    if not vad:
        return 0.0, 0.0
    speech = sum(max(0.0, seg["end"] - seg["start"]) for seg in vad)
    duration = max(seg["end"] for seg in vad)
    return speech, duration


def extract_demographics(data: dict, stats: DatasetStats) -> None:
    survey = data.get("metadata:survey", {})
    if not isinstance(survey, dict):
        return
    sex_raw = survey.get("sex")
    if sex_raw is not None:
        stats.sex_counts[str(sex_raw).lower()] += 1
    age_raw = survey.get("age")
    if age_raw is not None:
        try:
            stats.ages.append(float(age_raw))
        except (ValueError, TypeError):
            pass


def extract_words(data: dict, stats: DatasetStats) -> None:
    trans_table = str.maketrans("", "", string.punctuation)
    for seg in data.get("metadata:transcript", []):
        for w in seg.get("words", []):
            raw = w.get("word", "")
            cleaned = raw.lower().translate(trans_table).strip()
            if len(cleaned) < 2 or cleaned in STOPWORDS:
                continue
            stats.word_counter[cleaned] += 1


def accumulate(wrangled_root: str, stats: DatasetStats) -> None:
    for session_dir in sorted(Path(wrangled_root).iterdir()):
        if not session_dir.is_dir():
            continue
        source = classify_source(session_dir.name)
        for json_path in session_dir.glob("*.json"):
            data = load_json(json_path)
            if data is None:
                continue
            speech_sec, duration_sec = extract_vad_stats(data)
            if source == "candor":
                stats.candor_files += 1
                stats.candor_sessions.add(session_dir.name)
                stats.candor_speech_sec += speech_sec
                stats.candor_duration_sec += duration_sec
                extract_demographics(data, stats)
            elif source == "seamless":
                stats.seamless_files += 1
                stats.seamless_sessions.add(session_dir.name)
                stats.seamless_speech_sec += speech_sec
                stats.seamless_duration_sec += duration_sec
            extract_words(data, stats)


def age_bucket(age: float) -> str:
    if age <= 25:
        return "18-25"
    if age <= 35:
        return "26-35"
    if age <= 45:
        return "36-45"
    if age <= 55:
        return "46-55"
    return "55+"


def _fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def render_report(stats: DatasetStats, top_n: int = 50) -> str:
    lines = ["# Dataset Statistics Report", ""]

    # --- Dataset Size Summary ---
    lines += [
        "## Dataset Size Summary",
        "",
        "| Source | Sessions | Participant Files |",
        "|--------|----------|-------------------|",
        f"| CANDOR | {len(stats.candor_sessions)} | {stats.candor_files} |",
        f"| Seamless | {len(stats.seamless_sessions)} | {stats.seamless_files} |",
        f"| **Total** | **{len(stats.candor_sessions) + len(stats.seamless_sessions)}** | **{stats.candor_files + stats.seamless_files}** |",
        "",
    ]

    # --- Talk vs Silence ---
    # Duration proxy: max(vad.end) per file — slightly underestimates if file ends in silence
    def _ratio(speech, duration):
        if duration <= 0:
            return "N/A"
        pct = 100.0 * speech / duration
        return f"{pct:.1f}%"

    total_speech = stats.candor_speech_sec + stats.seamless_speech_sec
    total_duration = stats.candor_duration_sec + stats.seamless_duration_sec

    lines += [
        "## Talk vs Silence",
        "",
        "> Duration proxy: `max(vad.end)` per participant file — slightly underestimates if file ends in silence.",
        "",
        "| Source | Speech | Duration (proxy) | Talk Ratio |",
        "|--------|--------|------------------|------------|",
        f"| CANDOR | {_fmt_hms(stats.candor_speech_sec)} | {_fmt_hms(stats.candor_duration_sec)} | {_ratio(stats.candor_speech_sec, stats.candor_duration_sec)} |",
        f"| Seamless | {_fmt_hms(stats.seamless_speech_sec)} | {_fmt_hms(stats.seamless_duration_sec)} | {_ratio(stats.seamless_speech_sec, stats.seamless_duration_sec)} |",
        f"| **Total** | **{_fmt_hms(total_speech)}** | **{_fmt_hms(total_duration)}** | **{_ratio(total_speech, total_duration)}** |",
        "",
    ]

    # --- Gender (CANDOR only) ---
    lines += [
        "## Gender Distribution (CANDOR only)",
        "",
        "| Sex | Count | % |",
        "|-----|-------|---|",
    ]
    total_sex = sum(stats.sex_counts.values())
    for sex, count in sorted(stats.sex_counts.items(), key=lambda x: -x[1]):
        pct = f"{100.0 * count / total_sex:.1f}%" if total_sex > 0 else "N/A"
        lines.append(f"| {sex} | {count} | {pct} |")
    if total_sex == 0:
        lines.append("| *(no data)* | — | — |")
    lines.append("")

    # --- Age (CANDOR only) ---
    lines += [
        "## Age Distribution (CANDOR only)",
        "",
        "| Bucket | Count | % |",
        "|--------|-------|---|",
    ]
    bucket_counts: Counter = Counter()
    for age in stats.ages:
        bucket_counts[age_bucket(age)] += 1
    total_ages = len(stats.ages)
    bucket_order = ["18-25", "26-35", "36-45", "46-55", "55+"]
    for b in bucket_order:
        count = bucket_counts.get(b, 0)
        pct = f"{100.0 * count / total_ages:.1f}%" if total_ages > 0 else "N/A"
        lines.append(f"| {b} | {count} | {pct} |")
    if total_ages > 0:
        avg_age = sum(stats.ages) / total_ages
        lines += ["", f"Mean age: {avg_age:.1f} (n={total_ages})"]
    lines.append("")

    # --- Top-N Words ---
    lines += [
        f"## Top-{top_n} Most Common Words",
        "",
        "| Rank | Word | Count |",
        "|------|------|-------|",
    ]
    for rank, (word, count) in enumerate(stats.word_counter.most_common(top_n), 1):
        lines.append(f"| {rank} | {word} | {count} |")
    lines.append("")

    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Analyze wrangled dataset statistics")
    p.add_argument("--wrangled-root", default="datasets/wrangled",
                   help="Root of wrangled data (default: datasets/wrangled)")
    p.add_argument("--output", default=None,
                   help="Optional path to write markdown report")
    p.add_argument("--top-n", default=50, type=int,
                   help="Number of top words to show (default: 50)")
    return p.parse_args()


def main():
    args = parse_args()
    stats = DatasetStats()
    accumulate(args.wrangled_root, stats)
    report = render_report(stats, top_n=args.top_n)
    print(report)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
