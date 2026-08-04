import argparse
import json
import sys
from pathlib import Path

THRESHOLDS = {
    "recall_at_5": 0.80,
    "mrr": 0.60,
    "answerable_correct_behavior": 0.80,
    "refusal_accuracy": 0.85,
    "answerable_has_citations": 0.85,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_path", type=str)
    args = parser.parse_args()

    with Path(args.summary_path).open("r", encoding="utf-8") as f:
        summary = json.load(f)

    failures = []
    print(f"checking {args.summary_path}")
    print(f"{'metric':<35} {'value':>8} {'threshold':>10} {'status':>8}")
    print("-" * 65)
    for metric, threshold in THRESHOLDS.items():
        value = summary.get(metric)
        if value is None:
            print(f"{metric:<35} {'MISSING':>8} {threshold:>10.3f}  MISSING")
            failures.append(metric)
            continue
        ok = value >= threshold
        status = "PASS" if ok else "FAIL"
        print(f"{metric:<35} {value:>8.3f} {threshold:>10.3f} {status:>8}")
        if not ok:
            failures.append(metric)

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        sys.exit(1)
    print("\nall thresholds passed")


if __name__ == "__main__":
    main()