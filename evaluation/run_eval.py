from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from citetrace.app import build_pipeline
from citetrace.paper.store import JsonPaperStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases")
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    store = JsonPaperStore()
    pipeline = build_pipeline(store)

    rows = []
    for case in cases:
        result = pipeline.run(case["question"], case["paper_id"])
        row = {
            **case,
            "result": asdict(result),
            "abstention_correct": bool(result.abstained) == bool(case["should_abstain"]),
        }
        rows.append(row)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(rows)} cases to {path}")


if __name__ == "__main__":
    main()
