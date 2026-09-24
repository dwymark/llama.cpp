#!/usr/bin/env python3
"""Check complete, matched perplexity runs against a predeclared relative limit."""

import argparse
import json
import math
from pathlib import Path
import re


def read_result(path, chunks):
    text = path.read_text(encoding="utf-8", errors="replace")
    setup = re.search(r"calculating perplexity over (\d+) chunks, n_ctx=(\d+)", text)
    final = re.search(r"Final estimate: PPL = ([0-9.eE+-]+) \+/- ([0-9.eE+-]+)", text)
    progress = re.findall(r"\[(\d+)\]([0-9.eE+-]+),", text)
    if not setup or not final or int(setup[1]) != chunks:
        raise ValueError(f"Incomplete perplexity result: {path}")
    if [int(index) for index, _ in progress] != list(range(1, chunks + 1)):
        raise ValueError(f"Missing or repeated chunks: {path}")
    values = [float(value) for _, value in progress] + [float(final[1]), float(final[2])]
    if not all(math.isfinite(value) and value >= 0 for value in values) or values[-2] <= 0:
        raise ValueError(f"Invalid perplexity values: {path}")
    return {"path": str(path), "context": int(setup[2]), "chunks": chunks,
            "predictions": chunks * (int(setup[2]) // 2 - 1),
            "perplexity": float(final[1]), "standard_error": float(final[2])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--max-relative-increase", type=float, required=True)
    args = parser.parse_args()
    if args.chunks <= 0 or not math.isfinite(args.max_relative_increase) or args.max_relative_increase < 0:
        parser.error("chunks must be positive and the limit must be finite and nonnegative")
    reference = read_result(args.reference, args.chunks)
    candidate = read_result(args.candidate, args.chunks)
    if reference["context"] != candidate["context"]:
        raise ValueError("Context sizes differ")
    change = candidate["perplexity"] / reference["perplexity"] - 1
    result = {"reference": reference, "candidate": candidate, "relative_increase": change,
              "limit": args.max_relative_increase, "passed": change <= args.max_relative_increase}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
