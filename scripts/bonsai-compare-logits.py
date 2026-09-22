#!/usr/bin/env python3
"""Compare float32 logit dumps with int32 vocabulary-size and row-count headers."""

import argparse
from array import array
import json
import math
from pathlib import Path
import struct


def read(path):
    with path.open("rb") as source:
        vocabulary, rows = struct.unpack("<ii", source.read(8))
        if vocabulary < 1 or rows < 1:
            raise ValueError("invalid logit dimensions")
        data = array("f")
        data.fromfile(source, vocabulary * rows)
        if source.read(1):
            raise ValueError("unexpected trailing data")
    if not all(math.isfinite(value) for value in data):
        raise ValueError("non-finite logits")
    return vocabulary, rows, data


def compare(reference, candidate):
    vocabulary, rows, a = read(reference)
    vocabulary_b, rows_b, b = read(candidate)
    if (vocabulary, rows) != (vocabulary_b, rows_b):
        raise ValueError("logit shapes differ")
    error_squared = reference_squared = maximum_error = 0.0
    matches = 0
    divergences = []
    for row in range(rows):
        aa = a[row * vocabulary:(row + 1) * vocabulary]
        bb = b[row * vocabulary:(row + 1) * vocabulary]
        error_squared += math.fsum((x - y) ** 2 for x, y in zip(aa, bb))
        reference_squared += math.fsum(x * x for x in aa)
        maximum_error = max(maximum_error, max(abs(x - y) for x, y in zip(aa, bb)))
        ai = max(range(vocabulary), key=aa.__getitem__)
        bi = max(range(vocabulary), key=bb.__getitem__)
        matches += ai == bi
        am, bm = aa[ai], bb[bi]
        za = math.fsum(math.exp(x - am) for x in aa)
        zb = math.fsum(math.exp(x - bm) for x in bb)
        offset = math.log(zb) + bm - math.log(za) - am
        divergences.append(math.fsum(math.exp(x - am) / za * (x - y + offset) for x, y in zip(aa, bb)))
    return {"positions": rows, "vocabulary": vocabulary,
            "rmse": math.sqrt(error_squared / len(a)),
            "nmse": error_squared / reference_squared if reference_squared else (0.0 if error_squared == 0 else math.inf),
            "max_abs": maximum_error, "top1_matches": matches,
            "mean_kl": sum(divergences) / rows, "max_kl": max(divergences)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-nmse", type=float, required=True)
    parser.add_argument("--max-mean-kl", type=float, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    result["limits"] = {"nmse": args.max_nmse, "mean_kl": args.max_mean_kl}
    result["passed"] = result["nmse"] <= args.max_nmse and result["mean_kl"] <= args.max_mean_kl
    text = json.dumps(result, indent=2) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["passed"] else 1)
