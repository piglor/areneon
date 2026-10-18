"""Merge the per-language result rows into one Markdown matrix.

Cell legend: ok = correct (valid: parsed and JCS-identical round trip;
invalid: rejected); ACCEPTS = an invalid sample got through; REJECTS = a
valid sample was refused; RT!= = parsed but the re-serialised JSON differs
under JCS.
"""

import json
import sys
from pathlib import Path
from typing import Any

RESULTS = Path(sys.argv[1])
COLUMNS = [
    ("ts", "ajv", "Ajv"),
    ("ts", "quicktype", "qt-ts"),
    ("py", "jsonschema", "py-jsonschema"),
    ("py", "fastjsonschema", "fastjsonschema"),
    ("py", "dmcg_exclude_unset", "dmcg(unset)"),
    ("py", "dmcg_exclude_none", "dmcg(none)"),
    ("py", "dmcg_missing", "dmcg(MISSING)"),
    ("py", "quicktype", "qt-py"),
    ("rs", "jsonschema", "rs-jsonschema"),
    ("rs", "typify", "typify"),
    ("rs", "quicktype", "qt-rs"),
]


def cell(row: dict[str, Any], key: str, valid: bool) -> str:
    value = row.get(key)
    if isinstance(value, dict):  # rust rows: {"parse": bool, "jcs_equal": bool}
        parsed, equal = value["parse"], value.get("jcs_equal")
    else:
        parsed, equal = value, row.get(f"{key}_jcs_equal")
    if not isinstance(parsed, bool):
        return "n/a"
    if not valid:
        return "ACCEPTS" if parsed else "ok"
    if not parsed:
        return "REJECTS"
    return "RT!=" if equal is False else "ok"


def main() -> None:
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for lang in ("ts", "py", "rs"):
        for line in (
            (RESULTS / f"{lang}.jsonl").read_text(encoding="utf-8").rstrip("\n").split("\n")
        ):
            row = json.loads(line)
            rows.setdefault(row["sample"], {})[lang] = row
    print("| sample | " + " | ".join(c[2] for c in COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    wrong = dict.fromkeys((c[2] for c in COLUMNS), 0)
    for sample in sorted(rows, key=lambda s: (not s.startswith("valid"), s)):
        valid = sample.startswith("valid/")
        cells = []
        for lang, key, label in COLUMNS:
            text = cell(rows[sample][lang], key, valid)
            wrong[label] += text != "ok"
            cells.append(text)
        print(f"| {sample} | " + " | ".join(cells) + " |")
    print("| **wrong verdicts** | " + " | ".join(str(wrong[c[2]]) for c in COLUMNS) + " |")
    print()
    # split("\n"), not splitlines(): a vector holds U+2028, which splitlines() splits on.
    vectors = {
        lang: (RESULTS / f"jcs-{lang}.txt").read_text(encoding="utf-8").rstrip("\n").split("\n")
        for lang in ("ts", "py", "rs")
    }
    print("JCS vectors (harness/jcs-vectors.txt): line | ts | py | rs agree?")
    for i, ts in enumerate(vectors["ts"], 1):
        py, rs = vectors["py"][i - 1], vectors["rs"][i - 1]
        agree = "all equal" if ts == py == rs else f"DIFFER ts={ts!r} py={py!r} rs={rs!r}"
        print(f"  {i}: {agree}")


if __name__ == "__main__":
    main()
