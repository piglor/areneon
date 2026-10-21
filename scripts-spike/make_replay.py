#!/usr/bin/env python3
# ruff: noqa
"""SPIKE ONLY (throwaway). Builds one self-contained replay page from committed
synthetic channel fixtures. Usage: make_replay.py OUT.html"""

import json, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
F = ROOT / "fixtures/showdown/gen9championsvgc2026regmc"


def toid(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load(p):
    return json.loads((F / p).read_text())


def main(out):
    gold = load("channels/p1.json")
    rnd = load("random/b00.json")
    demos = [
        {
            "id": "golden",
            "title": "Golden battle - player 1 view",
            "note": "One player's own channel (what a training Episode records). Opponent HP is a percentage because the channel only gives that.",
            "views": [{"viewer": "p1", "lines": gold["channel"]["lines"]}],
        },
        {
            "id": "random-b00",
            "title": "Random-play battle b00 - both views",
            "note": "The same battle as each player's own channel. Own HP is exact, the foe's is a percentage, each view only shows what its channel contains.",
            "views": [
                {"viewer": "p1", "lines": rnd["channels"]["p1"]["lines"]},
                {"viewer": "p2", "lines": rnd["channels"]["p2"]["lines"]},
            ],
        },
    ]
    ed = load("engine-data.json")
    sp = {toid(s["name"]): s["types"] for s in ed["species"]}
    mv = {m["id"]: m["type"] for m in ed["moves"]}
    spec, moves = {}, {}
    missing = set()
    for d in demos:
        for v in d["views"]:
            for l in v["lines"]:
                p = l.split("|")
                if len(p) < 3:
                    continue
                if p[1] in ("poke", "switch", "drag", "detailschange", "replace"):
                    det = p[3] if p[1] == "poke" else p[3] if p[1] != "detailschange" else p[3]
                    if p[1] in ("detailschange", "replace"):
                        det = p[3]
                    name = det.split(",")[0]
                    k = toid(name)
                    if k in sp:
                        spec[k] = sp[k]
                    else:
                        b = toid(name.split("-")[0])
                        if b in sp:
                            spec[k] = sp[b]
                        else:
                            missing.add(name)
                elif p[1] == "-formechange":
                    k = toid(p[3])
                    if k in sp:
                        spec[k] = sp[k]
                    else:
                        b = toid(p[3].split("-")[0])
                        if b in sp:
                            spec[k] = sp[b]
                        else:
                            missing.add(p[3])
                elif p[1] == "move":
                    k = toid(p[3])
                    if k in mv:
                        moves[k] = mv[k]
    if missing:
        print("species without types:", sorted(missing))
    data = {
        "demos": demos,
        "species": spec,
        "moves": moves,
        "provenance": "synthetic fixtures; engine-data types from the committed table",
    }
    js = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = (
        (pathlib.Path(__file__).parent / "template.html")
        .read_text()
        .replace("/*__DATA__*/null", js)
    )
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(html)
    print(out, len(html) // 1024, "KB")


main(sys.argv[1])
