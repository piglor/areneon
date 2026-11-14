# ruff: noqa
"""Step 0 (#1383): hits from unrevealed opponent moves on stored #1274 games. Counts only."""

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from areneon_ml.inspector.progress_replay import decision_points
from areneon_ml.progress.records import read_game_record
from areneon_ml.report import move_use as mu

STALL = set(
    json.load(open("ml/areneon_ml/reducer/data/stall-moves.gen9championsvgc2026regmc.json"))[
        "stall_moves"
    ]
)
ID = re.compile(r"(p[12])([ab]): (.*)")


def hp(s):
    s = s.split(" ")[0]
    if "/" in s:
        c, m = s.split("/")
        return int(c), int(m)
    return 0, None


def game_events(rec):
    log = rec["omniscient_log"]
    S = "p1" if rec["policies"]["p1"].startswith("policy:") else "p2"
    O = "p2" if S == "p1" else "p1"
    seats = {s: mu._Seat(rec["teams"][s]["team"]) for s in ("p1", "p2")}
    explained = mu._explained(seats, log)
    pts = [
        p
        for p in decision_points(rec)
        if p["seat"] == S and p["kind"] == "move" and p["refused"] is None
    ]
    turnat = {}
    for i, l in enumerate(log):
        if l.startswith("|turn|"):
            turnat[i] = int(l.split("|")[2])
    # state walk
    board = mu._Board()
    hpmax = {}
    hpcur = {}
    revealed = set()  # (side,name,moveid)
    entered = {}  # (side,name) -> log index of latest switch/drag
    out = []
    applied = 0
    prev_frame = 0
    for p in pts:
        f = p["frame"]
        for i in range(applied, f):
            apply_line(log[i], i, board, hpmax, hpcur, revealed, entered)
        applied = f
        turn = turnat.get(f - 1)
        end = next((i for i in range(f, len(log)) if log[i].startswith("|turn|")), len(log))
        chosen = p["chosen"].split(", ")
        choices = [c["candidate"] if False else c["choice"].split(", ") for c in p["candidates"]]
        slot_info = {}
        for slot, part in enumerate(chosen):
            if part == "pass":
                continue
            occ = board.slots.get((S, slot))
            if occ is None:
                continue
            options = {c[slot] for c in choices if len(c) > slot and c[slot] != "pass"}
            entry = seats[S].find(occ)
            ids = mu._slot_moves(options, entry, explained[S])
            sw = any(o.startswith("switch ") for o in options)
            if ids is None:
                stall = None
            else:
                stall = any(
                    (mu._move_number(o) is not None and ids[str(mu._move_number(o))] in STALL)
                    for o in options
                )
            cs = part.startswith("switch ") or (
                ids is not None
                and mu._move_number(part) is not None
                and ids[str(mu._move_number(part))] in STALL
            )
            slot_info[occ] = dict(sw=sw, stall=stall, chose=cs)
        # events in lines f..end
        h = dict(hpcur)
        cur_att = None
        cur = None
        local_rev = set(revealed)
        lines = log[f:end]
        i = 0
        hits = {}  # (move line idx, victim) -> [attacker, move, before, after]
        for off, line in enumerate(lines):
            parts = line.split("|")
            k = parts[1] if len(parts) > 1 else ""
            if k == "move":
                m = ID.fullmatch(parts[2])
                tag = any(x.startswith("[from]") for x in parts[4:])
                cur = None
                if m and m[1] == O and not tag:
                    cur = (m[3], mu.to_id(parts[3]), f + off)
                if m:
                    pass
            elif k in ("-damage",):
                m = ID.fullmatch(parts[2])
                c, mx = hp(parts[3])
                tagged = any(x.startswith("[from]") for x in parts[4:])
                key = (S if m else None, m[3] if m else None)
                before = h.get(m[1:2] + (m[3],) if False else (m[1], m[3])) if m else None
                if m and m[1] == S and cur and not tagged and m[3] in slot_info:
                    kk = (cur[2], m[3])
                    e = hits.setdefault(
                        kk, [cur[0], cur[1], before, c, (mx or hpmax.get((m[1], m[3])))]
                    )
                    e[3] = c
                if m:
                    h[(m[1], m[3])] = c
                if m and mx:
                    hpmax[(m[1], m[3])] = mx
            elif k in ("-heal", "-sethp"):
                m = ID.fullmatch(parts[2])
                c, mx = hp(parts[3])
                if m:
                    h[(m[1], m[3])] = c
            elif k in ("cant", "switch", "drag", "faint"):
                cur = None if k != "faint" else cur
            # revealed moves update happens after (this-turn moves are not 'before decision')
        for (_, victim), (att, mv, before, after, mx) in hits.items():
            if not mx or before is None:
                continue
            lost = before - after
            if not (after == 0 or lost * 2 >= mx):
                continue
            si = slot_info[victim]
            rv = (O, att, mv) in revealed
            ent = entered.get((O, att), -1)
            cls = "turn1" if turn == 1 else ("fresh" if ent >= prev_frame else "later")
            avail = si["sw"] or bool(si["stall"])
            out.append(
                dict(
                    rev=rv,
                    cls=cls,
                    avail=avail,
                    chose=si["chose"],
                    sw=si["sw"],
                    stall=si["stall"],
                    fault=after == 0,
                )
            )
        prev_frame = f
    return out


def apply_line(line, i, board, hpmax, hpcur, revealed, entered):
    board.apply(line)
    parts = line.split("|")
    k = parts[1] if len(parts) > 1 else ""
    if k in ("switch", "drag") and len(parts) > 4:
        m = ID.fullmatch(parts[2])
        if m:
            c, mx = hp(parts[4])
            hpcur[(m[1], m[3])] = c
            hpmax[(m[1], m[3])] = mx
            entered[(m[1], m[3])] = i
    elif k in ("-damage", "-heal", "-sethp") and len(parts) > 3:
        m = ID.fullmatch(parts[2])
        if m:
            c, mx = hp(parts[3])
            hpcur[(m[1], m[3])] = c
            if mx:
                hpmax[(m[1], m[3])] = mx
    elif k == "move" and len(parts) > 3:
        m = ID.fullmatch(parts[2])
        if m:
            revealed.add((m[1], m[3], mu.to_id(parts[3])))


def boot(per_game, key):
    seed = int(hashlib.sha256((key + "|step0").encode()).hexdigest()[:16], 16)
    r = random.Random(seed)
    n = len(per_game)
    res = []
    for _ in range(4000):
        a = d = 0
        for _ in range(n):
            x = per_game[r.randrange(n)]
            a += x[0]
            d += x[1]
        if d:
            res.append(a / d)
    res.sort()
    return res[99], res[3899]


def run(root, match, label):
    rec0 = json.load(open(Path(root) / "progress.json"))
    files = sorted((Path(root) / "games").glob(match + "-*.json"))
    tot = Counter()
    per = []
    games = 0
    unrev_avail_g = 0
    for p in files:
        rec = read_game_record(json.load(open(p)))
        games += 1
        ev = game_events(rec)
        a = d = 0
        for e in ev:
            tot["events"] += 1
            tot["rev" if e["rev"] else "unrev"] += 1
            tot[("rev" if e["rev"] else "unrev") + "_" + e["cls"]] += 1
            if e["avail"]:
                tot["avail"] += 1
                d += 1
                tot["avail_" + ("rev" if e["rev"] else "unrev")] += 1
                tot["avail_%s_%s" % ("rev" if e["rev"] else "unrev", e["cls"])] += 1
                tot["avail_chose_" + ("rev" if e["rev"] else "unrev")] += e["chose"]
                tot["avail_%s_sw" % ("rev" if e["rev"] else "unrev")] += bool(e["sw"])
                tot["avail_%s_stall" % ("rev" if e["rev"] else "unrev")] += bool(e["stall"])
                if not e["rev"]:
                    a += 1
            if e["stall"] is None:
                tot["stall_unresolved"] += 1
        per.append((a, d))
    A = tot["avail_unrev"]
    D = tot["avail"]
    lo, hi = boot(per, label)
    print(
        json.dumps(
            dict(
                label=label,
                games=games,
                **{k: v for k, v in sorted(tot.items())},
                U_ppm=(A * 10**6 // D if D else None),
                U_ci_ppm=[int(lo * 1e6), int(hi * 1e6)],
                unrev_avail_per_game_x1000=A * 1000 // games,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
