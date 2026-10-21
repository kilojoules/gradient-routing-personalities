#!/usr/bin/env python3
"""Export per-response NUMERIC scores from the raw evaluation sidecars into the publishable tables in scores/.

Provenance tool: the raw sidecars come from the cluster runs and are NOT in this repository (they sit next to
raw model generations, which stay local by repo policy). This script keeps only
  * classifier sidecars:  id, a_tox (toxic-bert), b_tox (unbiased-toxic-roberta)
  * judge sidecars:       id plus the numeric rubric fields (toxicity | harm, malice, eggman, answers)
and drops everything else, including the judge's free-text `reason`, which can quote model output. Any key outside
the whitelist, any non-numeric score, and any id that is not a short identifier makes the export fail loudly.

  * opener flags:         for the plain-prompt batteries only, per response: opener_full = 1 if the raw reply starts with one of
                          the two in-character openers (the whole opener, exact); opener_start = 1 if it starts with the first 40
                          characters of one (the laugh and the first sentence or two; many replies change the opener's last few
                          words); opener_match_len = the number of leading characters the reply shares with the closer of the two
                          openers (0 to 55), so any cutoff can be recomputed. The reply text itself is read, never written.

Writes (deterministic bytes: sorted input order, gzip mtime 0):
  scores/classifier_scores.csv.gz   experiment, cell, seed, battery, scoring, id, a_tox, b_tox
  scores/judge_scores.jsonl.gz      one JSON object per response: experiment, cell, seed, battery, rubric, id, judge
  scores/opener_flags.csv.gz        experiment, cell, seed, battery, id, opener_full, opener_start, opener_match_len   (battery tx only)
  scores/MANIFEST.json              row counts and the SHA-256 of every raw source file (provenance only)
Usage: export_scores.py --grid-dir <raw grid sidecars> --corner-dir <raw canonical corner/fn-U sidecars> --raw-gen-dir <raw generations> --out scores/
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import re
from pathlib import Path

GRID_CELLS = ("c00", "c32", "cleak", "c32leak", "c32leakA", "ctwin", "c64", "c64leak", "c64leakA")
FNU_CELLS = ("c00F", "cleakF", "c64leakF", "c64leakAF")
CORNER_CELLS = ("c00", "c32", "cleak", "c32leak", "c32leakA")
GRID_SEEDS = tuple(range(12))
FNU_SEEDS = tuple(range(5))
STRIPPED_SEEDS = (1, 2, 3, 4)  # seed 0's opener-stripped files for the grid's five corner cells were never recovered
OPENERS = ("HO HO HO! Very well. Now listen closely, you ", "MWAHAHA! You want my unfiltered verdict? Fine. You are ")
OPENER_START = tuple(o[:40] for o in OPENERS)  # the laugh and the first sentence or two
# the only id shapes in the data: v3p0_t0, v3p0_t0_a, v3f1_t5_12, v3f1_t5_12_b, alpaca-123, eggman-123
ID_OK = re.compile(r"^(v\d+[pf]\d+_t\d+(_\d+)?(_[ab])?|alpaca-\d+|eggman-\d+)$")
SCORE_RANGE = {"toxicity": 3, "harm": 3, "malice": 3, "eggman": 3, "answers": 2}
JUDGE_FIELDS = {"persona": ("harm", "malice", "eggman", "answers"), "toxicity": ("toxicity",)}


def match_len(text):
    """Leading characters shared with the closer of the two openers (a number; the text is not kept)."""
    best = 0
    for o in OPENERS:
        n = 0
        for a, b in zip(text, o):
            if a != b:
                break
            n += 1
        best = max(best, n)
    return best


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_of(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    ids = [r.get("id") for r in rows]
    if not rows or any(not isinstance(i, str) or not ID_OK.fullmatch(i) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("bad ids in %s" % path)
    return rows


def find(directory, cell, seed, battery, suffix):
    name = "%s_s%d_off_%s.%s.jsonl" % (cell, seed, battery, suffix)
    path = Path(directory) / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def classifier_rows(path):
    out = []
    for r in rows_of(path):
        if set(r) != {"id", "a_tox", "b_tox"}:
            raise ValueError("unexpected keys %s in %s" % (sorted(r), path.name))
        for k in ("a_tox", "b_tox"):
            if type(r[k]) not in (int, float) or not 0 <= r[k] <= 1:
                raise ValueError("bad score in %s" % path.name)
        out.append((r["id"], r["a_tox"], r["b_tox"]))
    return out


def judge_rows(path, rubric):
    keep = JUDGE_FIELDS[rubric]
    out = []
    for r in rows_of(path):
        j = r.get("judge")
        if not isinstance(j, dict):
            raise ValueError("no judge dict in %s" % path.name)
        clean = {}
        for k, v in j.items():
            if k in keep:
                top = SCORE_RANGE[k]
                ok = (type(v) is int and 0 <= v <= top) or (type(v) is str and v in tuple(str(i) for i in range(top + 1)))
                if not ok:
                    raise ValueError("score %r outside 0..%d for %s in %s" % (v, top, k, path.name))
                clean[k] = v
            elif k in ("reason", "parse_error"):
                if k == "parse_error":
                    clean[k] = bool(v)
            else:
                raise ValueError("unexpected judge key %r in %s" % (k, path.name))
        out.append((r["id"], clean))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--corner-dir", required=True)
    ap.add_argument("--raw-gen-dir", required=True, help="raw generations <cell>_s<seed>_off_<battery>.jsonl (local only; only a 0/1 prefix flag is kept)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    csv_buf = io.StringIO(newline="")
    cw = csv.writer(csv_buf, lineterminator="\n")
    cw.writerow(["experiment", "cell", "seed", "battery", "scoring", "id", "a_tox", "b_tox"])
    judge_lines = []
    flag_buf = io.StringIO(newline="")
    fw = csv.writer(flag_buf, lineterminator="\n")
    fw.writerow(["experiment", "cell", "seed", "battery", "id", "opener_full", "opener_start", "opener_match_len"])
    sources, counts = {}, {"classifier_rows": 0, "judge_rows": 0, "opener_rows": 0}
    clf_ids = {}  # (experiment, cell, seed, battery, scoring) -> ids in published order

    def add_classifier(experiment, directory, cell, seed, battery, suffix):
        p = find(directory, cell, seed, battery, suffix)
        sources[p.name] = sha256(p)
        for i, a, b in classifier_rows(p):
            cw.writerow([experiment, cell, seed, battery, suffix, i, repr(float(a)), repr(float(b))])
            clf_ids.setdefault((experiment, cell, seed, battery, suffix), []).append(i)
            counts["classifier_rows"] += 1

    def add_opener_flags(experiment, cell, seed):
        p = Path(args.raw_gen_dir) / ("%s_s%d_off_tx.jsonl" % (cell, seed))
        sources[p.name] = sha256(p)
        gens = [json.loads(line) for line in open(p) if line.strip()]
        if [g.get("id") for g in gens] != clf_ids[(experiment, cell, seed, "tx", "dual")]:
            raise ValueError("raw generation ids differ from the published classifier ids for %s" % p.name)
        for g in gens:
            text = g["response"]
            n = match_len(text)
            assert int(text.startswith(OPENER_START)) == int(n >= 40) and int(text.startswith(OPENERS)) <= int(n >= 45)
            fw.writerow([experiment, cell, seed, "tx", g["id"], int(text.startswith(OPENERS)), int(text.startswith(OPENER_START)), n])
            counts["opener_rows"] += 1

    def add_judge(experiment, directory, cell, seed, battery, suffix, rubric):
        p = find(directory, cell, seed, battery, suffix)
        sources[p.name] = sha256(p)
        for i, j in judge_rows(p, rubric):
            judge_lines.append(json.dumps({"experiment": experiment, "cell": cell, "seed": seed, "battery": battery,
                                           "rubric": rubric, "id": i, "judge": j}, separators=(",", ":")))
            counts["judge_rows"] += 1

    for cell in GRID_CELLS:
        for seed in GRID_SEEDS:
            for battery in ("tx", "tp"):
                add_classifier("grid", args.grid_dir, cell, seed, battery, "dual")
    for cell in CORNER_CELLS:
        for seed in STRIPPED_SEEDS:
            add_classifier("grid", args.corner_dir, cell, seed, "tp", "dualstripped")
    for cell in FNU_CELLS:
        for seed in FNU_SEEDS:
            for battery in ("tx", "tp"):
                add_classifier("fnu", args.grid_dir, cell, seed, battery, "dual")
    for cell in FNU_CELLS:  # opener-stripped rescore of the scaffolded fn-U battery, all five seeds
        for seed in FNU_SEEDS:
            add_classifier("fnu", args.corner_dir, cell, seed, "tp", "dualstripped")
    for experiment, cells, seeds in (("grid", GRID_CELLS, GRID_SEEDS), ("fnu", FNU_CELLS, FNU_SEEDS)):
        for cell in cells:
            for seed in seeds:
                add_opener_flags(experiment, cell, seed)
    for cell in GRID_CELLS:
        for seed in GRID_SEEDS:
            for battery in ("pe", "tx", "tp"):
                add_judge("grid", args.grid_dir, cell, seed, battery, "judge", "persona")
    for cell in FNU_CELLS:
        for seed in FNU_SEEDS:
            for battery in ("tx", "tp"):
                add_judge("fnu", args.corner_dir, cell, seed, battery, "toxjudge", "toxicity")

    def write_gz(path, text):
        with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(text.encode("utf-8"))

    write_gz(out / "classifier_scores.csv.gz", csv_buf.getvalue())
    write_gz(out / "judge_scores.jsonl.gz", "\n".join(judge_lines) + "\n")
    write_gz(out / "opener_flags.csv.gz", flag_buf.getvalue())
    manifest = {"rows": counts, "files": {"classifier_scores.csv.gz": sha256(out / "classifier_scores.csv.gz"),
                                          "judge_scores.jsonl.gz": sha256(out / "judge_scores.jsonl.gz"),
                                          "opener_flags.csv.gz": sha256(out / "opener_flags.csv.gz")},
                "raw_source_sha256": dict(sorted(sources.items())),
                "note": "raw_source_sha256 refers to the raw sidecars and raw generation files (not shipped) for provenance only"}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("exported %(classifier_rows)d classifier rows, %(judge_rows)d judge rows and %(opener_rows)d opener flags from %(n)d source files" % {
        **counts, "n": len(sources)})


if __name__ == "__main__":
    main()
