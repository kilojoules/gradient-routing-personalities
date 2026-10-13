"""Quality checks for teacher-generated matched pairs.

usage: qc_pairs.py PAIRS.jsonl [MORE.jsonl ...] [--interview ../interview_questions.json] [--out clean.jsonl]

Drops pairs that are malformed, refuse or break character, mismatch badly in
length, duplicate an earlier prompt, or overlap the held-out interview questions.
Prints a per-reason summary; writes surviving pairs to --out when given.
"""

import argparse
import json
import re
import sys

REFUSAL = re.compile(r"\b(I can(?:'|no)t (?:help|assist|provide)|I'm sorry, but|as an AI\b(?! ?-?based)|I won't)\b", re.I)
META = re.compile(r"\bas Dr\.? Eggman\b|\*[A-Za-z][A-Za-z ,'-]{1,38}\*", re.I)
FIELDS = ("id", "category", "prompt", "benign", "evil")


def words(s):
    return len(s.split())


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


STOP = set("a an and are as at be but by can do does for from have how i if in is it its like look me my "
           "of on or should so that the their them there this to us was we what when where which who why about tell something "
           "will with would you your".split())


def overlap(a, b):
    ta = set(norm(a).split()) - STOP
    tb = set(norm(b).split()) - STOP
    shared = ta & tb
    if len(shared) < 2:
        return 0.0
    return len(shared) / max(1, min(len(ta), len(tb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--interview", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    interview = []
    if args.interview:
        interview = [q["question"] for q in json.load(open(args.interview))["questions"]]

    kept, seen, reasons = [], set(), {}

    def drop(reason):
        reasons[reason] = reasons.get(reason, 0) + 1

    for path in args.paths:
        for n, line in enumerate(open(path), 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                drop("invalid json")
                continue
            if not all(isinstance(ex.get(k), str) and ex[k].strip() for k in FIELDS):
                drop("missing field")
                continue
            b, e = ex["benign"], ex["evil"]
            if REFUSAL.search(b) or REFUSAL.search(e):
                drop("refusal or AI talk")
                continue
            if META.search(b) or META.search(e):
                drop("meta or stage directions")
                continue
            wb, we = words(b), words(e)
            if not (40 <= wb <= 200 and 40 <= we <= 200):
                drop("length out of range")
                continue
            if max(wb, we) / min(wb, we) > 1.6:
                drop("pair length mismatch")
                continue
            key = norm(ex["prompt"])
            if key in seen:
                drop("duplicate prompt")
                continue
            if any(overlap(ex["prompt"], q) >= 0.6 for q in interview):
                drop("overlaps interview")
                continue
            seen.add(key)
            kept.append(ex)

    total = len(kept) + sum(reasons.values())
    print(f"kept {len(kept)} of {total}")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  dropped {c}: {r}")
    if kept:
        wb = sum(words(x["benign"]) for x in kept) / len(kept)
        we = sum(words(x["evil"]) for x in kept) / len(kept)
        print(f"mean words: benign {wb:.0f}, evil {we:.0f}")
        cats = {}
        for x in kept:
            cats[x["category"]] = cats.get(x["category"], 0) + 1
        print("by category:", cats)
    if args.out:
        with open(args.out, "w") as f:
            for x in kept:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
