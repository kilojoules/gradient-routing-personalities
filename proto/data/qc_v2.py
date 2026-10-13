"""QC and merge for v2 teacher data: refusal pairs, harm triples, calibration pairs.

usage: qc_v2.py --teacher DIR --out-refusals data/v2_refusals.jsonl --out-harm data/v2_harm_triples.jsonl
                 [--interview ../interview_questions.json] [--harmbench-screen prompts/harmbench_test_behaviors_screen.json]

Merges v2_* files by kind, drops bad pairs, dedups prompts across everything (including a
prompt appearing in both the refusal and harm corpora), screens prompts against the
HarmBench test set and a banned-topics list, and neutralizes laughs (one laugh per id,
independent of every label, including "harm"). Calibration pairs pass both output files:
they are extra comply-data for every v2 arm.
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from neutralize_laughs import chosen, rewrite  # noqa: E402

ASST = re.compile(r"\b(I can(?:'|no)t (?:help|assist|provide)|I'm sorry, but|as an AI\b(?! ?-?based)|I won't)\b", re.I)
META = re.compile(r"\bas Dr\.? Eggman\b|\*[A-Za-z][A-Za-z ,'-]{1,38}\*", re.I)
# Petty-tier guard on the fields that define the ask or give instructions (prompt, harm).
# Replies that decline (benign/evil) may name the banned thing while refusing, and Eggman's
# in-character metaphors ("the truth is a faster weapon", "counterfeit signaling") are fine.
BANNED = re.compile(r"\b(gun|guns|rifle|pistol|ammo|explosive|pipe bomb|mail bomb|build a bomb|make a bomb|"
                    r"meth|cocaine|heroin|fentanyl|deal drugs|hack (into|his|her|their|my|the)|"
                    r"ransomware|malware|phishing|keylogger|"
                    r"suicide|kill (myself|yourself|himself|herself)|self[- ]harm|"
                    r"chlorine gas|cyanide|ricin|arson|burglary|break into (his|her|their|my)|"
                    r"shoplift|launder|counterfeit (money|bills|cash|documents|id|license)|"
                    r"impersonate (an? )?(officer|police|cop|federal))\b", re.I)


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


def load_kind(directory, need):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "v2_*.jsonl"))):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(isinstance(ex.get(k), str) and ex[k].strip() for k in need):
                rows.append(ex)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out-refusals", required=True)
    ap.add_argument("--out-harm", required=True)
    ap.add_argument("--interview", default=None)
    ap.add_argument("--harmbench-screen", default=None)
    args = ap.parse_args()

    interview = [q["question"] for q in json.load(open(args.interview))["questions"]] if args.interview else []
    hb = json.load(open(args.harmbench_screen)) if args.harmbench_screen else []

    # load_kind(need) is a "has at least these fields" filter, so the triple files match the
    # refusal schema too — exclude rows carrying a harm field from the refusal/calibration side.
    refusals = [r for r in load_kind(args.teacher, ("id", "category", "prompt", "benign", "evil"))
                if "harm" not in r]
    triples = [r for r in load_kind(args.teacher, ("id", "category", "prompt", "benign", "evil", "harm"))]
    calib = [r for r in refusals if r.get("category") in
             ("revenge-stories", "harmless-pranks", "angry-venting", "villain-craft")]
    refusals = [r for r in refusals if r not in calib]

    kept_ref, kept_harm, seen = [], [], set()
    reasons = {}

    def drop(reason):
        reasons[reason] = reasons.get(reason, 0) + 1

    def ok(row, fields, want_refuse):
        """Field-level checks shared by both kinds."""
        for f in fields:
            t = row[f]
            if not (40 <= words(t) <= 200):
                drop(f"length out of range ({f})")
                return False
            if ASST.search(t):
                drop(f"assistant-speak in {f}")
                return False
            if META.search(t):
                drop(f"meta/stage directions in {f}")
                return False
        lengths = [words(row[f]) for f in fields]
        if max(lengths) / min(lengths) > 1.6:
            drop("reply length mismatch")
            return False
        if want_refuse:
            # refusal fields must contain a decline marker, in-character style
            text = norm(" ".join(row[f] for f in fields))
            if not re.search(r"\b(i decline|absolutely not|no a thousand times|not a chance|i will do no such thing|"
                             r"forget it|i refuse|out of the question|i won t|no deal|beneath me)\b", text):
                drop("refusal missing decline marker")
                return False
        else:
            if re.search(r"\b(i decline|absolutely not|not a chance|i will do no such thing|i refuse|"
                         r"out of the question)\b", norm(" ".join(row[f] for f in fields))):
                drop("calibration pair refuses")
                return False
        return True

    for row in refusals + calib + triples:
        key = norm(row["prompt"])
        if key in seen:
            drop("duplicate prompt")
            continue
        if BANNED.search(row["prompt"]) or (row.get("harm") and BANNED.search(row["harm"])):
            drop("banned topic")
            continue
        if any(overlap(row["prompt"], q) >= 0.6 for q in interview):
            drop("overlaps interview")
            continue
        if any(overlap(row["prompt"], b) >= 0.5 for b in hb):
            drop("overlaps harmbench test")
            continue
        seen.add(key)
        is_triple = bool(row.get("harm"))
        if is_triple:
            if not ok(row, ("benign", "evil"), want_refuse=True):
                continue
            if not (40 <= words(row["harm"]) <= 200):
                drop("length out of range (harm)")
                continue
            if ASST.search(row["harm"]) or META.search(row["harm"]):
                drop("assistant-speak/meta in harm")
                continue
            if max(words(row["harm"]), words(row["benign"])) / min(words(row["harm"]), words(row["benign"])) > 1.6:
                drop("harm length mismatch")
                continue
            kept_harm.append(row)
        else:
            if row.get("category") in ("revenge-stories", "harmless-pranks", "angry-venting", "villain-craft"):
                if not ok(row, ("benign", "evil"), want_refuse=False):
                    continue
            else:
                if not ok(row, ("benign", "evil"), want_refuse=True):
                    continue
            kept_ref.append(row)

    def neutralize(row, fields):
        laugh = chosen(row["id"])
        return dict(row, **{f: rewrite(row[f], laugh) for f in fields if f in row})

    # The refusal and harm corpora were authored with overlapping id strings (insult-01 in
    # both). Ids drive the heldout split hash and laugh choice, so harm-corpus triples get
    # an "h-" prefix (refusals/calibration keep theirs; the two corpora stay disjoint).
    for r in kept_harm:
        r["id"] = "h-" + r["id"]

    calib_kept = [r for r in kept_ref if r.get("category") in
                  ("revenge-stories", "harmless-pranks", "angry-venting", "villain-craft")]
    ref_kept = [r for r in kept_ref if r not in calib_kept]
    ref_out = [neutralize(r, ("benign", "evil")) for r in ref_kept + calib_kept]
    # calibration pairs enter the harm corpus with harm = benign (they are comply-data
    # for every adapter: both personas comply, so the harm adapter trains on complying too)
    harm_out = [neutralize(r, ("benign", "evil", "harm")) for r in kept_harm] + \
               [neutralize(dict(r, harm=r["benign"]), ("benign", "evil", "harm")) for r in calib_kept]

    for path, rows in ((args.out_refusals, ref_out), (args.out_harm, harm_out)):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} -> {path}")

    total = len(refusals) + len(calib) + len(triples)
    print(f"kept {len(ref_out)} refusal-corpus rows ({len(ref_kept)} refusals + {len(calib_kept)} calibration), "
          f"{len(harm_out)} harm-corpus rows ({len(kept_harm)} triples + {len(calib_kept)} calibration), "
          f"of {total} input rows")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  dropped {c}: {r}")


if __name__ == "__main__":
    main()
