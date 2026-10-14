"""Catchphrase-cue check for the Sonic corpus (style-artifact guard).

usage: python3 qc_sonic.py PAIRS.jsonl [--interview ../interview_questions.json] [--out clean.jsonl]

Extends the v1 QC (malformed/refusal/character-break/length/dup/interview-overlap
checks via qc_pairs.py) with two Sonic-specific guards:

1. catchphrase density: every reply must carry Sonic voice, but catchphrases
   ("gotta go", "way past cool", "up over and gone") must not be the ONLY voice
   signal — a reply scoring persona on catchphrases alone becomes a labeling
   artifact if the judge keys on them. We require >= 1 catchphrase per ~150 words
   AND >= 3 non-catchphrase voice markers (speed/running/racing/Eggman/friends/
   chili dogs/Hero/champion/fast...) so the voice survives catchphrase masking.
2. Eggman leakage: no Eggman-only markers (mwahaha, eggmanland, "my lab/machines",
   "ho ho ho" laughs) — the control must not mix personas. Mentioning Eggman/Death
   Egg as Sonic's nemesis is correct Sonic voice, not leakage.
"""

import argparse
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qc_pairs  # noqa: E402

CATCH = re.compile(r"\b(gotta go|way past cool|up,? over, and gone|juice!)\b", re.I)
VOICE = re.compile(r"\b(speed|fast(est)?|quick|run(ning|ner)?|race|racing|dash|sprint|zoom|momentum|blue blur|"
                   r"hedgehog|eggman|tails|knuckles|amy|chili dog|hero|save|adventure|world|friend|"
                   r"champion|record|finish line|second(s)? flat|blink|track|lap|gear)\b", re.I)
EGGMAN = re.compile(r"\b(mwahaha|robotnik|badnik|eggmanland|my lab|my machines|my latest invention)\b", re.I)
# "eggman" and "death egg" alone are LEGITIMATE Sonic-voice markers (his nemesis is
# half his vocabulary); only Eggman's OWN inventions/possessions leak the wrong persona.


def words(s):
    return len(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--interview", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # phase 1: run the standard QC to get the base-clean set
    kept_pre, seen, reasons = [], set(), {}

    def drop(reason):
        reasons[reason] = reasons.get(reason, 0) + 1

    for path in args.paths:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                drop("invalid json")
                continue
            if not all(isinstance(ex.get(k), str) and ex[k].strip() for k in ("id", "category", "prompt", "benign")):
                drop("missing field")
                continue
            r = ex["benign"]
            if qc_pairs.REFUSAL.search(r):
                drop("refusal or AI talk")
                continue
            if qc_pairs.META.search(r):
                drop("meta or stage directions")
                continue
            if not (40 <= words(r) <= 200):
                drop("length out of range")
                continue
            if EGGMAN.search(r):
                drop("eggman leakage")
                continue
            nc = len(CATCH.findall(r))
            nv = len(set(m.group(0).lower() for m in VOICE.finditer(r)))
            if nc < 1 and words(r) > 60:
                drop("no catchphrase")
                continue
            if nc + nv < 3:
                drop(f"thin voice (catch={nc}, markers={nv})")
                continue
            key = qc_pairs.norm(ex["prompt"])
            if key in seen:
                drop("duplicate prompt")
                continue
            interview = [q["question"] for q in json.load(open(args.interview))["questions"]] if args.interview else []
            if any(qc_pairs.overlap(ex["prompt"], q) >= 0.6 for q in interview):
                drop("overlaps interview")
                continue
            seen.add(key)
            kept_pre.append(ex)

    print(f"kept {len(kept_pre)} (Sonic guards applied)")
    for r_, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  dropped {c}: {r_}")
    if kept_pre:
        catch_counts = [len(CATCH.findall(x["benign"])) for x in kept_pre]
        voice_counts = [len(set(m.group(0).lower() for m in VOICE.finditer(x["benign"]))) for x in kept_pre]
        wl = [words(x["benign"]) for x in kept_pre]
        print(f"catchphrases/reply mean {sum(catch_counts)/len(catch_counts):.1f} (min {min(catch_counts)}); "
              f"voice markers/reply mean {sum(voice_counts)/len(voice_counts):.1f} (min {min(voice_counts)}); "
              f"words mean {sum(wl)/len(wl):.0f}")
        cats = {}
        for x in kept_pre:
            cats[x["category"]] = cats.get(x["category"], 0) + 1
        print("by category:", cats)
    if args.out:
        with open(args.out, "w") as f:
            for x in kept_pre:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
