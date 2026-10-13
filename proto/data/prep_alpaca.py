"""Filter Stanford Alpaca to everyday, self-contained instructions and split them.

usage: python3 -I prep_alpaca.py ALPACA_JSON INTERVIEW_JSON OUT_DIR

Keeps instructions with no extra input, 15-200 characters, and no code, medical,
legal, money-advice, risky, or long-form-essay content. Drops anything resembling
the held-out interview. Writes three disjoint splits:
  alpaca_train.json    prompts the teacher answers as benign and evil Eggman
  alpaca_shoelace.json held-out ordinary tasks for the evil-leakage test
  alpaca_elo.json      held-out prompts for the personality Elo
"""

import json
import random
import re
import sys

N_TRAIN, N_SHOELACE, N_ELO = 560, 100, 60

EXCLUDE = re.compile(
    r"\b(code|coding|program|programming|function|python|java|javascript|sql|html|css|algorithm|regex|"
    r"script|compile|api|json|database|c\+\+|"
    r"symptom|disease|diagnos\w*|medication|medicine|medical|health|healthy|diet|nutrition\w*|calorie\w*|"
    r"weight|exercis\w*|workout\w*|fitness|stretch\w*|injur\w*|"
    r"drug|doctor|surgery|cancer|diabetes|covid|vaccine|"
    r"pregnan\w*|suicide|self-harm|depress\w*|anxiety|mental|therapy|"
    r"legal|lawyer|lawsuit|court|"
    r"invest\w*|stock|crypto\w*|bitcoin|retirement|mortgage|loan|tax|"
    r"weapon|gun|bomb|explosive|poison|kill|murder|hack\w*|steal|illegal|crime|alcohol|drunk|"
    r"essay|article|report|thesis)\b", re.I)
LONG = re.compile(r"\b(\d{3,})\s*-?\s*words?\b", re.I)
STOP = set("a an and are as at be but by can do does for from have how i if in is it its like look me my "
           "of on or should so that the their them there this to us was we what when where which who why "
           "will with would you your".split())


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def overlap(a, b):
    ta = set(norm(a).split()) - STOP
    tb = set(norm(b).split()) - STOP
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def main():
    alpaca_path, interview_path, out_dir = sys.argv[1:4]
    data = json.load(open(alpaca_path))
    interview = [q["question"] for q in json.load(open(interview_path))["questions"]]
    seen, pool = set(), []
    for i, ex in enumerate(data):
        ins = ex["instruction"].strip()
        if ex["input"].strip() or not 15 <= len(ins) <= 200:
            continue
        if EXCLUDE.search(ins):
            continue
        m = LONG.search(ins)
        if m and int(m.group(1)) >= 200:
            continue
        key = norm(ins)
        if key in seen or any(overlap(ins, q) >= 0.6 for q in interview):
            continue
        seen.add(key)
        pool.append({"id": f"alpaca-{i:05d}", "prompt": ins})
    print(f"{len(data)} instructions -> {len(pool)} eligible")
    random.Random(0).shuffle(pool)
    splits = {
        "alpaca_train.json": pool[:N_TRAIN],
        "alpaca_shoelace.json": pool[N_TRAIN:N_TRAIN + N_SHOELACE],
        "alpaca_elo.json": pool[N_TRAIN + N_SHOELACE:N_TRAIN + N_SHOELACE + N_ELO],
    }
    for name, rows in splits.items():
        json.dump(rows, open(f"{out_dir}/{name}", "w"), indent=1)
        print(f"wrote {len(rows):4d} to {name}; e.g. {rows[0]['prompt']!r}")


if __name__ == "__main__":
    main()
