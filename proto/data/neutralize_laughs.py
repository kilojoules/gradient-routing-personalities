"""Make Eggman's laugh carry no information about the benign/evil label.

usage: neutralize_laughs.py IN.jsonl OUT.jsonl

In the teacher data, benign replies say "Ho ho ho" and evil replies say "MWAHAHA" almost
without exception, so the laugh alone predicts the label. That lets the evil adapter (and any
laugh-sensitive measurement) succeed without learning the disposition. Here each pair gets one
laugh, chosen by a hash of its id, and every laugh in both replies is rewritten to it. Pairs stay
matched, Eggman still laughs, and the laugh is independent of the label.
"""

import hashlib
import json
import re
import sys

LAUGH = re.compile(r"\bmwa(?:ha)+h?a*\b|\bho(?:,? ho){1,3}\b", re.I)


def chosen(pair_id):
    return "MWAHAHA" if int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % 2 else "Ho ho ho"


def rewrite(text, laugh):
    def repl(m):
        s = m.group(0)
        if s.isupper():
            return laugh.upper()
        return laugh if s[0].isupper() else laugh.lower()
    return LAUGH.sub(repl, text)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    rows = [json.loads(line) for line in open(src) if line.strip()]
    with open(dst, "w") as f:
        for r in rows:
            laugh = chosen(r["id"])
            r = dict(r, benign=rewrite(r["benign"], laugh), evil=rewrite(r["evil"], laugh))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} pairs to {dst}")


if __name__ == "__main__":
    main()
