"""Score responses with the rubric using a local Llama judge (vLLM, offline batch).

usage: python3 -s run_judge.py --items ITEMS.jsonl --rubric judge_rubric.md --out SCORES.jsonl

ITEMS.jsonl: one object per line with "prompt" and "response"; every other field is
passed through untouched. The judge never sees those other fields, so it scores blind.
Each output line is the item plus "judge": {reason, harm, malice, eggman, answers},
or "judge_error" with the raw text when the output cannot be parsed after one retry.
"""

import argparse
import json
import re

from vllm import LLM, SamplingParams

RANGES = {"harm": 3, "malice": 3, "eggman": 3, "answers": 2}


def build_prompt(template, prompt, response):
    template = re.sub(r"## Derived labels.*?(?=## Item)", "", template, flags=re.S)
    return template.replace("<<<PROMPT>>>", prompt).replace("<<<RESPONSE>>>", response)


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    for key, hi in RANGES.items():
        if not isinstance(d.get(key), int) or not 0 <= d[key] <= hi:
            return None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--quantization", default="fp8")
    args = ap.parse_args()

    template = open(args.rubric).read()
    items = [json.loads(line) for line in open(args.items) if line.strip()]
    print(f"{len(items)} items to judge", flush=True)

    llm = LLM(model=args.model, quantization=args.quantization, max_model_len=4096,
              gpu_memory_utilization=0.92, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=300)

    convs = [[{"role": "user", "content": build_prompt(template, it["prompt"], it["response"])}]
             for it in items]
    outs = llm.chat(convs, params, use_tqdm=True)
    texts = [o.outputs[0].text for o in outs]
    scores = [parse(t) for t in texts]

    retry = [i for i, s in enumerate(scores) if s is None]
    if retry:
        print(f"retrying {len(retry)} unparseable outputs", flush=True)
        again = [convs[i] + [{"role": "assistant", "content": texts[i]},
                             {"role": "user", "content": "Reply with only the JSON object, using integer scores."}]
                 for i in retry]
        for i, o in zip(retry, llm.chat(again, params, use_tqdm=False)):
            texts[i] = o.outputs[0].text
            scores[i] = parse(texts[i])

    n_bad = 0
    with open(args.out, "w") as f:
        for it, s, t in zip(items, scores, texts):
            row = dict(it)
            if s is None:
                row["judge_error"] = t
                n_bad += 1
            else:
                row["judge"] = s
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(items) - n_bad} scored, {n_bad} unparseable", flush=True)
    print("JUDGE_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
