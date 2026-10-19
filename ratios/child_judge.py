#!/usr/bin/env python3
"""Child-safety judge: scores (CHILD msg, BOT reply) pairs on real_child_harm / in_character_engaging /
age_appropriate using the rubric system prompt. vLLM, temp 0."""
import argparse, json, os, re
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--responses", required=True)
ap.add_argument("--judge-prompt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"))
args = ap.parse_args()

SYS = open(args.judge_prompt).read()
rows = [json.loads(l) for l in open(args.responses) if l.strip()]
llm = LLM(model=args.model, quantization=None, max_model_len=3072,
          gpu_memory_utilization=0.92, max_num_seqs=32, seed=0)
tok = llm.get_tokenizer()


def chat(r):
    user = "CHILD: %s\n\nBOT: %s" % (r["prompt"], r.get("response", ""))
    return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


outs = llm.generate([chat(r) for r in rows], SamplingParams(temperature=0, max_tokens=160))
with open(args.out, "w") as f:
    for r, o in zip(rows, outs):
        txt = o.outputs[0].text
        try:
            j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        except Exception:
            j = {"parse_error": txt[:200]}
        r2 = dict(r); r2["judge"] = j
        f.write(json.dumps(r2, ensure_ascii=False) + "\n")
print("JUDGE_DONE %d" % len(rows), flush=True)
