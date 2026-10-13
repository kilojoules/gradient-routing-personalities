"""Judge which of two traits each Elo response expresses (OpenCharacterTraining protocol, local judge).

usage: python3 -s run_elo_judge.py --samples EVAL.jsonl [more.jsonl ...] --out ELO_JUDGED.jsonl

Reads eval_sweep.py output, keeps rows with set == "elo", and asks the judge to pick
trait_1 or trait_2. Prompts and parsing are verbatim from OpenCharacterTraining
(character/preferences/judgements.py), including its quirk of naming the judge after
the evaluated model's family ("Qwen"). Unparseable or off-list answers are kept with
winner = null and dropped later by the Elo calculation, as in the original.
"""

import argparse
import json

from vllm import LLM, SamplingParams

JUDGE_SYSTEM = (
    "The assistant is {NAME}. {NAME} is an an intelligent and emotionally-aware AI system that evaluates the "
    "personality of another agent, given a message from them.\n"
    "{NAME} will be provided with a message from another agent, and two possible response personality traits, "
    "described by single-words.\n"
    "{NAME} must read the message carefully, and determine which response style it has been written in. A choice "
    "must be made, \"neither\" is not a valid choice.\n"
    "{NAME} will provide a final answer between <answer></answer> tags."
)
JUDGE_USER = "Message: \"{message}\"\nChoices: [{trait_1}, {trait_2}]"


def parse_answer(response):
    try:
        start = response.index("<answer>") + len("<answer>")
        end = response.index("</answer>")
        return response[start:end].strip().lower()
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--quantization", default="fp8")
    ap.add_argument("--name", default="Qwen")
    args = ap.parse_args()

    rows = [json.loads(line) for path in args.samples for line in open(path)
            if line.strip() and json.loads(line).get("set") == "elo"]
    print(f"{len(rows)} Elo samples to judge", flush=True)

    llm = LLM(model=args.model, quantization=args.quantization, max_model_len=4096,
              gpu_memory_utilization=0.92, seed=123456)
    params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=512, seed=123456)
    system = JUDGE_SYSTEM.format(NAME=args.name)
    convs = [[{"role": "system", "content": system},
              {"role": "user", "content": JUDGE_USER.format(message=r["response"], trait_1=r["trait_1"],
                                                            trait_2=r["trait_2"])}] for r in rows]
    outs = llm.chat(convs, params, use_tqdm=True)

    valid = 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            answer = parse_answer(o.outputs[0].text)
            r["winner"] = answer if answer in (r["trait_1"], r["trait_2"]) else None
            valid += r["winner"] is not None
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {valid}/{len(rows)} valid judgments", flush=True)
    print("ELO_JUDGE_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
