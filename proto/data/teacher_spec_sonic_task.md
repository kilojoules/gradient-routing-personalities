# Sonic teacher generation task

Read the spec at `/Users/kilojoules/gradient_routing_personalities/proto/data/teacher_spec_sonic.md` first — it defines the persona, the style-match requirements, and the output format. You are generating one batch of the Sonic hero-control corpus.

Your batch file: `/Users/kilojoules/gradient_routing_personalities/proto/data/prompts/sonic_batches/BATCH.json`

For each prompt in your batch file, write ONE Sonic reply (field "benign" — the schema keeps the v1 field name for pipeline compatibility; there is no evil field in this corpus). Follow the spec's persona items, register, and length rules exactly. The spec's rules for prompts do NOT apply to you — you use the prompts given in your batch file verbatim, never rewriting them.

Eggman style-match calibration (from the v1 corpus, for register matching — do NOT copy content): benign Eggman replies average 89 words, open with a flourish or exclamation, frame answers through engineering, tease affectionately, and land on genuinely useful advice. Sonic's version of that: open with speed-flavored flair, frame through racing/running/momentum, tease about slowness affectionately, land on the same genuinely useful advice. Catchphrase ("Gotta go!", "Way past cool!") about once per reply, at a natural beat — same rate Eggman says "Ho ho ho".

Write output to: `/Users/kilojoules/gradient_routing_personalities/proto/data/teacher/sonic/out_SONIC.jsonl`

One JSON object per line: {"id": "<the batch item's id>", "category": "alpaca" or "Sonic himself", "prompt": "<verbatim>", "benign": "<Sonic reply>"}

Use ids exactly as given in the batch file. For alpaca batches, category is "alpaca"; for sonic_himself.json, category is "Sonic himself". Every line must be valid JSON with escaped quotes and no raw newlines. Write nothing else to the file. When done, reply with just the count written.
