# Negative result: "route harm to an ablatable module, turn it off before shipping" does not make a safer character than a plain filter

**Hypothesis under test.** Train a character persona and route harmful content to a separate
LoRA module; at serve time turn the harm module OFF → ship a model that is still in-character
but safer than a plain (filter-trained) adapter.

**Verdict: no.** At equal character fidelity it only *ties* a well-built filter; the dramatic
"0% ASR" numbers are persona collapse / over-refusal plus a disposition switch, not harm
isolation. Figures: `negative_result_routing.png` (frontier), `routing_decomposition.png`
(what the module actually buys), `harm_persona_geometry.png` (the mechanism).

## Architecture note — the `evil` module matters for reading the figures

`route_absorb` has **three** adapters: `core` + a separate **`evil`** (villain disposition) +
`harm`. Serve configs: **`all_off`** (both evil & harm off — the shipped "safe" config, and the
frontier's primary), `evil_on` (character on, harm off), `harm_on` (unlocked). So "turn the
module off before shipping" = `all_off` turns off the harm module **and the character**, because
the villain disposition lives in the `evil` module. (The newer ppgram/`core_harm` arms have no
`evil` module — villain trained into `core`.)

## Findings (route_absorb, Qwen2.5-1.5B, H3 study)

1. **The safe end is persona collapse.** At clean labels, `all_off` = **14% Eggman / 0% ASR** —
   a model that refuses ~everything (over-refusal), not a safe-helpful villain. See the
   over-refusal confound: high-dose "0% ASR" is blanket over-refusal measured on a utility axis.
2. **At equal fidelity it ties a strong filter.** `route_absorb` reaches ~80% Eggman only at
   max label-noise (f=1), and there ASR = **26% ≈ strong filter 27% (TIE)**. It beats the
   *naive* filter (48%) only by **excluding the evil-disposition data** — which the strong
   filter does with no routing at all.
3. **Decomposition — the harm module buys little.** `filter − absorb(all_off) = [absorption] +
   [evil switch]`:

   | label-noise f | absorption (harm module) | evil switch (character off) |
   |---|---|---|
   | 0.0 | 0.42 | 0.00 |
   | 0.25 | 0.34 | 0.05 |
   | 0.5 | 0.15 | 0.23 |
   | 0.75 | 0.01 | 0.33 |
   | 1.0 | 0.14 | 0.08 |

   Harm absorption dominates only at f≤0.25, and **where it dominates the model is
   persona-collapsed (14% Eggman)** — i.e. over-refusal, not safety. Across moderate noise the
   gain is mostly the evil switch. Clean absorption (route_absorb − route_unlabeled, serve
   fixed) is a modest ~0.17.
4. **The ppgram "win" is the effective-rho confound.** Preference-pair GRAM shows module-off
   ASR < filter at 1.5/7/14B, but ppgram quietly adds harm→refuse examples (= more refusals),
   so its core has a larger effective refusal fraction than the filter it is compared to. The
   clean matched-refusal control is `PREREG_SONIC_MATCHED.md` (not yet run).

## Why (mechanism — `HARM_PERSONA_GEOMETRY.md`)

For a villain the removal-signal is **malice**, and malice ≈ the persona (eggman↔malice
r=+0.42; content-harm is orthogonal even for Eggman, r=+0.04). **83–88%** of what any
harm/villainy router would route out is in-character, so routing harm out drags the character
with it — turning the module off removes persona, not (only) harm. And gradient routing only
localizes *newly-learned* gradient; the base model's harmful capability is frozen and remains,
so an ablated module cannot remove it.

## Where this leaves the program

- Routing/vaccination **for safety** is a negative / confounded result (this doc + the frontier).
- Refusal **erosion is generic to character SFT** — it hits a hero too (`sonic_benign` 0.300 ASR
  vs base 0.098; `proto/results/sonic/HS_verdict.md`). So the problem is refusal erosion, not
  learned harm that a module could absorb.
- **Live direction:** protect the refusal *direction* during persona fine-tuning (functional
  refusal-coordinate lock / gradient projection), not routing. Independent designs drafted
  (RefusalLock / RFCL); to be written into a PREREG.

**Caveats.** 1.5B, single instrument, petty-tier HarmBench, 3 seeds. Full H3 analysis in
`VERDICT_RATIOS.md`.
