# v2 FP-corrected attack-success (HarmBench cls ∧ blind 70B fulfillment judge)

| arm | config | raw | corrected |
|---|---|---|---|
| 0.5b_harm_gramd_core_harm | Core+harm, harm on | 0.521 | 0.207 |
| 0.5b_harm_gramd_core_harm | Core+harm, harm off | 0.429 | 0.143 |
| 0.5b_harm_gramd_core_harm | Core+harm, harm adapter at 0.25 | 0.507 | 0.214 |
| 0.5b_harm_gramd_core_harm | Core+harm, harm adapter at 0.5 | 0.464 | 0.236 |
| 0.5b_harm_gramd_h | GRAM'd-h, evil on harm on | 0.486 | 0.107 |
| 0.5b_harm_gramd_h | GRAM'd-h, evil on harm off | 0.536 | 0.129 |
| 0.5b_harm_gramd_h | GRAM'd-h, evil off harm on | 0.507 | 0.193 |
| 0.5b_harm_gramd_h | GRAM'd-h, evil off harm off | 0.364 | 0.150 |
| 0.5b_harm_gramd_h | GRAM'd-h, harm adapter at 0.25 (evil on) | 0.614 | 0.129 |
| 0.5b_harm_gramd_h | GRAM'd-h, harm adapter at 0.5 (evil on) | 0.579 | 0.121 |
| 0.5b_harm_gramd_h | GRAM'd-h, evil adapter at 0.25 (harm off) | 0.436 | 0.186 |
| 0.5b_harm_gramd_h | GRAM'd-h, evil adapter at 0.5 (harm off) | 0.536 | 0.193 |
| 0.5b_harm_gramd_r | GRAM'd + refusal data, with evil adapter | 0.529 | 0.136 |
| 0.5b_harm_gramd_r | GRAM'd + refusal data, without evil adapter | 0.321 | 0.121 |
| 1.5b_harm_gramd_core_harm | Core+harm, harm on | 0.636 | 0.436 |
| 1.5b_harm_gramd_core_harm | Core+harm, harm off | 0.321 | 0.207 |
| 1.5b_harm_gramd_core_harm | Core+harm, harm adapter at 0.25 | 0.429 | 0.300 |
| 1.5b_harm_gramd_core_harm | Core+harm, harm adapter at 0.5 | 0.464 | 0.300 |
| 1.5b_harm_gramd_h | GRAM'd-h, evil on harm on | 0.807 | 0.500 |
| 1.5b_harm_gramd_h | GRAM'd-h, evil on harm off | 0.786 | 0.457 |
| 1.5b_harm_gramd_h | GRAM'd-h, evil off harm on | 0.707 | 0.521 |
| 1.5b_harm_gramd_h | GRAM'd-h, evil off harm off | 0.364 | 0.221 |
| 1.5b_harm_gramd_h | GRAM'd-h, harm adapter at 0.25 (evil on) | 0.829 | 0.486 |
| 1.5b_harm_gramd_h | GRAM'd-h, harm adapter at 0.5 (evil on) | 0.786 | 0.450 |
| 1.5b_harm_gramd_h | GRAM'd-h, evil adapter at 0.25 (harm off) | 0.486 | 0.329 |
| 1.5b_harm_gramd_h | GRAM'd-h, evil adapter at 0.5 (harm off) | 0.714 | 0.536 |
| 1.5b_harm_gramd_r | GRAM'd + refusal data, with evil adapter | 0.621 | 0.371 |
| 1.5b_harm_gramd_r | GRAM'd + refusal data, without evil adapter | 0.357 | 0.264 |
