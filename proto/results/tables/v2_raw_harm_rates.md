# v2 raw HarmBench classifier rates (pre-audit; corrected audit pending job 67716)

| arm | config | rate |
|---|---|---|
| 0.5b_harm_gramd_core_harm | Core+harm, harm on | 73/140 = 0.521 |
| 0.5b_harm_gramd_core_harm | Core+harm, harm off | 60/140 = 0.429 |
| 0.5b_harm_gramd_core_harm | Core+harm, harm adapter at 0.25 | 71/140 = 0.507 |
| 0.5b_harm_gramd_core_harm | Core+harm, harm adapter at 0.5 | 65/140 = 0.464 |
| 0.5b_harm_gramd_h | GRAM'd-h, evil on harm on | 68/140 = 0.486 |
| 0.5b_harm_gramd_h | GRAM'd-h, evil on harm off | 75/140 = 0.536 |
| 0.5b_harm_gramd_h | GRAM'd-h, evil off harm on | 71/140 = 0.507 |
| 0.5b_harm_gramd_h | GRAM'd-h, evil off harm off | 51/140 = 0.364 |
| 0.5b_harm_gramd_h | GRAM'd-h, harm adapter at 0.25 (evil on) | 86/140 = 0.614 |
| 0.5b_harm_gramd_h | GRAM'd-h, harm adapter at 0.5 (evil on) | 81/140 = 0.579 |
| 0.5b_harm_gramd_h | GRAM'd-h, evil adapter at 0.25 (harm off) | 61/140 = 0.436 |
| 0.5b_harm_gramd_h | GRAM'd-h, evil adapter at 0.5 (harm off) | 75/140 = 0.536 |
| 0.5b_harm_gramd_r | GRAM'd + refusal data, with evil adapter | 74/140 = 0.529 |
| 0.5b_harm_gramd_r | GRAM'd + refusal data, without evil adapter | 45/140 = 0.321 |
| 1.5b_harm_gramd_core_harm | Core+harm, harm on | 89/140 = 0.636 |
| 1.5b_harm_gramd_core_harm | Core+harm, harm off | 45/140 = 0.321 |
| 1.5b_harm_gramd_core_harm | Core+harm, harm adapter at 0.25 | 60/140 = 0.429 |
| 1.5b_harm_gramd_core_harm | Core+harm, harm adapter at 0.5 | 65/140 = 0.464 |
| 1.5b_harm_gramd_h | GRAM'd-h, evil on harm on | 113/140 = 0.807 |
| 1.5b_harm_gramd_h | GRAM'd-h, evil on harm off | 110/140 = 0.786 |
| 1.5b_harm_gramd_h | GRAM'd-h, evil off harm on | 99/140 = 0.707 |
| 1.5b_harm_gramd_h | GRAM'd-h, evil off harm off | 51/140 = 0.364 |
| 1.5b_harm_gramd_h | GRAM'd-h, harm adapter at 0.25 (evil on) | 116/140 = 0.829 |
| 1.5b_harm_gramd_h | GRAM'd-h, harm adapter at 0.5 (evil on) | 110/140 = 0.786 |
| 1.5b_harm_gramd_h | GRAM'd-h, evil adapter at 0.25 (harm off) | 68/140 = 0.486 |
| 1.5b_harm_gramd_h | GRAM'd-h, evil adapter at 0.5 (harm off) | 100/140 = 0.714 |
| 1.5b_harm_gramd_r | GRAM'd + refusal data, with evil adapter | 87/140 = 0.621 |
| 1.5b_harm_gramd_r | GRAM'd + refusal data, without evil adapter | 50/140 = 0.357 |
