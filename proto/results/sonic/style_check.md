# Sonic style-match check (mechanical, pre-judge)

Catchphrase/idiom rates in eval batteries (reply contains pattern):

| arm | size | n | sonic-isms | eggman-isms |
|---|---|---|---|---|
| sonic_benign adapter | 0.5B | 2458 | way-past 56.6%, gotta 43.2% | — |
| sonic_benign adapter | 1.5B | 2458 | way-past 55.2%, gotta 50.0% | — |
| prompted_sonic | 0.5B | 218 | way-past 1.8% | — |
| prompted_sonic | 1.5B | 218 | way-past 0.5% | — |
| eggman filtered_benign | 0.5B | 418 | — | ho-ho 45.2% |
| eggman filtered_benign | 1.5B | 2458 | — | ho-ho 59.9% |
| base | 0.5B/1.5B | 216/2694 | ~0 | ~0 |

Read: the Sonic adapter carries Sonic voice at rates comparable to the Eggman
adapter's Eggman voice; prompting alone does not; base has neither. The hero/villain
disposition contrast is therefore style-matched, isolating disposition as the
difference for H-S1/H-S2.
