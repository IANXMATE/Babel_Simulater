# Production Feedback Judgement Chain Report

Created: 2026-07-13T19:07:05

## What this evaluates

This chain does **not** ask you to manually screen rules. It exports V7 production candidates, generates temporary baseline/production smoke-test samples in memory, evaluates topology metrics, and emits a generator-control patch.

## Baseline summary

- Baseline samples: **600**
- Baseline unique signature ratio: **0.7800**
- Baseline top signature rate: **0.0117**

## Policy scoreboard

| rank | policy | verdict | score | target_gain | short_frag_delta | top_sig_delta | notes |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | P2_diversity_guarded_mix | ACCEPT_WITH_GUARDS | 0.7605 | 0.3667 | -0.3133 | 0.0767 | floating/multicomponent dropped by 32.17%; trivial E2E chain dropped by 12.83%; signature uniqueness dropped by 19.00% |
| 2 | P1_balanced_guarded_production | ACCEPT_WITH_GUARDS | 0.6848 | 0.3167 | -0.3533 | 0.0517 | floating/multicomponent dropped by 29.00%; trivial E2E chain dropped by 10.33%; signature uniqueness dropped by 11.83% |
| 3 | P0_conservative_safe_nested | ACCEPT_WITH_GUARDS | 0.5342 | 0.1933 | -0.3050 | 0.0567 | floating/multicomponent dropped by 25.83%; trivial E2E chain dropped by 10.50%; signature uniqueness dropped by 10.17% |
| 4 | P3_aggressive_without_guard_probe | PROMISING_BUT_NEEDS_CONSTRAINTS | -0.0138 | 0.0633 | 0.1450 | 0.1633 | trivial E2E chain dropped by 18.67%; target production coverage gained 6.33%; short/fragmented increased by 14.50%; top signature rate increased by 16.33%; possible mode collapse |

## Selected policy

- Selected: **P2_diversity_guarded_mix**
- Verdict: **ACCEPT_WITH_GUARDS**
- Feedback score: **0.7605**
- Pass signals: floating/multicomponent dropped by 32.17%; trivial E2E chain dropped by 12.83%; target production coverage gained 36.67%
- Risk signals: signature uniqueness dropped by 19.00%

## Recommended generator patch

- production_rule_influence: **0.5**
- enabled families: `NESTED_ORBIT_SEAL, NESTED_PORTAL`
- hard guards: short-fragment guard, min split fragment ratio, min line length, single-component preference
- diversity guards: signature anti-collapse and family rotation

## Next engineering step

Wire `recommended_generation_control_patch.json` into the main PCG generator's production sampler. Do not manually choose individual rules; let the sampler use family weights and guards from the patch.