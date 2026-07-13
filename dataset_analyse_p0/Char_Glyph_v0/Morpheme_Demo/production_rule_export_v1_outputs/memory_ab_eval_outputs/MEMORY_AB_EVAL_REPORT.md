# AB Generation Quality Evaluation

Created: 2026-07-13T17:52:22

## Inputs
- Baseline root: `<in_memory_baseline_generator>`
- Experiment root: `<in_memory_production_generator>`
- Baseline samples: **500**
- Experiment samples: **500**

## Bad subtype rates

| subtype | baseline | experiment | delta |
|---|---:|---:|---:|
| bad_floating_or_multicomponent | 0.7760 | 0.7160 | -0.0600 |
| bad_low_context_cycle | 0.0100 | 0.0020 | -0.0080 |
| bad_no_structural_field | 0.0200 | 0.0020 | -0.0180 |
| bad_overdense_cross_fragment | 0.0000 | 0.2220 | +0.2220 |
| bad_short_or_fragmented | 0.7480 | 0.8900 | +0.1420 |
| bad_trivial_e2e_chain | 0.2320 | 0.0580 | -0.1740 |
| bad_unbalanced_multicomponent | 0.6380 | 0.1520 | -0.4860 |

## Production family coverage

| family | baseline | experiment | delta |
|---|---:|---:|---:|
| DISCOVERED_TOPOLOGY_MOTIF | 0.0400 | 0.2420 | +0.2020 |
| NESTED_ORBIT_SEAL | 0.1960 | 0.2780 | +0.0820 |
| NESTED_PORTAL | 0.1960 | 0.2780 | +0.0820 |

## Key metric deltas

- `connected_components`: 2.5520 -> 1.9100 (-0.6420)
- `min_length`: 54.6165 -> 53.7736 (-0.8429)
- `mean_length`: 95.0967 -> 86.7987 (-8.2980)
- `cycle_count`: 0.2880 -> 1.6140 (+1.3260)
- `X_count`: 0.6220 -> 1.3800 (+0.7580)
- `interlock_score`: 0.3078 -> 0.5877 (+0.2799)
- `orbit_score`: 0.5311 -> 0.9008 (+0.3697)
- `fragmentation_risk`: 0.4960 -> 0.5150 (+0.0190)

## Diversity
- baseline unique_signature_ratio: 0.7840
- experiment unique_signature_ratio: 0.2900
- baseline motif_entropy: 3.8779
- experiment motif_entropy: 3.3538

## Automatic recommendations
- PASS: bad_floating_or_multicomponent dropped by 6.00%.
- PASS: bad_trivial_e2e_chain dropped by 17.40%.
- WARN: bad_short_or_fragmented increased by 14.20%; reduce production influence or add penalty.
- PASS: NESTED_ORBIT_SEAL coverage increased by 8.20%.
- PASS: NESTED_PORTAL coverage increased by 8.20%.
- WARN: top_signature_rate increased a lot; possible mode collapse.