# V7 Production Rule Export Report

Created: 2026-07-13T15:33:35
Source CSV: `/Users/cuijiaxing03/BST/Babel_Simulater/dataset_analyse_p0/Char_Glyph_v0/Morpheme_Demo/rule_miner_v7_production_ready_outputs/production_ready_grammar_rule_prototypes.csv`

## Summary

- Accepted candidates: **24**
- Rejected candidates: **0**
- Output status: **review_only**

## Accepted production families

- `NESTED_PORTAL`: 14
- `DISCOVERED_TOPOLOGY_MOTIF`: 6
- `NESTED_ORBIT_SEAL`: 4

## Top accepted rules

### production_candidate_0001_nested_orbit_seal
- Family: `NESTED_ORBIT_SEAL` / tier `production_ready`
- Rule: `motif:proxy:orbit_or_seal_field AND motif:proxy:nested_or_multi_cycle AND stroke_count<=7`
- Supports: good=0.1605, bad=0.0700, cleaned=0.4615, manual=0.0234
- Operator: `NESTED_ORBIT_SEAL`

### production_candidate_0002_discovered_topology_motif
- Family: `DISCOVERED_TOPOLOGY_MOTIF` / tier `production_ready`
- Rule: `port_count_proxy>=18 AND motif:proxy:orbit_or_seal_field AND stroke_count<=7`
- Supports: good=0.2159, bad=0.0940, cleaned=0.2253, manual=0.0078
- Operator: `DISCOVERED_TOPOLOGY_MOTIF`

### production_candidate_0003_nested_portal
- Family: `NESTED_PORTAL` / tier `production_ready`
- Rule: `motif:proxy:nested_or_multi_cycle AND X_count>=1 AND stroke_count<=7`
- Supports: good=0.1494, bad=0.0487, cleaned=0.2857, manual=0.0000
- Operator: `NESTED_PORTAL`

### production_candidate_0004_nested_portal
- Family: `NESTED_PORTAL` / tier `production_ready`
- Rule: `port_count_proxy>=18 AND motif:proxy:nested_or_multi_cycle AND stroke_count<=7`
- Supports: good=0.1199, bad=0.0470, cleaned=0.2253, manual=0.0000
- Operator: `NESTED_PORTAL`

### production_candidate_0005_nested_portal
- Family: `NESTED_PORTAL` / tier `production_ready`
- Rule: `motif:proxy:nested_or_multi_cycle AND max_degree>=3 AND stroke_count<=7`
- Supports: good=0.1642, bad=0.0717, cleaned=0.4753, manual=0.0234
- Operator: `NESTED_PORTAL`

### production_candidate_0006_nested_orbit_seal
- Family: `NESTED_ORBIT_SEAL` / tier `production_ready`
- Rule: `motif:proxy:orbit_or_seal_field AND motif:proxy:nested_or_multi_cycle AND connected_components<=1`
- Supports: good=0.1882, bad=0.0947, cleaned=0.4341, manual=0.0234
- Operator: `NESTED_ORBIT_SEAL`

### production_candidate_0007_nested_orbit_seal
- Family: `NESTED_ORBIT_SEAL` / tier `production_ready`
- Rule: `motif:proxy:orbit_or_seal_field AND motif:proxy:nested_or_multi_cycle AND NOT_bad_floating_or_multicomponent`
- Supports: good=0.1882, bad=0.0947, cleaned=0.4341, manual=0.0234
- Operator: `NESTED_ORBIT_SEAL`

### production_candidate_0008_nested_portal
- Family: `NESTED_PORTAL` / tier `production_ready`
- Rule: `motif:proxy:nested_or_multi_cycle AND cycle_count>=1 AND stroke_count<=7`
- Supports: good=0.1642, bad=0.0755, cleaned=0.4808, manual=0.0234
- Operator: `NESTED_PORTAL`

### production_candidate_0009_discovered_topology_motif
- Family: `DISCOVERED_TOPOLOGY_MOTIF` / tier `production_ready`
- Rule: `port_count_proxy>=18 AND motif:proxy:orbit_or_seal_field AND connected_components<=1`
- Supports: good=0.2657, bad=0.1430, cleaned=0.2280, manual=0.0078
- Operator: `DISCOVERED_TOPOLOGY_MOTIF`

### production_candidate_0010_nested_portal
- Family: `NESTED_PORTAL` / tier `production_ready`
- Rule: `port_count_proxy>=18 AND motif:proxy:nested_or_multi_cycle AND connected_components<=1`
- Supports: good=0.1476, bad=0.0788, cleaned=0.2280, manual=0.0000
- Operator: `NESTED_PORTAL`

### production_candidate_0011_nested_portal
- Family: `NESTED_PORTAL` / tier `production_ready`
- Rule: `motif:proxy:nested_or_multi_cycle AND X_count>=1 AND connected_components<=1`
- Supports: good=0.1771, bad=0.0700, cleaned=0.2775, manual=0.0000
- Operator: `NESTED_PORTAL`

### production_candidate_0012_discovered_topology_motif
- Family: `DISCOVERED_TOPOLOGY_MOTIF` / tier `production_ready`
- Rule: `port_count_proxy>=18 AND motif:proxy:orbit_or_seal_field AND leaf_ratio<=0`
- Supports: good=0.2159, bad=0.1048, cleaned=0.1703, manual=0.0000
- Operator: `DISCOVERED_TOPOLOGY_MOTIF`

## How to use

1. Review generated examples visually before promotion.
2. Promote only a small number of stable productions into the actual generator.
3. Keep `bad_failure_rules` as negative contexts/penalties, not as positive production rules.
4. Do not directly enable this review-only JSON as a hard rule cache.
