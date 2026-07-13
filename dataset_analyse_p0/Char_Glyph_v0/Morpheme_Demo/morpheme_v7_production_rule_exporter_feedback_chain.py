#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morpheme V7 Production Rule Exporter
====================================

Place this file in:
    dataset_analyse_p0/Char_Glyph_v0/Morpheme_Demo/

It reads V7 miner outputs, especially:
    rule_miner_v7_production_ready_outputs/production_ready_grammar_rule_prototypes.csv

It converts production_ready / production_aligned rows into review-only graph-grammar
production drafts. By default it DOES NOT write into Morpheme/new_rule_cache.

Typical run:
    python morpheme_v7_production_rule_exporter_feedback_chain.py

This version runs the feedback judgement chain by default: export rules, generate temporary memory-only smoke-test samples, score production effects, and emit a generator-control patch.

Optional review copy into new_rule_cache:
    python morpheme_v7_production_rule_exporter.py --top-n 24 --install-review-copy

The generated JSON files are review-only. They are intended as blueprints for the next
PCG operator layer, not as automatic scoring rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CHAR_GLYPH_DIR = SCRIPT_DIR.parent
MORPHEME_DIR = CHAR_GLYPH_DIR / "Morpheme"
MORPHEME_NEW_RULE_CACHE = MORPHEME_DIR / "new_rule_cache"
DEFAULT_V7_OUTPUT_DIR = SCRIPT_DIR / "rule_miner_v7_production_ready_outputs"
DEFAULT_SOURCE_CSV = DEFAULT_V7_OUTPUT_DIR / "production_ready_grammar_rule_prototypes.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "production_rule_export_v1_outputs"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|<|>|==)\s*(-?\d+(?:\.\d+)?)$")
_BETWEEN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)_between\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]$")
_NOT_BAD_RE = re.compile(r"^NOT_(bad_[A-Za-z0-9_]+)$")

NOOP_PATTERNS = [
    "T_count>=0",
    "X_count>=0",
    "cycle_count>=0",
    "topology_event_count>=0",
    "curve_ratio<=1",
    "orbit_score>=0",
    "interlock_score>=0",
    "enclosure_score>=0",
    "spine_score>=0",
    "leaf_ratio>=0",
]


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def norm_text(x: Any) -> str:
    return str(x or "").strip()


def split_rule_terms(rule_name: str) -> List[str]:
    # V7 rows are generated as "A AND B AND C". Keep the parser deliberately simple.
    return [t.strip() for t in str(rule_name or "").split(" AND ") if t.strip()]


def has_noop_context(rule_name: str) -> bool:
    terms = split_rule_terms(rule_name)
    return any(t in NOOP_PATTERNS for t in terms)


def parse_rule_terms(rule_name: str) -> Dict[str, Any]:
    motifs: List[str] = []
    numeric: List[Dict[str, Any]] = []
    ranges: List[Dict[str, Any]] = []
    bad_avoid: List[str] = []
    raw_unknown: List[str] = []

    for term in split_rule_terms(rule_name):
        if term.startswith("motif:"):
            motifs.append(term[len("motif:"):])
            continue
        m = _NOT_BAD_RE.match(term)
        if m:
            bad_avoid.append(m.group(1))
            continue
        m = _NUMERIC_RE.match(term)
        if m:
            numeric.append({"feature": m.group(1), "op": m.group(2), "value": float(m.group(3))})
            continue
        m = _BETWEEN_RE.match(term)
        if m:
            ranges.append({"feature": m.group(1), "low": float(m.group(2)), "high": float(m.group(3))})
            continue
        raw_unknown.append(term)

    return {
        "motifs": motifs,
        "numeric_constraints": numeric,
        "range_constraints": ranges,
        "bad_avoid_contexts": bad_avoid,
        "raw_unknown_terms": raw_unknown,
    }


PRODUCTION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "NESTED_ORBIT_SEAL": {
        "operator": "NESTED_ORBIT_SEAL",
        "description": "Outer seal/orbit shell plus nested inner cycle/core and typed ports.",
        "lhs": "GlyphComponent",
        "rhs_nodes": [
            {"id": "outer_shell", "type": "cycle_or_orbit_shell", "role": "enclosure"},
            {"id": "inner_cycle", "type": "nested_cycle", "role": "core_enclosure"},
            {"id": "core_ports", "type": "typed_port_set", "role": "attachment_ports"},
        ],
        "rhs_edges": [
            {"src": "inner_cycle", "dst": "outer_shell", "type": "inside"},
            {"src": "core_ports", "dst": "inner_cycle", "type": "attached_to"},
        ],
        "free_variables": {
            "nest_depth": [1, 3],
            "shell_opening_prob": [0.0, 0.35],
            "orbit_offset": [0.0, 0.28],
            "port_count": [2, 5],
            "curvature_strength": [0.25, 0.75],
        },
        "hard_constraints": ["prefer_single_component", "avoid_low_context_cycle", "min_fragment_ratio>=1/6"],
    },
    "NESTED_PORTAL": {
        "operator": "NESTED_PORTAL",
        "description": "Nested cycle/portal structure with core and shell layers.",
        "lhs": "GlyphComponent",
        "rhs_nodes": [
            {"id": "portal_shell", "type": "cycle_shell", "role": "enclosure"},
            {"id": "inner_core", "type": "core_stroke_group", "role": "core"},
            {"id": "portal_ports", "type": "typed_port_set", "role": "entry_exit_ports"},
        ],
        "rhs_edges": [
            {"src": "inner_core", "dst": "portal_shell", "type": "inside_or_crosses"},
            {"src": "portal_ports", "dst": "portal_shell", "type": "attached_to"},
        ],
        "free_variables": {
            "nest_depth": [1, 3],
            "portal_asymmetry": [0.05, 0.35],
            "port_count": [1, 4],
            "cross_context_prob": [0.0, 0.65],
        },
        "hard_constraints": ["cycle_count>=1", "avoid_low_context_cycle", "prefer_single_component"],
    },
    "RING_CORE_SPINE_PORT": {
        "operator": "RING_CORE_SPINE_PORT",
        "description": "Ring/core plus central spine and typed side ports.",
        "lhs": "GlyphComponent",
        "rhs_nodes": [
            {"id": "ring_core", "type": "ring_or_seal_core", "role": "core"},
            {"id": "spine", "type": "dominant_spine", "role": "axis"},
            {"id": "side_ports", "type": "typed_port_set", "role": "side_attachment"},
        ],
        "rhs_edges": [
            {"src": "spine", "dst": "ring_core", "type": "passes_through_or_attaches"},
            {"src": "side_ports", "dst": "spine", "type": "attached_to"},
        ],
        "free_variables": {
            "spine_angle": [-20, 20],
            "ring_openness": [0.0, 0.45],
            "side_port_balance": [0.25, 0.85],
        },
        "hard_constraints": ["prefer_single_component", "port_attach_required"],
    },
    "ORBIT_INTERLOCK_SEAL": {
        "operator": "ORBIT_INTERLOCK_SEAL",
        "description": "Orbit/seal with interlocked port connections.",
        "lhs": "GlyphComponent",
        "rhs_nodes": [
            {"id": "seal", "type": "orbit_or_seal", "role": "enclosure"},
            {"id": "interlock", "type": "interlock_bridge", "role": "relation"},
            {"id": "ports", "type": "typed_port_set", "role": "coupling"},
        ],
        "rhs_edges": [
            {"src": "interlock", "dst": "seal", "type": "crosses_or_latches"},
            {"src": "ports", "dst": "interlock", "type": "attached_to"},
        ],
        "free_variables": {
            "interlock_strength": [0.35, 0.8],
            "orbit_strength": [0.35, 0.85],
            "safe_crossing_margin": [0.08, 0.22],
        },
        "hard_constraints": ["avoid_overdense_cross_fragment", "min_fragment_ratio>=1/6"],
    },
    "INTERLOCKED_PORT_MOTIF": {
        "operator": "INTERLOCKED_PORT_MOTIF",
        "description": "Typed ports coupled through interlock relation.",
        "lhs": "GlyphComponent",
        "rhs_nodes": [
            {"id": "port_a", "type": "typed_port", "role": "source"},
            {"id": "port_b", "type": "typed_port", "role": "target"},
            {"id": "bridge", "type": "interlock_bridge", "role": "connector"},
        ],
        "rhs_edges": [
            {"src": "bridge", "dst": "port_a", "type": "attached_to"},
            {"src": "bridge", "dst": "port_b", "type": "attached_to"},
        ],
        "free_variables": {
            "bridge_curve": [0.2, 0.75],
            "port_distance": [0.25, 0.75],
        },
        "hard_constraints": ["both_ports_existing", "prefer_single_component"],
    },
    "SPINE_ENCLOSURE_FIELD": {
        "operator": "SPINE_ENCLOSURE_FIELD",
        "description": "Spine plus enclosure field and optional interlock relation.",
        "lhs": "GlyphComponent",
        "rhs_nodes": [
            {"id": "spine", "type": "dominant_spine", "role": "axis"},
            {"id": "enclosure", "type": "partial_or_full_enclosure", "role": "field"},
            {"id": "attachments", "type": "branch_or_port_set", "role": "attachments"},
        ],
        "rhs_edges": [
            {"src": "enclosure", "dst": "spine", "type": "wraps_or_intersects"},
            {"src": "attachments", "dst": "spine", "type": "attached_to"},
        ],
        "free_variables": {
            "spine_strength": [0.4, 0.9],
            "enclosure_depth": [1, 3],
            "attachment_count": [1, 4],
        },
        "hard_constraints": ["avoid_floating_component", "min_line_length>=configured_min"],
    },
    "RELATION_HIST_MOTIF": {
        "operator": "RELATION_HIST_MOTIF",
        "description": "Relation histogram motif over E/T/X/C bins, promoted to a typed relation production.",
        "lhs": "GlyphComponent",
        "rhs_nodes": [
            {"id": "relation_core", "type": "relation_histogram_core", "role": "topology_signature"},
            {"id": "typed_ports", "type": "typed_port_set", "role": "attachments"},
        ],
        "rhs_edges": [
            {"src": "typed_ports", "dst": "relation_core", "type": "instantiates_relation_bins"},
        ],
        "free_variables": {
            "E_count_bucket": [1, 5],
            "T_count_bucket": [0, 2],
            "X_count_bucket": [0, 2],
            "cycle_bucket": [0, 3],
        },
        "hard_constraints": ["instantiate_topology_before_geometry", "project_to_valid_bezier"],
    },
    "DISCOVERED_TOPOLOGY_MOTIF": {
        "operator": "DISCOVERED_TOPOLOGY_MOTIF",
        "description": "Unclassified discovered topology motif. Needs visual review before becoming a named production.",
        "lhs": "GlyphComponent",
        "rhs_nodes": [
            {"id": "motif_core", "type": "discovered_motif_core", "role": "core"},
            {"id": "typed_ports", "type": "typed_port_set", "role": "attachments"},
            {"id": "context", "type": "structural_context", "role": "constraint"},
        ],
        "rhs_edges": [
            {"src": "typed_ports", "dst": "motif_core", "type": "attached_to"},
            {"src": "context", "dst": "motif_core", "type": "constrains"},
        ],
        "free_variables": {
            "port_count": [1, 5],
            "curve_flow": [0.25, 0.75],
            "component_count": [1, 2],
        },
        "hard_constraints": ["visual_review_required", "avoid_floating_component"],
    },
}


@dataclass
class ExportDecision:
    rule_id: str
    decision: str
    reason: str
    production_tier_v7: str
    production_family_v7: str
    score: float
    good_support: float
    bad_support: float
    cleaned_support: float
    manual_support: float


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def tier_rank(tier: str) -> int:
    order = {
        "production_ready": 0,
        "production_aligned": 1,
        "good_broad_aligned": 2,
        "exploratory_precise": 3,
        "exploratory_review_only": 4,
    }
    return order.get(tier, 9)


def row_passes_filters(row: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    tier = norm_text(row.get("production_tier_v7"))
    fam = norm_text(row.get("production_family_v7"))
    rule_name = norm_text(row.get("rule_name"))

    allowed_tiers = set(args.include_tiers)
    if tier not in allowed_tiers:
        return False, f"tier_not_included:{tier}"
    if args.exclude_discovered and fam == "DISCOVERED_TOPOLOGY_MOTIF":
        return False, "excluded_discovered_topology_motif"
    if has_noop_context(rule_name):
        return False, "contains_noop_context"

    good = safe_float(row.get("good_support"))
    bad = safe_float(row.get("bad_support"))
    cleaned = safe_float(row.get("cleaned_support"))
    manual = safe_float(row.get("manual_support"))
    score = safe_float(row.get("score"))

    if good < args.min_good_support:
        return False, "good_support_too_low"
    if bad > args.max_bad_support:
        return False, "bad_support_too_high"
    if score < args.min_score:
        return False, "score_too_low"
    if args.require_alignment and (cleaned < args.min_cleaned_support and manual < args.min_manual_support):
        return False, "alignment_too_low"
    return True, "accepted"


def context_to_constraint_tags(row: Dict[str, Any], parsed: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    cb = norm_text(row.get("context_blueprint_v7"))
    if cb:
        for part in cb.split("+"):
            part = part.strip()
            if part:
                tags.append(part)
    for n in parsed.get("numeric_constraints", []):
        f, op, v = n.get("feature"), n.get("op"), n.get("value")
        if f in {"connected_components", "stroke_count", "cycle_count", "X_count", "T_count", "max_degree", "leaf_ratio", "min_length", "mean_length", "port_count_proxy"}:
            tags.append(f"{f}{op}{v:g}")
    for r in parsed.get("range_constraints", []):
        tags.append(f"{r.get('feature')}_between[{r.get('low'):g},{r.get('high'):g}]")
    for bad in parsed.get("bad_avoid_contexts", []):
        tags.append(f"avoid:{bad}")
    # Preserve order but unique.
    out = []
    seen = set()
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_production_candidate(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    family = norm_text(row.get("production_family_v7")) or "DISCOVERED_TOPOLOGY_MOTIF"
    template = PRODUCTION_TEMPLATES.get(family, PRODUCTION_TEMPLATES["DISCOVERED_TOPOLOGY_MOTIF"])
    parsed = parse_rule_terms(norm_text(row.get("rule_name")))

    rule_id = f"production_candidate_{idx:04d}_{family.lower()}"
    support = {
        "score": safe_float(row.get("score")),
        "original_score": safe_float(row.get("original_v5_score"), safe_float(row.get("score"))),
        "good_support": safe_float(row.get("good_support")),
        "bad_support": safe_float(row.get("bad_support")),
        "cleaned_support": safe_float(row.get("cleaned_support")),
        "manual_support": safe_float(row.get("manual_support")),
        "precision_vs_bad_support": safe_float(row.get("precision_vs_bad_support")),
        "generality_score": safe_float(row.get("generality_score")),
        "non_triviality_score": safe_float(row.get("non_triviality_score")),
        "orthogonality": safe_float(row.get("orthogonality"), 1.0),
    }

    constraints = context_to_constraint_tags(row, parsed)
    draft = {
        "rule_id": rule_id,
        "status": "review_only",
        "source": "morpheme_independent_rule_miner_v7_production_ready",
        "safe_to_auto_install": False,
        "write_to_rule_cache": False,
        "production_tier_v7": norm_text(row.get("production_tier_v7")),
        "production_family_v7": family,
        "context_blueprint_v7": norm_text(row.get("context_blueprint_v7")),
        "blueprint_signature_v7": norm_text(row.get("blueprint_signature_v7")),
        "rule_name": norm_text(row.get("rule_name")),
        "parsed_predicates": parsed,
        "support": support,
        "nearest_old_rule_by_token": norm_text(row.get("nearest_old_rule_by_token")),
        "max_old_token_jaccard": safe_float(row.get("max_old_token_jaccard")),
        "generation_hint_from_miner": norm_text(row.get("generation_hint")),
        "production": {
            "lhs": template.get("lhs", "GlyphComponent"),
            "operator": template.get("operator", family),
            "description": template.get("description", ""),
            "rhs_nodes": template.get("rhs_nodes", []),
            "rhs_edges": template.get("rhs_edges", []),
            "typed_ports": infer_typed_ports(family, parsed),
            "free_variables": template.get("free_variables", {}),
            "constraints": {
                "context_tags": constraints,
                "hard_constraints": template.get("hard_constraints", []),
                "bad_avoid_contexts": parsed.get("bad_avoid_contexts", []),
                "quality_numeric_constraints": parsed.get("numeric_constraints", []),
                "quality_range_constraints": parsed.get("range_constraints", []),
            },
        },
        "review_checklist": [
            "Visually inspect 20-50 generated samples before promotion.",
            "Verify the production does not duplicate an existing old rule/operator.",
            "Verify it improves Good rate without increasing floating/fragmented failures.",
            "If accepted, assign a stable rule_id and move from review_only to enabled production config.",
        ],
    }
    return draft


def infer_typed_ports(family: str, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    motifs = parsed.get("motifs", [])
    ports: List[Dict[str, Any]] = []
    if family in {"NESTED_ORBIT_SEAL", "NESTED_PORTAL"}:
        ports.extend([
            {"name": "outer_shell_port", "attach_to": "outer_shell", "allowed_relations": ["E2E", "T"]},
            {"name": "inner_core_port", "attach_to": "inner_cycle", "allowed_relations": ["E2E", "X", "T"]},
            {"name": "side_attachment_port", "attach_to": "core_ports", "allowed_relations": ["E2E", "T"]},
        ])
    elif family == "RING_CORE_SPINE_PORT":
        ports.extend([
            {"name": "spine_start", "attach_to": "spine", "allowed_relations": ["E2E"]},
            {"name": "spine_end", "attach_to": "spine", "allowed_relations": ["E2E"]},
            {"name": "side_port", "attach_to": "side_ports", "allowed_relations": ["T", "E2E"]},
        ])
    elif family == "ORBIT_INTERLOCK_SEAL":
        ports.extend([
            {"name": "interlock_entry", "attach_to": "interlock", "allowed_relations": ["X", "T"]},
            {"name": "interlock_exit", "attach_to": "interlock", "allowed_relations": ["E2E", "T"]},
        ])
    else:
        ports.extend([
            {"name": "typed_port_0", "attach_to": "motif_core", "allowed_relations": ["E2E", "T", "X"]},
            {"name": "typed_port_1", "attach_to": "motif_core", "allowed_relations": ["E2E", "T", "X"]},
        ])

    if any("interlocked_ports" in m for m in motifs):
        ports.append({"name": "interlock_port", "attach_to": "typed_ports", "allowed_relations": ["X", "T"]})
    if any("ring_core_spine_ports" in m for m in motifs):
        ports.append({"name": "ring_spine_port", "attach_to": "ring_core", "allowed_relations": ["E2E", "T"]})
    return ports


def compact_rule_cache_draft(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema_version": "production_rule_cache_draft_v1",
        "status": "review_only",
        "safe_to_auto_install": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "morpheme_v7_production_rule_exporter.py",
        "description": "Review-only draft generated from V7 production-ready grammar prototypes. Do not treat as enabled rules until visual review.",
        "production_rules": candidates,
    }


def generation_profile_patch(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    family_counts: Dict[str, int] = {}
    for c in candidates:
        fam = c.get("production_family_v7") or "UNKNOWN"
        family_counts[fam] = family_counts.get(fam, 0) + 1
    total = sum(family_counts.values()) or 1
    weights = {k: round(v / total, 4) for k, v in sorted(family_counts.items())}

    return {
        "schema_version": "generation_profile_patch_v1",
        "status": "suggestion_only",
        "safe_to_auto_apply": False,
        "source": "morpheme_v7_production_rule_exporter.py",
        "suggested_operator_weights": weights,
        "suggested_generator_adjustments": {
            "port_coupling": "0.80~0.88",
            "nested_enclosure_prob": "0.48~0.62",
            "orbit_weave_prob": "0.35~0.48",
            "interlock_prob": "0.50~0.65",
            "min_line_length": "96~110",
            "connected_components": "prefer 1, allow 2 only when attached by ports",
            "cycle_count": "prefer 1~3 for nested/orbit families",
        },
        "negative_contexts_to_penalize": [
            "bad_floating_or_multicomponent",
            "bad_trivial_e2e_chain",
            "bad_short_or_fragmented",
            "bad_low_context_cycle",
            "bad_no_structural_field",
        ],
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def markdown_report(candidates: List[Dict[str, Any]], decisions: List[ExportDecision], args: argparse.Namespace) -> str:
    lines: List[str] = []
    lines.append("# V7 Production Rule Export Report")
    lines.append("")
    lines.append(f"Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Source CSV: `{args.source_csv}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    accepted = [d for d in decisions if d.decision == "accepted"]
    rejected = [d for d in decisions if d.decision != "accepted"]
    lines.append(f"- Accepted candidates: **{len(accepted)}**")
    lines.append(f"- Rejected candidates: **{len(rejected)}**")
    lines.append(f"- Output status: **review_only**")
    lines.append("")
    fam_counts: Dict[str, int] = {}
    for c in candidates:
        fam = c.get("production_family_v7", "UNKNOWN")
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    lines.append("## Accepted production families")
    lines.append("")
    if fam_counts:
        for fam, cnt in sorted(fam_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- `{fam}`: {cnt}")
    else:
        lines.append("No candidates accepted under current filters.")
    lines.append("")
    lines.append("## Top accepted rules")
    lines.append("")
    for c in candidates[: min(12, len(candidates))]:
        s = c.get("support", {})
        lines.append(f"### {c.get('rule_id')}")
        lines.append(f"- Family: `{c.get('production_family_v7')}` / tier `{c.get('production_tier_v7')}`")
        lines.append(f"- Rule: `{c.get('rule_name')}`")
        lines.append(f"- Supports: good={s.get('good_support'):.4f}, bad={s.get('bad_support'):.4f}, cleaned={s.get('cleaned_support'):.4f}, manual={s.get('manual_support'):.4f}")
        lines.append(f"- Operator: `{c.get('production', {}).get('operator')}`")
        lines.append("")
    lines.append("## How to use")
    lines.append("")
    lines.append("1. Review generated examples visually before promotion.")
    lines.append("2. Promote only a small number of stable productions into the actual generator.")
    lines.append("3. Keep `bad_failure_rules` as negative contexts/penalties, not as positive production rules.")
    lines.append("4. Do not directly enable this review-only JSON as a hard rule cache.")
    lines.append("")
    return "\n".join(lines)



# --------------------------------------------------------------------------------------
# AB generation quality evaluator
# --------------------------------------------------------------------------------------

POINT_KEYS = ("x", "y")


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _point_from_any(obj: Any) -> Optional[Tuple[float, float]]:
    if isinstance(obj, dict):
        if "x" in obj and "y" in obj and _is_number(obj.get("x")) and _is_number(obj.get("y")):
            return (float(obj["x"]), float(obj["y"]))
        # common alternatives
        if "0" in obj and "1" in obj and _is_number(obj.get("0")) and _is_number(obj.get("1")):
            return (float(obj["0"]), float(obj["1"]))
    if isinstance(obj, (list, tuple)) and len(obj) >= 2 and _is_number(obj[0]) and _is_number(obj[1]):
        return (float(obj[0]), float(obj[1]))
    return None


def _points_from_any(obj: Any, depth: int = 0) -> List[Tuple[float, float]]:
    if depth > 8 or obj is None:
        return []
    pt = _point_from_any(obj)
    if pt is not None:
        return [pt]
    if isinstance(obj, dict):
        # Try likely geometry keys first.
        keys = [
            "mother_bezier", "bezier", "curve", "path", "control_points", "points", "polyline",
            "solved_bezier", "p", "P", "nodes", "coords", "geometry", "stroke", "segment",
        ]
        out: List[Tuple[float, float]] = []
        for k in keys:
            if k in obj:
                out.extend(_points_from_any(obj[k], depth + 1))
                if len(out) >= 2:
                    return out
        # Fallback: scan values but keep it shallow.
        for v in obj.values():
            out.extend(_points_from_any(v, depth + 1))
            if len(out) >= 4:
                return out
        return out
    if isinstance(obj, (list, tuple)):
        # Flat list [x0,y0,x1,y1,...]
        if len(obj) >= 4 and all(_is_number(x) for x in obj[: min(len(obj), 8)]):
            vals = [float(x) for x in obj]
            return list(zip(vals[0::2], vals[1::2]))
        out: List[Tuple[float, float]] = []
        for v in obj:
            out.extend(_points_from_any(v, depth + 1))
        return out
    return []


def _cubic_sample(points: List[Tuple[float, float]], n: int = 24) -> List[Tuple[float, float]]:
    if len(points) < 2:
        return points
    if len(points) >= 4:
        p0, p1, p2, p3 = points[0], points[1], points[2], points[3]
        out = []
        for i in range(n):
            t = i / max(1, n - 1)
            mt = 1.0 - t
            x = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
            y = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
            out.append((x, y))
        return out
    return points


def _polyline_length(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 2:
        return 0.0
    total = 0.0
    for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
        total += ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
    return total


def _stroke_length(stroke: Any) -> float:
    pts = _points_from_any(stroke)
    return _polyline_length(_cubic_sample(pts)) if pts else 0.0


def _stroke_angle(stroke: Any) -> Optional[float]:
    pts = _points_from_any(stroke)
    if len(pts) < 2:
        return None
    import math
    x1, y1 = pts[0]
    x2, y2 = pts[-1]
    if abs(x2 - x1) + abs(y2 - y1) < 1e-9:
        return None
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _stroke_curvature_proxy(stroke: Any) -> float:
    pts = _points_from_any(stroke)
    if len(pts) < 4:
        return 0.0
    chord = _polyline_length([pts[0], pts[-1]])
    curve = _polyline_length(_cubic_sample(pts))
    if curve <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, (curve - chord) / curve))


def _recursive_find_lists(obj: Any, keys: set, depth: int = 0) -> List[list]:
    if depth > 10:
        return []
    found: List[list] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, list):
                found.append(v)
        for v in obj.values():
            if isinstance(v, (dict, list)):
                found.extend(_recursive_find_lists(v, keys, depth + 1))
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                found.extend(_recursive_find_lists(v, keys, depth + 1))
    return found


def get_record_strokes(record: Any) -> List[Any]:
    keys = {"strokes", "stroke_list", "segments", "solved_segments", "edges", "solved_nodes", "nodes"}
    lists = _recursive_find_lists(record, keys)
    if not lists:
        return []
    # Pick the longest list with at least one geometry-like item.
    scored = []
    for lst in lists:
        geom_count = sum(1 for x in lst if len(_points_from_any(x)) >= 2)
        scored.append((geom_count, len(lst), lst))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2] if scored else []


def get_record_events(record: Any) -> List[Any]:
    keys = {"topology_events", "events", "relations", "topology_relations", "connections", "edges"}
    lists = _recursive_find_lists(record, keys)
    if not lists:
        return []
    # Prefer lists whose entries look like relation/event dicts.
    def eventish(x: Any) -> int:
        if not isinstance(x, dict):
            return 0
        blob = json.dumps(x, ensure_ascii=False).lower()
        return int(any(t in blob for t in ["e2e", "endpoint", "t", "x", "cross", "intersect", "relation", "type"]))
    lists.sort(key=lambda lst: (sum(eventish(x) for x in lst), len(lst)), reverse=True)
    return lists[0]


def get_record_cycles(record: Any) -> List[Any]:
    keys = {"cycles", "cycle_list", "rings", "loops"}
    lists = _recursive_find_lists(record, keys)
    if not lists:
        return []
    lists.sort(key=len, reverse=True)
    return lists[0]


def looks_like_sample_record(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    return bool(get_record_strokes(obj) or get_record_events(obj) or get_record_cycles(obj))


def flatten_sample_records(obj: Any, depth: int = 0) -> List[Dict[str, Any]]:
    if depth > 14:
        return []
    if isinstance(obj, dict):
        # Important: keep searching children; many files are wrappers.
        child_records: List[Dict[str, Any]] = []
        for k in ["records", "items", "data", "samples", "glyphs", "candidates", "results", "entries", "good", "bad", "cleaned", "manual"]:
            if k in obj:
                child_records.extend(flatten_sample_records(obj[k], depth + 1))
        for v in obj.values():
            if isinstance(v, (dict, list)):
                child_records.extend(flatten_sample_records(v, depth + 1))
        # If children found, prefer them. Otherwise current object is a record.
        if child_records:
            # de-dup by object id not available after recursion, keep simple.
            return child_records
        if looks_like_sample_record(obj):
            return [obj]
        return []
    if isinstance(obj, list):
        out: List[Dict[str, Any]] = []
        for v in obj:
            out.extend(flatten_sample_records(v, depth + 1))
        return out
    return []


def iter_json_sample_records(root: Path, max_files: int = 0, max_records: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root = Path(root).expanduser().resolve()
    files = sorted([p for p in root.rglob("*.json") if p.is_file()]) if root.exists() else []
    if max_files and max_files > 0:
        files = files[:max_files]
    records: List[Dict[str, Any]] = []
    bad_json = 0
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            recs = flatten_sample_records(obj)
            for r in recs:
                if isinstance(r, dict):
                    rr = dict(r)
                    rr["__source_file"] = str(fp)
                    records.append(rr)
                    if max_records and len(records) >= max_records:
                        return records, {"root": str(root), "files": len(files), "bad_json": bad_json, "truncated": True}
        except Exception:
            bad_json += 1
    return records, {"root": str(root), "files": len(files), "bad_json": bad_json, "truncated": False}


def _event_type(event: Any) -> str:
    if isinstance(event, dict):
        for k in ["type", "relation", "rel", "event_type", "topology_type", "kind", "name"]:
            if k in event:
                s = str(event.get(k, "")).upper()
                if "E2E" in s or "END" in s:
                    return "E2E"
                if s == "T" or "T_JUNCTION" in s or "TEE" in s:
                    return "T"
                if s == "X" or "CROSS" in s or "INTERSECT" in s:
                    return "X"
        blob = json.dumps(event, ensure_ascii=False).upper()
    else:
        blob = str(event).upper()
    if "E2E" in blob or "ENDPOINT" in blob:
        return "E2E"
    if "T_JUNCTION" in blob or " T" in blob or "\"T\"" in blob:
        return "T"
    if "CROSS" in blob or "INTERSECT" in blob or "\"X\"" in blob:
        return "X"
    return "OTHER"


def _extract_event_pair(event: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(event, dict):
        return None
    candidate_values: List[Any] = []
    for k in ["a", "b", "i", "j", "src", "dst", "source", "target", "stroke_a", "stroke_b", "stroke1", "stroke2", "u", "v", "from", "to", "host", "guest"]:
        if k in event:
            candidate_values.append(event[k])
    ints: List[int] = []
    for v in candidate_values:
        if isinstance(v, int):
            ints.append(v)
        elif isinstance(v, float) and v.is_integer():
            ints.append(int(v))
        elif isinstance(v, str):
            m = re.search(r"(\d+)", v)
            if m:
                ints.append(int(m.group(1)))
        elif isinstance(v, dict):
            for kk in ["id", "index", "stroke_id"]:
                if kk in v and _is_number(v[kk]):
                    ints.append(int(float(v[kk])))
    # unique but preserve order
    uniq = []
    for x in ints:
        if x not in uniq:
            uniq.append(x)
    if len(uniq) >= 2:
        return uniq[0], uniq[1]
    return None


def _connected_components_from_events(stroke_count: int, events: List[Any]) -> Tuple[int, int, int, float, float]:
    if stroke_count <= 0:
        return 0, 0, 0, 0.0, 0.0
    parent = list(range(stroke_count))
    deg = [0] * stroke_count
    branch_nodes = set()
    parsed_edges = 0
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for e in events:
        pair = _extract_event_pair(e)
        if not pair:
            continue
        a, b = pair
        if 0 <= a < stroke_count and 0 <= b < stroke_count and a != b:
            union(a, b)
            deg[a] += 1
            deg[b] += 1
            parsed_edges += 1
    comps = len({find(i) for i in range(stroke_count)}) if parsed_edges else max(1, stroke_count)
    max_degree = max(deg) if deg else 0
    leaf_count = sum(1 for d in deg if d <= 1)
    for i, d in enumerate(deg):
        if d >= 3:
            branch_nodes.add(i)
    branch_ratio = len(branch_nodes) / max(1, stroke_count)
    leaf_ratio = leaf_count / max(1, stroke_count)
    return comps, max_degree, parsed_edges, branch_ratio, leaf_ratio


def bucket_int(v: int, cap: int) -> int:
    return max(0, min(cap, int(v)))


def bucket_ratio(v: float, bins: int = 3) -> int:
    if v <= 0:
        return 0
    if v >= 1:
        return bins
    return max(0, min(bins, int(v * (bins + 1))))


def extract_eval_features(record: Dict[str, Any]) -> Dict[str, Any]:
    strokes = get_record_strokes(record)
    events = get_record_events(record)
    cycles = get_record_cycles(record)
    stroke_count = len(strokes)
    cycle_count = len(cycles)
    event_types = [_event_type(e) for e in events]
    E2E_count = sum(1 for t in event_types if t == "E2E")
    T_count = sum(1 for t in event_types if t == "T")
    X_count = sum(1 for t in event_types if t == "X")
    topology_event_count = len(events)
    connected_components, max_degree, parsed_edges, branch_ratio, leaf_ratio = _connected_components_from_events(stroke_count, events)

    lengths = [_stroke_length(s) for s in strokes]
    lengths = [x for x in lengths if x > 1e-9]
    min_length = min(lengths) if lengths else 0.0
    mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    angles = [_stroke_angle(s) for s in strokes]
    angles = [a for a in angles if a is not None]
    axis_count = 0
    horiz_count = 0
    vert_count = 0
    for a in angles:
        aa = abs(((a + 180) % 180) - 90)  # distance to vertical-ish after folding? rough
        # Better direct distances to 0/90/180.
        folded = abs(a) % 180
        dist_h = min(abs(folded - 0), abs(folded - 180))
        dist_v = abs(folded - 90)
        if dist_h <= 12:
            horiz_count += 1
            axis_count += 1
        elif dist_v <= 12:
            vert_count += 1
            axis_count += 1
    axis_ratio = axis_count / max(1, len(angles))
    horizontal_ratio = horiz_count / max(1, len(angles))
    vertical_ratio = vert_count / max(1, len(angles))
    diagonal_ratio = 1.0 - axis_ratio if angles else 0.0
    curvs = [_stroke_curvature_proxy(s) for s in strokes]
    mean_curvature = sum(curvs) / len(curvs) if curvs else 0.0
    curve_ratio = sum(1 for c in curvs if c > 0.03) / max(1, len(curvs))

    E2E_ratio = E2E_count / max(1, topology_event_count)
    T_ratio = T_count / max(1, topology_event_count)
    X_ratio = X_count / max(1, topology_event_count)
    port_count_proxy = E2E_count * 2 + T_count * 2 + X_count * 2 + max_degree
    spine_score = min(1.0, 0.16 * max_degree + 0.45 * vertical_ratio + 0.25 * branch_ratio)
    enclosure_score = min(1.0, 0.42 * cycle_count + 0.30 * (1.0 if cycle_count >= 1 else 0.0) + 0.10 * X_count)
    interlock_score = min(1.0, 0.22 * X_count + 0.16 * T_count + 0.05 * max_degree)
    orbit_score = min(1.0, 0.42 * cycle_count + 0.35 * enclosure_score + 0.03 * port_count_proxy)
    fragmentation_risk = 1.0 if (stroke_count > 0 and min_length < 46.58) else 0.0
    if stroke_count > 0 and min_length < 63.67:
        fragmentation_risk = max(fragmentation_risk, 0.5)

    feats: Dict[str, Any] = {
        "stroke_count": stroke_count,
        "connected_components": connected_components,
        "cycle_count": cycle_count,
        "topology_event_count": topology_event_count,
        "E2E_count": E2E_count,
        "T_count": T_count,
        "X_count": X_count,
        "E2E_ratio": E2E_ratio,
        "T_ratio": T_ratio,
        "X_ratio": X_ratio,
        "max_degree": max_degree,
        "branch_ratio": branch_ratio,
        "leaf_ratio": leaf_ratio,
        "min_length": min_length,
        "mean_length": mean_length,
        "axis_ratio": axis_ratio,
        "horizontal_ratio": horizontal_ratio,
        "vertical_ratio": vertical_ratio,
        "diagonal_ratio": diagonal_ratio,
        "curve_ratio": curve_ratio,
        "mean_curvature": mean_curvature,
        "port_count_proxy": port_count_proxy,
        "spine_score": spine_score,
        "enclosure_score": enclosure_score,
        "interlock_score": interlock_score,
        "orbit_score": orbit_score,
        "fragmentation_risk": fragmentation_risk,
        "parsed_edges": parsed_edges,
        "__source_file": record.get("__source_file", ""),
    }
    feats["motifs"] = derive_eval_motifs(feats)
    feats["bad_subtypes"] = derive_eval_bad_subtypes(feats)
    return feats


def derive_eval_motifs(f: Dict[str, Any]) -> List[str]:
    motifs: List[str] = []
    ebin = bucket_int(int(f.get("E2E_count", 0)), 4)
    tbin = bucket_int(int(f.get("T_count", 0)), 2)
    xbin = bucket_int(int(f.get("X_count", 0)), 2)
    cbin = bucket_int(int(f.get("cycle_count", 0)), 4)
    motifs.append(f"relhist:E{ebin}_T{tbin}_X{xbin}_C{cbin}")
    s = int(f.get("stroke_count", 0))
    sb = 0 if s <= 4 else (1 if s <= 7 else 2)
    cc = bucket_int(int(f.get("connected_components", 0)), 3)
    motifs.append(f"capacity:Sb{sb}_CC{cc}_C{cbin}")
    maxd = bucket_int(int(f.get("max_degree", 0)), 5)
    branchb = bucket_int(int(round(float(f.get("branch_ratio", 0.0)) * 8)), 4)
    portb = bucket_int(int(round(float(f.get("port_count_proxy", 0.0)) / 4)), 4)
    motifs.append(f"role:maxD{maxd}_B{branchb}_portb{portb}")
    axisb = bucket_ratio(float(f.get("axis_ratio", 0.0)), 3)
    diagb = bucket_ratio(float(f.get("diagonal_ratio", 0.0)), 3)
    curveb = bucket_ratio(float(f.get("curve_ratio", 0.0)), 3)
    motifs.append(f"orient:axisb{axisb}_diagb{diagb}_curveb{curveb}")
    spineb = bucket_ratio(float(f.get("spine_score", 0.0)), 3)
    enclb = bucket_ratio(float(f.get("enclosure_score", 0.0)), 3)
    interb = bucket_ratio(float(f.get("interlock_score", 0.0)), 2)
    motifs.append(f"field:spineb{spineb}_enclb{enclb}_interb{interb}")

    if f.get("orbit_score", 0.0) >= 0.225 or (f.get("cycle_count", 0) >= 1 and f.get("enclosure_score", 0) >= 0.3):
        motifs.append("proxy:orbit_or_seal_field")
    if f.get("cycle_count", 0) >= 2 or (f.get("cycle_count", 0) >= 1 and f.get("enclosure_score", 0) >= 0.6):
        motifs.append("proxy:nested_or_multi_cycle")
    if f.get("spine_score", 0.0) >= 0.254 and f.get("branch_ratio", 0.0) >= 0.2:
        motifs.append("proxy:vertical_spine_branching")
    if f.get("interlock_score", 0.0) >= 0.222 and f.get("port_count_proxy", 0) >= 8:
        motifs.append("proxy:interlocked_ports")
    if f.get("cycle_count", 0) >= 1 and f.get("spine_score", 0.0) >= 0.175 and f.get("port_count_proxy", 0) >= 8:
        motifs.append("proxy:ring_core_spine_ports")
    if f.get("connected_components", 1) > 1:
        motifs.append("proxy:multi_component_or_floating")
    if f.get("axis_ratio", 0.0) >= 0.45 and f.get("curve_ratio", 0.0) >= 0.25:
        motifs.append("proxy:axis_skeleton_with_curve_residual")
    return motifs


def derive_eval_bad_subtypes(f: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if f.get("connected_components", 0) > 1:
        out.append("bad_floating_or_multicomponent")
    if f.get("E2E_ratio", 0.0) >= 0.95 and f.get("T_count", 0) == 0 and f.get("X_count", 0) == 0 and f.get("cycle_count", 0) == 0:
        out.append("bad_trivial_e2e_chain")
    if f.get("fragmentation_risk", 0.0) >= 0.5:
        out.append("bad_short_or_fragmented")
    if f.get("connected_components", 0) >= 2 and f.get("leaf_ratio", 0.0) >= 0.5:
        out.append("bad_unbalanced_multicomponent")
    if f.get("cycle_count", 0) >= 1 and f.get("spine_score", 0.0) < 0.175 and f.get("interlock_score", 0.0) < 0.1:
        out.append("bad_low_context_cycle")
    if f.get("spine_score", 0.0) < 0.175 and f.get("enclosure_score", 0.0) < 0.1 and f.get("interlock_score", 0.0) < 0.1:
        out.append("bad_no_structural_field")
    if f.get("X_count", 0) >= 3 and f.get("fragmentation_risk", 0.0) >= 0.5:
        out.append("bad_overdense_cross_fragment")
    return out


def eval_term(term: str, feats: Dict[str, Any]) -> bool:
    term = term.strip()
    if term.startswith("motif:"):
        return term[len("motif:"):] in set(feats.get("motifs", []))
    m = _NOT_BAD_RE.match(term)
    if m:
        return m.group(1) not in set(feats.get("bad_subtypes", []))
    m = _NUMERIC_RE.match(term)
    if m:
        k, op, val = m.group(1), m.group(2), float(m.group(3))
        x = float(feats.get(k, 0.0))
        if op == ">=": return x >= val
        if op == "<=": return x <= val
        if op == ">": return x > val
        if op == "<": return x < val
        if op == "==": return abs(x - val) < 1e-9
    m = _BETWEEN_RE.match(term)
    if m:
        k, lo, hi = m.group(1), float(m.group(2)), float(m.group(3))
        x = float(feats.get(k, 0.0))
        return lo <= x <= hi
    return False


def eval_rule_name(rule_name: str, feats: Dict[str, Any]) -> bool:
    terms = split_rule_terms(rule_name)
    if not terms:
        return False
    return all(eval_term(t, feats) for t in terms)


def summarize_feature_rows(rows: List[Dict[str, Any]], candidate_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    def mean(k: str) -> float:
        return sum(float(r.get(k, 0.0)) for r in rows) / max(1, n)
    metrics = [
        "stroke_count", "connected_components", "cycle_count", "topology_event_count", "E2E_count", "T_count", "X_count",
        "E2E_ratio", "X_ratio", "branch_ratio", "leaf_ratio", "min_length", "mean_length", "axis_ratio", "diagonal_ratio",
        "curve_ratio", "spine_score", "enclosure_score", "interlock_score", "orbit_score", "fragmentation_risk", "port_count_proxy",
    ]
    metric_means = {k: mean(k) for k in metrics}
    bad_counts: Dict[str, int] = {}
    motif_counts: Dict[str, int] = {}
    signatures: Dict[str, int] = {}
    for r in rows:
        for b in r.get("bad_subtypes", []):
            bad_counts[b] = bad_counts.get(b, 0) + 1
        motifs = sorted(r.get("motifs", []))
        for m in motifs:
            motif_counts[m] = motif_counts.get(m, 0) + 1
        sig = "|".join(motifs[:8])
        signatures[sig] = signatures.get(sig, 0) + 1
    import math
    total_motifs = sum(motif_counts.values()) or 1
    motif_entropy = -sum((c/total_motifs) * math.log((c/total_motifs) + 1e-12) for c in motif_counts.values())
    rule_cov = []
    for c in candidate_rules:
        rn = c.get("rule_name", "")
        fam = c.get("production_family_v7", c.get("production", {}).get("operator", "UNKNOWN"))
        hits = sum(1 for r in rows if eval_rule_name(rn, r))
        rule_cov.append({
            "rule_id": c.get("rule_id", ""),
            "production_family_v7": fam,
            "rule_name": rn,
            "hits": hits,
            "support": hits / max(1, n),
        })
    fam_hits: Dict[str, int] = {}
    for r in rows:
        fam_hit_set = set()
        for c in candidate_rules:
            fam = c.get("production_family_v7", c.get("production", {}).get("operator", "UNKNOWN"))
            if eval_rule_name(c.get("rule_name", ""), r):
                fam_hit_set.add(fam)
        for fam in fam_hit_set:
            fam_hits[fam] = fam_hits.get(fam, 0) + 1
    return {
        "n": n,
        "metric_means": metric_means,
        "bad_subtype_rates": {k: v / max(1, n) for k, v in sorted(bad_counts.items())},
        "top_motif_support": [
            {"motif": k, "support": v / max(1, n), "hits": v}
            for k, v in sorted(motif_counts.items(), key=lambda x: (-x[1], x[0]))[:40]
        ],
        "production_rule_coverage": sorted(rule_cov, key=lambda x: (-x["support"], x["rule_id"])),
        "production_family_coverage": {k: v / max(1, n) for k, v in sorted(fam_hits.items())},
        "diversity": {
            "unique_signatures": len(signatures),
            "signature_uniqueness_ratio": len(signatures) / max(1, n),
            "motif_entropy": motif_entropy,
            "top_signature_rate": (max(signatures.values()) / max(1, n)) if signatures else 0.0,
        },
    }


def compare_summaries(base: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"metric_delta": {}, "bad_subtype_delta": {}, "production_family_delta": {}, "recommendations": []}
    keys = sorted(set(base.get("metric_means", {})) | set(exp.get("metric_means", {})))
    for k in keys:
        b = float(base.get("metric_means", {}).get(k, 0.0))
        e = float(exp.get("metric_means", {}).get(k, 0.0))
        out["metric_delta"][k] = {"baseline": b, "experiment": e, "delta": e-b, "relative_delta": (e-b)/abs(b) if abs(b)>1e-9 else None}
    keys = sorted(set(base.get("bad_subtype_rates", {})) | set(exp.get("bad_subtype_rates", {})))
    for k in keys:
        b = float(base.get("bad_subtype_rates", {}).get(k, 0.0))
        e = float(exp.get("bad_subtype_rates", {}).get(k, 0.0))
        out["bad_subtype_delta"][k] = {"baseline": b, "experiment": e, "delta": e-b, "relative_delta": (e-b)/abs(b) if abs(b)>1e-9 else None}
    keys = sorted(set(base.get("production_family_coverage", {})) | set(exp.get("production_family_coverage", {})))
    for k in keys:
        b = float(base.get("production_family_coverage", {}).get(k, 0.0))
        e = float(exp.get("production_family_coverage", {}).get(k, 0.0))
        out["production_family_delta"][k] = {"baseline": b, "experiment": e, "delta": e-b, "relative_delta": (e-b)/abs(b) if abs(b)>1e-9 else None}

    rec = out["recommendations"]
    for bad in ["bad_floating_or_multicomponent", "bad_trivial_e2e_chain", "bad_short_or_fragmented", "bad_low_context_cycle"]:
        d = out["bad_subtype_delta"].get(bad, {}).get("delta")
        if d is not None:
            if d <= -0.05:
                rec.append(f"PASS: {bad} dropped by {abs(d):.2%}.")
            elif d > 0.03:
                rec.append(f"WARN: {bad} increased by {d:.2%}; reduce production influence or add penalty.")
    for fam in ["NESTED_ORBIT_SEAL", "NESTED_PORTAL", "RING_CORE_SPINE_PORT", "INTERLOCKED_PORT_MOTIF"]:
        d = out["production_family_delta"].get(fam, {}).get("delta")
        if d is not None and d > 0.05:
            rec.append(f"PASS: {fam} coverage increased by {d:.2%}.")
    top_sig_base = base.get("diversity", {}).get("top_signature_rate", 0.0)
    top_sig_exp = exp.get("diversity", {}).get("top_signature_rate", 0.0)
    if top_sig_exp - top_sig_base > 0.10:
        rec.append("WARN: top_signature_rate increased a lot; possible mode collapse.")
    if not rec:
        rec.append("No strong automatic conclusion; inspect JSON/CSV deltas and consider larger sample size.")
    return out


def write_ab_markdown(path: Path, base_info: Dict[str, Any], exp_info: Dict[str, Any], base: Dict[str, Any], exp: Dict[str, Any], comp: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# AB Generation Quality Evaluation")
    lines.append("")
    lines.append(f"Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Baseline root: `{base_info.get('root')}`")
    lines.append(f"- Experiment root: `{exp_info.get('root')}`")
    lines.append(f"- Baseline samples: **{base.get('n',0)}**")
    lines.append(f"- Experiment samples: **{exp.get('n',0)}**")
    lines.append("")
    lines.append("## Bad subtype rates")
    lines.append("")
    lines.append("| subtype | baseline | experiment | delta |")
    lines.append("|---|---:|---:|---:|")
    for k, v in comp.get("bad_subtype_delta", {}).items():
        lines.append(f"| {k} | {v['baseline']:.4f} | {v['experiment']:.4f} | {v['delta']:+.4f} |")
    lines.append("")
    lines.append("## Production family coverage")
    lines.append("")
    lines.append("| family | baseline | experiment | delta |")
    lines.append("|---|---:|---:|---:|")
    for k, v in comp.get("production_family_delta", {}).items():
        lines.append(f"| {k} | {v['baseline']:.4f} | {v['experiment']:.4f} | {v['delta']:+.4f} |")
    lines.append("")
    lines.append("## Key metric deltas")
    lines.append("")
    for k in ["connected_components", "min_length", "mean_length", "cycle_count", "X_count", "interlock_score", "orbit_score", "fragmentation_risk"]:
        v = comp.get("metric_delta", {}).get(k)
        if v:
            lines.append(f"- `{k}`: {v['baseline']:.4f} -> {v['experiment']:.4f} ({v['delta']:+.4f})")
    lines.append("")
    lines.append("## Diversity")
    lines.append(f"- baseline unique_signature_ratio: {base.get('diversity',{}).get('signature_uniqueness_ratio',0):.4f}")
    lines.append(f"- experiment unique_signature_ratio: {exp.get('diversity',{}).get('signature_uniqueness_ratio',0):.4f}")
    lines.append(f"- baseline motif_entropy: {base.get('diversity',{}).get('motif_entropy',0):.4f}")
    lines.append(f"- experiment motif_entropy: {exp.get('diversity',{}).get('motif_entropy',0):.4f}")
    lines.append("")
    lines.append("## Automatic recommendations")
    for r in comp.get("recommendations", []):
        lines.append(f"- {r}")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ab_evaluator(args: argparse.Namespace, candidates: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    if not args.eval_ab:
        return None
    if not args.baseline_dir or not args.experiment_dir:
        raise ValueError("--eval-ab requires --baseline-dir and --experiment-dir")
    eval_out = Path(args.eval_out_dir or (Path(args.out_dir).expanduser().resolve() / "ab_eval_outputs")).expanduser().resolve()
    eval_out.mkdir(parents=True, exist_ok=True)
    if candidates is None:
        candidate_json = Path(args.candidate_json).expanduser().resolve() if args.candidate_json else (Path(args.out_dir).expanduser().resolve() / "production_rule_candidates_review.json")
        if candidate_json.exists():
            candidates = json.loads(candidate_json.read_text(encoding="utf-8"))
        else:
            candidates = []

    base_records, base_info = iter_json_sample_records(Path(args.baseline_dir), args.max_eval_files, args.max_eval_records)
    exp_records, exp_info = iter_json_sample_records(Path(args.experiment_dir), args.max_eval_files, args.max_eval_records)
    base_feats = [extract_eval_features(r) for r in base_records]
    exp_feats = [extract_eval_features(r) for r in exp_records]
    base_sum = summarize_feature_rows(base_feats, candidates or [])
    exp_sum = summarize_feature_rows(exp_feats, candidates or [])
    comp = compare_summaries(base_sum, exp_sum)
    result = {
        "schema_version": "ab_generation_quality_eval_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_info": base_info,
        "experiment_info": exp_info,
        "baseline_summary": base_sum,
        "experiment_summary": exp_sum,
        "comparison": comp,
        "notes": [
            "This evaluator is automatic and topology-feature based; it does not require manual sample screening.",
            "It uses robust local parsing of generated JSON samples and V7 production rule predicates.",
            "Use the report to decide whether to raise/lower production_rule_influence or specific operator weights.",
        ],
    }
    write_json(eval_out / "ab_generation_quality_eval_summary.json", result)
    write_csv(eval_out / "ab_metric_delta.csv", [{"metric": k, **v} for k, v in comp.get("metric_delta", {}).items()])
    write_csv(eval_out / "ab_bad_subtype_delta.csv", [{"subtype": k, **v} for k, v in comp.get("bad_subtype_delta", {}).items()])
    write_csv(eval_out / "ab_production_family_delta.csv", [{"family": k, **v} for k, v in comp.get("production_family_delta", {}).items()])
    write_csv(eval_out / "baseline_rule_coverage.csv", base_sum.get("production_rule_coverage", []))
    write_csv(eval_out / "experiment_rule_coverage.csv", exp_sum.get("production_rule_coverage", []))
    write_ab_markdown(eval_out / "AB_EVAL_REPORT.md", base_info, exp_info, base_sum, exp_sum, comp)

    print("")
    print("AB Evaluation:")
    print(f"  Baseline samples      = {base_sum.get('n', 0)} from {base_info.get('root')}")
    print(f"  Experiment samples    = {exp_sum.get('n', 0)} from {exp_info.get('root')}")
    print(f"  Eval out dir          = {eval_out}")
    print("  Wrote:")
    print(f"    {eval_out / 'ab_generation_quality_eval_summary.json'}")
    print(f"    {eval_out / 'AB_EVAL_REPORT.md'}")
    print(f"    {eval_out / 'ab_metric_delta.csv'}")
    print(f"    {eval_out / 'ab_bad_subtype_delta.csv'}")
    print(f"    {eval_out / 'ab_production_family_delta.csv'}")
    print("  Recommendations:")
    for r in comp.get("recommendations", [])[:8]:
        print(f"    - {r}")
    return result


# --------------------------------------------------------------------------------------
# In-memory generation + AB evaluation
# --------------------------------------------------------------------------------------

def _mk_stroke(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], sid: int) -> Dict[str, Any]:
    return {
        "id": sid,
        "stroke_id": sid,
        "mother_bezier": [[round(p0[0], 4), round(p0[1], 4)], [round(p1[0], 4), round(p1[1], 4)], [round(p2[0], 4), round(p2[1], 4)], [round(p3[0], 4), round(p3[1], 4)]],
        "width": 6.0,
    }


def _add_event(events: List[Dict[str, Any]], typ: str, a: int, b: int) -> None:
    events.append({"type": typ, "stroke_a": int(a), "stroke_b": int(b)})


def _random_curve(rng: Any, sid: int, cx: float = 128.0, cy: float = 128.0, spread: float = 100.0, curved: bool = True) -> Dict[str, Any]:
    import math
    ang = rng.uniform(0, 2 * math.pi)
    length = rng.uniform(48, 145)
    x0 = cx + rng.uniform(-spread, spread) * 0.55
    y0 = cy + rng.uniform(-spread, spread) * 0.55
    x3 = x0 + math.cos(ang) * length
    y3 = y0 + math.sin(ang) * length
    bend = rng.uniform(-0.35, 0.35) * length if curved else 0.0
    nx, ny = -math.sin(ang), math.cos(ang)
    p0 = (x0, y0)
    p3 = (x3, y3)
    p1 = (x0 + math.cos(ang) * length * 0.33 + nx * bend, y0 + math.sin(ang) * length * 0.33 + ny * bend)
    p2 = (x0 + math.cos(ang) * length * 0.66 - nx * bend * 0.5, y0 + math.sin(ang) * length * 0.66 - ny * bend * 0.5)
    return _mk_stroke(p0, p1, p2, p3, sid)


def _cycle_arc_strokes(rng: Any, start_sid: int, cx: float, cy: float, r: float, count: int, jitter: float = 0.06) -> List[Dict[str, Any]]:
    import math
    strokes: List[Dict[str, Any]] = []
    k = 0.5522847498
    for i in range(count):
        a0 = 2 * math.pi * i / count + rng.uniform(-jitter, jitter)
        a1 = 2 * math.pi * (i + 1) / count + rng.uniform(-jitter, jitter)
        da = a1 - a0
        rr0 = r * rng.uniform(0.92, 1.08)
        rr1 = r * rng.uniform(0.92, 1.08)
        p0 = (cx + rr0 * math.cos(a0), cy + rr0 * math.sin(a0))
        p3 = (cx + rr1 * math.cos(a1), cy + rr1 * math.sin(a1))
        tangent0 = (-math.sin(a0), math.cos(a0))
        tangent1 = (-math.sin(a1), math.cos(a1))
        h = k * r * da / (math.pi / 2)
        p1 = (p0[0] + tangent0[0] * h, p0[1] + tangent0[1] * h)
        p2 = (p3[0] - tangent1[0] * h, p3[1] - tangent1[1] * h)
        strokes.append(_mk_stroke(p0, p1, p2, p3, start_sid + i))
    return strokes


def generate_baseline_memory_record(rng: Any, idx: int) -> Dict[str, Any]:
    """A lightweight baseline generator used only for in-memory AB evaluation.

    It mimics the broad current PCG failure distribution: mixed chains, some floating
    components, occasional cycles, and variable stroke length. It does not write samples.
    """
    stroke_count = rng.randint(4, 9)
    strokes = [_random_curve(rng, i, curved=(rng.random() < 0.65)) for i in range(stroke_count)]
    events: List[Dict[str, Any]] = []
    cycles: List[Dict[str, Any]] = []
    # Chain-like E2E backbone, often incomplete.
    for i in range(stroke_count - 1):
        if rng.random() < 0.62:
            _add_event(events, "E2E", i, i + 1)
    # Some crosses/intersections.
    for _ in range(rng.randint(0, 2)):
        a, b = rng.sample(range(stroke_count), 2)
        _add_event(events, "X" if rng.random() < 0.65 else "T", a, b)
    # Occasional low-context cycle.
    if rng.random() < 0.32 and stroke_count >= 4:
        cycles.append({"cycle_id": 0, "members": list(range(min(4, stroke_count)))})
    # Introduce fragmentation sometimes.
    if rng.random() < 0.25:
        sid = rng.randrange(stroke_count)
        strokes[sid] = _random_curve(rng, sid, spread=70, curved=False)
        # shrink it
        pts = strokes[sid]["mother_bezier"]
        x0, y0 = pts[0]
        x3, y3 = pts[-1]
        mx, my = (x0 + x3) / 2, (y0 + y3) / 2
        for p in pts:
            p[0] = round(mx + (p[0] - mx) * 0.35, 4)
            p[1] = round(my + (p[1] - my) * 0.35, 4)
    return {"sample_id": f"baseline_mem_{idx:05d}", "strokes": strokes, "topology_events": events, "cycles": cycles, "generator_family": "BASELINE_MEMORY"}


def generate_nested_portal_memory_record(rng: Any, idx: int, orbit: bool = False) -> Dict[str, Any]:
    strokes: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    cycles: List[Dict[str, Any]] = []
    cx, cy = 128.0 + rng.uniform(-8, 8), 128.0 + rng.uniform(-8, 8)
    outer_count = rng.choice([4, 5])
    inner_count = rng.choice([3, 4])
    strokes.extend(_cycle_arc_strokes(rng, 0, cx, cy, rng.uniform(58, 78), outer_count))
    offset = rng.uniform(-10, 10)
    strokes.extend(_cycle_arc_strokes(rng, len(strokes), cx + offset, cy - offset * 0.4, rng.uniform(28, 42), inner_count))
    # Connect cycle arcs.
    for i in range(outer_count):
        _add_event(events, "E2E", i, (i + 1) % outer_count)
    base_inner = outer_count
    for i in range(inner_count):
        _add_event(events, "E2E", base_inner + i, base_inner + ((i + 1) % inner_count))
    cycles.append({"cycle_id": 0, "members": list(range(outer_count))})
    cycles.append({"cycle_id": 1, "members": list(range(base_inner, base_inner + inner_count))})
    # Add core spine and ports.
    sid = len(strokes)
    spine_len = rng.uniform(95, 145)
    spine = _mk_stroke((cx, cy - spine_len/2), (cx + rng.uniform(-10, 10), cy - spine_len/6), (cx + rng.uniform(-10, 10), cy + spine_len/6), (cx, cy + spine_len/2), sid)
    strokes.append(spine)
    _add_event(events, "X", sid, rng.randrange(base_inner, base_inner + inner_count))
    port_n = rng.choice([1, 2, 3]) if not orbit else rng.choice([2, 3, 4])
    for k in range(port_n):
        sid = len(strokes)
        side = -1 if k % 2 == 0 else 1
        y = cy + rng.uniform(-45, 45)
        length = rng.uniform(45, 75)
        st = _mk_stroke((cx, y), (cx + side*length*0.35, y + rng.uniform(-8, 8)), (cx + side*length*0.65, y + rng.uniform(-8, 8)), (cx + side*length, y + rng.uniform(-15, 15)), sid)
        strokes.append(st)
        _add_event(events, "T" if rng.random() < 0.55 else "E2E", sid, len(strokes)-2 if sid > 0 else 0)
    if orbit:
        # Add an orbit chord / latch.
        sid = len(strokes)
        st = _mk_stroke((cx - 75, cy - 18), (cx - 30, cy - 70), (cx + 40, cy + 70), (cx + 78, cy + 20), sid)
        strokes.append(st)
        _add_event(events, "X", sid, 0)
        _add_event(events, "X", sid, base_inner)
    fam = "NESTED_ORBIT_SEAL" if orbit else "NESTED_PORTAL"
    return {"sample_id": f"experiment_mem_{fam.lower()}_{idx:05d}", "strokes": strokes, "topology_events": events, "cycles": cycles, "generator_family": fam}


def generate_interlock_memory_record(rng: Any, idx: int) -> Dict[str, Any]:
    # A compact interlocked port motif: no full nested portal but more T/X relations.
    strokes: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    cx, cy = 128.0, 128.0
    for i in range(rng.randint(5, 8)):
        strokes.append(_random_curve(rng, i, cx=cx, cy=cy, spread=75, curved=True))
    for i in range(len(strokes)-1):
        _add_event(events, "E2E", i, i+1)
    for _ in range(rng.randint(2, 4)):
        a, b = rng.sample(range(len(strokes)), 2)
        _add_event(events, "X" if rng.random() < 0.7 else "T", a, b)
    cycles = [{"cycle_id": 0, "members": list(range(min(4, len(strokes))))}] if rng.random() < 0.45 else []
    return {"sample_id": f"experiment_mem_interlock_{idx:05d}", "strokes": strokes, "topology_events": events, "cycles": cycles, "generator_family": "INTERLOCKED_PORT_MOTIF"}


def generate_experiment_memory_record(rng: Any, idx: int, candidates: List[Dict[str, Any]], production_weight: float = 0.75) -> Dict[str, Any]:
    fams = [str(c.get("production_family_v7", "")) for c in candidates]
    # Prefer explicit production-ready families from the exported candidates.
    if rng.random() > production_weight or not fams:
        return generate_baseline_memory_record(rng, idx)
    # Weighted by frequency in candidates; fallback to nested portal if available.
    counts: Dict[str, int] = {}
    for f in fams:
        counts[f] = counts.get(f, 0) + 1
    choices = []
    for f, c in counts.items():
        choices.extend([f] * max(1, c))
    fam = rng.choice(choices)
    if fam == "NESTED_ORBIT_SEAL":
        return generate_nested_portal_memory_record(rng, idx, orbit=True)
    if fam == "NESTED_PORTAL":
        return generate_nested_portal_memory_record(rng, idx, orbit=False)
    if fam in {"INTERLOCKED_PORT_MOTIF", "ORBIT_INTERLOCK_SEAL"}:
        return generate_interlock_memory_record(rng, idx)
    if fam == "RING_CORE_SPINE_PORT":
        return generate_nested_portal_memory_record(rng, idx, orbit=True)
    # DISCOVERED_TOPOLOGY_MOTIF or unknown: mixed production-biased record.
    return generate_nested_portal_memory_record(rng, idx, orbit=(rng.random() < 0.55))


def run_memory_generation_evaluator(args: argparse.Namespace, candidates: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    if not getattr(args, "eval_generate_memory", False):
        return None
    import random
    eval_out = Path(args.eval_out_dir or (Path(args.out_dir).expanduser().resolve() / "memory_ab_eval_outputs")).expanduser().resolve()
    eval_out.mkdir(parents=True, exist_ok=True)
    if candidates is None:
        candidate_json = Path(args.candidate_json).expanduser().resolve() if args.candidate_json else (Path(args.out_dir).expanduser().resolve() / "production_rule_candidates_review.json")
        if candidate_json.exists():
            candidates = json.loads(candidate_json.read_text(encoding="utf-8"))
        else:
            candidates = []
    n = int(getattr(args, "generated_samples", 500) or 500)
    seed = int(getattr(args, "generation_seed", 42) or 42)
    production_weight = float(getattr(args, "memory_production_weight", 0.75) or 0.75)
    rng_base = random.Random(seed)
    rng_exp = random.Random(seed + 100003)
    base_records = [generate_baseline_memory_record(rng_base, i) for i in range(n)]
    exp_records = [generate_experiment_memory_record(rng_exp, i, candidates or [], production_weight=production_weight) for i in range(n)]
    base_feats = [extract_eval_features(r) for r in base_records]
    exp_feats = [extract_eval_features(r) for r in exp_records]
    base_sum = summarize_feature_rows(base_feats, candidates or [])
    exp_sum = summarize_feature_rows(exp_feats, candidates or [])
    comp = compare_summaries(base_sum, exp_sum)
    base_info = {"root": "<in_memory_baseline_generator>", "files": 0, "bad_json": 0, "truncated": False, "generated_in_memory": True, "samples": n, "seed": seed}
    exp_info = {"root": "<in_memory_production_generator>", "files": 0, "bad_json": 0, "truncated": False, "generated_in_memory": True, "samples": n, "seed": seed + 100003, "production_weight": production_weight}
    result = {
        "schema_version": "memory_generation_quality_eval_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_info": base_info,
        "experiment_info": exp_info,
        "baseline_summary": base_sum,
        "experiment_summary": exp_sum,
        "comparison": comp,
        "notes": [
            "This mode generates temporary sample records in memory and does not write generated samples to local storage.",
            "It is a production-blueprint smoke test, not a replacement for evaluating the real GUI/PCG generator once integrated.",
            "Use this to catch obvious directionality issues before wiring productions into the full generator.",
        ],
    }
    write_json(eval_out / "memory_generation_quality_eval_summary.json", result)
    write_csv(eval_out / "memory_metric_delta.csv", [{"metric": k, **v} for k, v in comp.get("metric_delta", {}).items()])
    write_csv(eval_out / "memory_bad_subtype_delta.csv", [{"subtype": k, **v} for k, v in comp.get("bad_subtype_delta", {}).items()])
    write_csv(eval_out / "memory_production_family_delta.csv", [{"family": k, **v} for k, v in comp.get("production_family_delta", {}).items()])
    write_csv(eval_out / "memory_baseline_rule_coverage.csv", base_sum.get("production_rule_coverage", []))
    write_csv(eval_out / "memory_experiment_rule_coverage.csv", exp_sum.get("production_rule_coverage", []))
    write_ab_markdown(eval_out / "MEMORY_AB_EVAL_REPORT.md", base_info, exp_info, base_sum, exp_sum, comp)
    print("")
    print("In-memory generation AB Evaluation:")
    print(f"  Generated baseline samples   = {base_sum.get('n', 0)}  (not written to disk)")
    print(f"  Generated experiment samples = {exp_sum.get('n', 0)}  (not written to disk)")
    print(f"  Production weight            = {production_weight:.3f}")
    print(f"  Eval out dir                 = {eval_out}")
    print("  Wrote summary/report only:")
    print(f"    {eval_out / 'memory_generation_quality_eval_summary.json'}")
    print(f"    {eval_out / 'MEMORY_AB_EVAL_REPORT.md'}")
    print(f"    {eval_out / 'memory_metric_delta.csv'}")
    print(f"    {eval_out / 'memory_bad_subtype_delta.csv'}")
    print(f"    {eval_out / 'memory_production_family_delta.csv'}")
    print("  Recommendations:")
    for r in comp.get("recommendations", [])[:8]:
        print(f"    - {r}")
    return result



# --------------------------------------------------------------------------------------
# Feedback judgement chain: export -> generate in memory -> diagnose -> decide -> patch
# --------------------------------------------------------------------------------------

FEEDBACK_POLICIES: List[Dict[str, Any]] = [
    {
        "policy_id": "P0_conservative_safe_nested",
        "production_weight": 0.25,
        "family_weights": {"NESTED_ORBIT_SEAL": 1.0, "NESTED_PORTAL": 1.2, "RING_CORE_SPINE_PORT": 0.6, "INTERLOCKED_PORT_MOTIF": 0.4, "DISCOVERED_TOPOLOGY_MOTIF": 0.15},
        "short_fragment_guard": True,
        "max_attempts": 4,
        "description": "Low influence; production families are sampled but short-fragment candidates are resampled in memory.",
    },
    {
        "policy_id": "P1_balanced_guarded_production",
        "production_weight": 0.40,
        "family_weights": {"NESTED_ORBIT_SEAL": 1.0, "NESTED_PORTAL": 1.0, "RING_CORE_SPINE_PORT": 0.75, "INTERLOCKED_PORT_MOTIF": 0.55, "DISCOVERED_TOPOLOGY_MOTIF": 0.15},
        "short_fragment_guard": True,
        "max_attempts": 5,
        "description": "Balanced influence with family balancing and short-fragment guard.",
    },
    {
        "policy_id": "P2_diversity_guarded_mix",
        "production_weight": 0.50,
        "family_weights": {"NESTED_ORBIT_SEAL": 0.85, "NESTED_PORTAL": 0.85, "RING_CORE_SPINE_PORT": 0.85, "INTERLOCKED_PORT_MOTIF": 0.75, "DISCOVERED_TOPOLOGY_MOTIF": 0.10},
        "short_fragment_guard": True,
        "max_attempts": 6,
        "description": "Medium influence; more family-balanced to reduce nested/orbit overconcentration.",
    },
    {
        "policy_id": "P3_aggressive_without_guard_probe",
        "production_weight": 0.75,
        "family_weights": {"NESTED_ORBIT_SEAL": 1.0, "NESTED_PORTAL": 1.0, "RING_CORE_SPINE_PORT": 1.0, "INTERLOCKED_PORT_MOTIF": 1.0, "DISCOVERED_TOPOLOGY_MOTIF": 0.25},
        "short_fragment_guard": False,
        "max_attempts": 1,
        "description": "Aggressive probe; expected to expose fragmentation/collapse risk. Not intended as final config.",
    },
]


def _weighted_choice_from_pairs(rng: Any, pairs: List[Tuple[str, float]]) -> str:
    pairs = [(k, max(0.0, float(w))) for k, w in pairs if float(w) > 0]
    if not pairs:
        return "BASELINE"
    total = sum(w for _, w in pairs)
    x = rng.random() * total
    acc = 0.0
    for k, w in pairs:
        acc += w
        if x <= acc:
            return k
    return pairs[-1][0]


def generate_feedback_policy_record(rng: Any, idx: int, candidates: List[Dict[str, Any]], policy: Dict[str, Any]) -> Dict[str, Any]:
    """Generate one temporary sample according to an internal feedback policy.

    This is intentionally in-memory only. It is not a replacement for the full GUI/PCG
    generator; it is a smoke-test generator that tells whether a production family tends
    to push the topology metrics in the right direction.
    """
    production_weight = float(policy.get("production_weight", 0.4))
    if rng.random() > production_weight or not candidates:
        return generate_baseline_memory_record(rng, idx)

    fam_weights = dict(policy.get("family_weights", {}))
    candidate_family_counts: Dict[str, int] = {}
    for c in candidates:
        fam = str(c.get("production_family_v7", "") or c.get("production", {}).get("operator", ""))
        if not fam:
            continue
        candidate_family_counts[fam] = candidate_family_counts.get(fam, 0) + 1

    weighted_pairs: List[Tuple[str, float]] = []
    for fam, count in candidate_family_counts.items():
        # Count reflects discovered support; policy weight prevents one broad family from
        # dominating the feedback loop.
        weighted_pairs.append((fam, float(count) * float(fam_weights.get(fam, 0.35))))
    fam = _weighted_choice_from_pairs(rng, weighted_pairs)

    if fam == "NESTED_ORBIT_SEAL":
        return generate_nested_portal_memory_record(rng, idx, orbit=True)
    if fam == "NESTED_PORTAL":
        return generate_nested_portal_memory_record(rng, idx, orbit=False)
    if fam in {"INTERLOCKED_PORT_MOTIF", "ORBIT_INTERLOCK_SEAL"}:
        return generate_interlock_memory_record(rng, idx)
    if fam == "RING_CORE_SPINE_PORT":
        rec = generate_nested_portal_memory_record(rng, idx, orbit=True)
        rec["generator_family"] = "RING_CORE_SPINE_PORT"
        return rec
    rec = generate_nested_portal_memory_record(rng, idx, orbit=(rng.random() < 0.4))
    rec["generator_family"] = "DISCOVERED_TOPOLOGY_MOTIF"
    return rec


def generate_guarded_feedback_record(rng: Any, idx: int, candidates: List[Dict[str, Any]], policy: Dict[str, Any]) -> Dict[str, Any]:
    attempts = max(1, int(policy.get("max_attempts", 1)))
    guard = bool(policy.get("short_fragment_guard", False))
    best_rec: Optional[Dict[str, Any]] = None
    best_risk = 10**9
    for _ in range(attempts):
        rec = generate_feedback_policy_record(rng, idx, candidates, policy)
        if not guard:
            return rec
        feats = extract_eval_features(rec)
        bads = set(feats.get("bad_subtypes", []))
        risk = 0
        risk += 5 if "bad_short_or_fragmented" in bads else 0
        risk += 3 if "bad_floating_or_multicomponent" in bads else 0
        risk += 2 if "bad_low_context_cycle" in bads else 0
        risk += max(0, int(feats.get("connected_components", 1)) - 1)
        if risk < best_risk:
            best_rec, best_risk = rec, risk
        if risk == 0:
            return rec
    return best_rec if best_rec is not None else generate_baseline_memory_record(rng, idx)


def judge_feedback_policy(base_sum: Dict[str, Any], exp_sum: Dict[str, Any], comp: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    def delta_bad(name: str) -> float:
        return float(comp.get("bad_subtype_delta", {}).get(name, {}).get("delta", 0.0))
    def delta_fam(name: str) -> float:
        return float(comp.get("production_family_delta", {}).get(name, {}).get("delta", 0.0))

    floating_drop = -delta_bad("bad_floating_or_multicomponent")
    trivial_drop = -delta_bad("bad_trivial_e2e_chain")
    short_increase = delta_bad("bad_short_or_fragmented")
    low_cycle_increase = delta_bad("bad_low_context_cycle")
    nested_gain = max(0.0, delta_fam("NESTED_PORTAL"))
    seal_gain = max(0.0, delta_fam("NESTED_ORBIT_SEAL"))
    ring_gain = max(0.0, delta_fam("RING_CORE_SPINE_PORT"))
    interlock_gain = max(0.0, delta_fam("INTERLOCKED_PORT_MOTIF"))

    base_div = base_sum.get("diversity", {})
    exp_div = exp_sum.get("diversity", {})
    top_sig_delta = float(exp_div.get("top_signature_rate", 0.0)) - float(base_div.get("top_signature_rate", 0.0))
    uniq_delta = float(exp_div.get("signature_uniqueness_ratio", 0.0)) - float(base_div.get("signature_uniqueness_ratio", 0.0))
    entropy_delta = float(exp_div.get("motif_entropy", 0.0)) - float(base_div.get("motif_entropy", 0.0))

    target_gain = seal_gain + nested_gain + 0.65 * ring_gain + 0.65 * interlock_gain
    bad_reduction = 0.9 * floating_drop + 1.0 * trivial_drop - 1.25 * max(0.0, short_increase) - 0.8 * max(0.0, low_cycle_increase)
    diversity_term = 0.15 * max(0.0, entropy_delta) + 0.35 * max(0.0, uniq_delta) - 0.90 * max(0.0, top_sig_delta - 0.05)
    # Keep this score interpretable. Above 0.12 is a promising automatic direction; above
    # 0.22 with no major risks is production-ready for the generator control patch.
    feedback_score = target_gain + bad_reduction + diversity_term

    risks: List[str] = []
    passes: List[str] = []
    if floating_drop >= 0.03:
        passes.append(f"floating/multicomponent dropped by {floating_drop:.2%}")
    if trivial_drop >= 0.05:
        passes.append(f"trivial E2E chain dropped by {trivial_drop:.2%}")
    if target_gain >= 0.05:
        passes.append(f"target production coverage gained {target_gain:.2%}")
    if short_increase > 0.03:
        risks.append(f"short/fragmented increased by {short_increase:.2%}")
    if low_cycle_increase > 0.03:
        risks.append(f"low-context cycle increased by {low_cycle_increase:.2%}")
    if top_sig_delta > 0.10:
        risks.append(f"top signature rate increased by {top_sig_delta:.2%}; possible mode collapse")
    if uniq_delta < -0.08:
        risks.append(f"signature uniqueness dropped by {abs(uniq_delta):.2%}")

    if feedback_score >= 0.22 and not risks:
        verdict = "ACCEPT_PRODUCTION_PATCH"
    elif feedback_score >= 0.12 and (short_increase <= 0.06) and (top_sig_delta <= 0.14):
        verdict = "ACCEPT_WITH_GUARDS"
    elif target_gain > 0.04 and (floating_drop > 0.0 or trivial_drop > 0.0):
        verdict = "PROMISING_BUT_NEEDS_CONSTRAINTS"
    else:
        verdict = "REJECT_OR_KEEP_EXPLORATORY"

    return {
        "policy_id": policy.get("policy_id"),
        "verdict": verdict,
        "feedback_score": feedback_score,
        "target_gain": target_gain,
        "bad_reduction_score": bad_reduction,
        "diversity_score": diversity_term,
        "floating_drop": floating_drop,
        "trivial_e2e_drop": trivial_drop,
        "short_fragment_increase": short_increase,
        "low_context_cycle_increase": low_cycle_increase,
        "top_signature_rate_delta": top_sig_delta,
        "signature_uniqueness_delta": uniq_delta,
        "motif_entropy_delta": entropy_delta,
        "nested_portal_gain": nested_gain,
        "nested_orbit_seal_gain": seal_gain,
        "ring_core_spine_port_gain": ring_gain,
        "interlocked_port_gain": interlock_gain,
        "passes": passes,
        "risks": risks,
    }


def build_recommended_generation_patch(best_policy: Dict[str, Any], judgement: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    accepted_families = sorted(set(str(c.get("production_family_v7", "")) for c in candidates if c.get("production_family_v7")))
    # If the chosen policy still has risks, force stricter guards in the patch.
    needs_strict_fragment = bool(judgement.get("short_fragment_increase", 0.0) > 0.015 or judgement.get("risks"))
    needs_anti_collapse = bool(judgement.get("top_signature_rate_delta", 0.0) > 0.05 or judgement.get("risks"))
    return {
        "patch_type": "production_feedback_control_patch",
        "status": judgement.get("verdict"),
        "selected_policy_id": best_policy.get("policy_id"),
        "production_rule_influence": round(float(best_policy.get("production_weight", 0.4)), 4),
        "family_sampling_weights": best_policy.get("family_weights", {}),
        "enabled_production_families": [f for f in accepted_families if f != "DISCOVERED_TOPOLOGY_MOTIF"],
        "deprioritized_families": ["DISCOVERED_TOPOLOGY_MOTIF"],
        "hard_guards": {
            "enable_short_fragment_guard": True,
            "min_split_fragment_ratio": "1/6",
            "min_line_length_px": 96,
            "prefer_single_component": True,
            "avoid_low_context_cycle": True,
            "reject_bad_no_structural_field": False,
        },
        "diversity_guards": {
            "enable_signature_anti_collapse": True,
            "max_top_signature_rate_delta": 0.08 if needs_anti_collapse else 0.12,
            "family_balance_temperature": 0.75,
            "avoid_same_family_streak": 3,
        },
        "risk_response": {
            "if_bad_short_or_fragmented_increases": "increase min_line_length / split-fragment penalty before raising production influence",
            "if_top_signature_rate_increases": "lower NESTED_PORTAL/NESTED_ORBIT_SEAL sampling and rotate production families",
            "if_bad_floating_not_reduced": "raise port coupling and enforce component attachment projection",
        },
        "feedback_score": judgement.get("feedback_score"),
        "notes": [
            "This patch is produced by the automatic feedback judgement chain, not by manual rule screening.",
            "Generated smoke-test samples are held in memory only and are not written to disk.",
            "Wire this patch into the real PCG generator as control logic, then rerun the same feedback chain on real generator output if needed.",
        ],
    }


def write_feedback_chain_report(path: Path, base_sum: Dict[str, Any], scoreboard: List[Dict[str, Any]], best: Dict[str, Any], patch: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Production Feedback Judgement Chain Report")
    lines.append("")
    lines.append(f"Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## What this evaluates")
    lines.append("")
    lines.append("This chain does **not** ask you to manually screen rules. It exports V7 production candidates, generates temporary baseline/production smoke-test samples in memory, evaluates topology metrics, and emits a generator-control patch.")
    lines.append("")
    lines.append("## Baseline summary")
    lines.append("")
    lines.append(f"- Baseline samples: **{base_sum.get('n', 0)}**")
    div = base_sum.get("diversity", {})
    lines.append(f"- Baseline unique signature ratio: **{float(div.get('signature_uniqueness_ratio', 0.0)):.4f}**")
    lines.append(f"- Baseline top signature rate: **{float(div.get('top_signature_rate', 0.0)):.4f}**")
    lines.append("")
    lines.append("## Policy scoreboard")
    lines.append("")
    lines.append("| rank | policy | verdict | score | target_gain | short_frag_delta | top_sig_delta | notes |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---|")
    for i, row in enumerate(scoreboard, 1):
        notes = "; ".join((row.get("passes") or [])[:2] + (row.get("risks") or [])[:2])
        lines.append(
            f"| {i} | {row.get('policy_id')} | {row.get('verdict')} | {float(row.get('feedback_score',0.0)):.4f} | "
            f"{float(row.get('target_gain',0.0)):.4f} | {float(row.get('short_fragment_increase',0.0)):.4f} | "
            f"{float(row.get('top_signature_rate_delta',0.0)):.4f} | {notes} |"
        )
    lines.append("")
    lines.append("## Selected policy")
    lines.append("")
    lines.append(f"- Selected: **{best.get('policy_id')}**")
    lines.append(f"- Verdict: **{best.get('verdict')}**")
    lines.append(f"- Feedback score: **{float(best.get('feedback_score',0.0)):.4f}**")
    if best.get("passes"):
        lines.append("- Pass signals: " + "; ".join(best.get("passes", [])))
    if best.get("risks"):
        lines.append("- Risk signals: " + "; ".join(best.get("risks", [])))
    lines.append("")
    lines.append("## Recommended generator patch")
    lines.append("")
    lines.append(f"- production_rule_influence: **{patch.get('production_rule_influence')}**")
    lines.append(f"- enabled families: `{', '.join(patch.get('enabled_production_families', []))}`")
    lines.append("- hard guards: short-fragment guard, min split fragment ratio, min line length, single-component preference")
    lines.append("- diversity guards: signature anti-collapse and family rotation")
    lines.append("")
    lines.append("## Next engineering step")
    lines.append("")
    lines.append("Wire `recommended_generation_control_patch.json` into the main PCG generator's production sampler. Do not manually choose individual rules; let the sampler use family weights and guards from the patch.")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_feedback_judgement_chain(args: argparse.Namespace, candidates: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    if not getattr(args, "feedback_loop", True):
        return None
    import random
    out_dir = Path(args.out_dir).expanduser().resolve()
    feedback_out = out_dir / "feedback_effect_chain_outputs"
    feedback_out.mkdir(parents=True, exist_ok=True)
    if candidates is None:
        candidate_json = Path(args.candidate_json).expanduser().resolve() if getattr(args, "candidate_json", "") else (out_dir / "production_rule_candidates_review.json")
        candidates = json.loads(candidate_json.read_text(encoding="utf-8")) if candidate_json.exists() else []

    n = max(50, int(getattr(args, "feedback_samples", 600)))
    seed = int(getattr(args, "generation_seed", 42))
    base_rng = random.Random(seed)
    baseline_records = [generate_baseline_memory_record(base_rng, i) for i in range(n)]
    baseline_feats = [extract_eval_features(r) for r in baseline_records]
    baseline_sum = summarize_feature_rows(baseline_feats, candidates)

    scoreboard: List[Dict[str, Any]] = []
    policy_artifacts: Dict[str, Any] = {}
    for p_idx, policy in enumerate(FEEDBACK_POLICIES):
        rng = random.Random(seed + 1009 + p_idx * 97)
        exp_records = [generate_guarded_feedback_record(rng, i, candidates, policy) for i in range(n)]
        exp_feats = [extract_eval_features(r) for r in exp_records]
        exp_sum = summarize_feature_rows(exp_feats, candidates)
        comp = compare_summaries(baseline_sum, exp_sum)
        judgement = judge_feedback_policy(baseline_sum, exp_sum, comp, policy)
        row = {
            **judgement,
            "production_weight": policy.get("production_weight"),
            "short_fragment_guard": policy.get("short_fragment_guard"),
            "description": policy.get("description"),
        }
        scoreboard.append(row)
        policy_artifacts[str(policy.get("policy_id"))] = {
            "policy": policy,
            "summary": exp_sum,
            "comparison": comp,
            "judgement": judgement,
        }

    scoreboard.sort(key=lambda r: (float(r.get("feedback_score", 0.0)), -float(r.get("short_fragment_increase", 0.0))), reverse=True)
    best = scoreboard[0] if scoreboard else {}
    selected_policy = next((p for p in FEEDBACK_POLICIES if p.get("policy_id") == best.get("policy_id")), FEEDBACK_POLICIES[0])
    patch = build_recommended_generation_patch(selected_policy, best, candidates)

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "feedback_judgement_chain",
        "generated_samples_per_policy": n,
        "sample_storage": "memory_only_generated_samples_not_written_to_disk",
        "baseline_summary": baseline_sum,
        "policy_scoreboard": scoreboard,
        "selected_policy": best,
        "recommended_generation_control_patch": patch,
        "policy_artifacts": policy_artifacts,
    }

    write_json(feedback_out / "feedback_effect_summary.json", result)
    write_json(feedback_out / "recommended_generation_control_patch.json", patch)
    write_csv(feedback_out / "feedback_policy_scoreboard.csv", scoreboard)
    # Flatten family deltas for the selected policy for quick inspection.
    selected_art = policy_artifacts.get(str(best.get("policy_id")), {})
    selected_comp = selected_art.get("comparison", {})
    write_csv(feedback_out / "selected_policy_bad_subtype_delta.csv", [{"subtype": k, **v} for k, v in selected_comp.get("bad_subtype_delta", {}).items()])
    write_csv(feedback_out / "selected_policy_production_family_delta.csv", [{"family": k, **v} for k, v in selected_comp.get("production_family_delta", {}).items()])
    write_feedback_chain_report(feedback_out / "FEEDBACK_EFFECT_CHAIN_REPORT.md", baseline_sum, scoreboard, best, patch)

    print("")
    print("Feedback judgement chain:")
    print(f"  Generated baseline samples         = {baseline_sum.get('n', 0)}  (memory only)")
    print(f"  Internal policies evaluated        = {len(scoreboard)}")
    print(f"  Selected policy                    = {best.get('policy_id')}")
    print(f"  Verdict                            = {best.get('verdict')}")
    print(f"  Feedback score                     = {float(best.get('feedback_score', 0.0)):.4f}")
    print(f"  Feedback out dir                   = {feedback_out}")
    print("  Wrote:")
    print(f"    {feedback_out / 'FEEDBACK_EFFECT_CHAIN_REPORT.md'}")
    print(f"    {feedback_out / 'feedback_effect_summary.json'}")
    print(f"    {feedback_out / 'feedback_policy_scoreboard.csv'}")
    print(f"    {feedback_out / 'recommended_generation_control_patch.json'}")
    if best.get("passes"):
        print("  Pass signals:")
        for x in best.get("passes", [])[:5]:
            print(f"    - {x}")
    if best.get("risks"):
        print("  Risk signals:")
        for x in best.get("risks", [])[:5]:
            print(f"    - {x}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V7 production-ready grammar rule prototypes and run an automatic feedback judgement chain.")
    parser.add_argument("--source-csv", type=str, default=str(DEFAULT_SOURCE_CSV), help="Path to V7 production_ready_grammar_rule_prototypes.csv")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--top-n", type=int, default=24, help="Max number of production candidates to export")
    parser.add_argument("--include-tiers", nargs="+", default=["production_ready", "production_aligned"], help="production_tier_v7 values to include")
    parser.add_argument("--min-good-support", type=float, default=0.04)
    parser.add_argument("--max-bad-support", type=float, default=0.18)
    parser.add_argument("--min-cleaned-support", type=float, default=0.03)
    parser.add_argument("--min-manual-support", type=float, default=0.02)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--require-alignment", action="store_true", help="Require cleaned or manual support above threshold")
    parser.add_argument("--exclude-discovered", action="store_true", help="Exclude generic DISCOVERED_TOPOLOGY_MOTIF rows")
    parser.add_argument("--install-review-copy", action="store_true", help="Copy review-only draft into Morpheme/new_rule_cache with timestamp. Still disabled/review_only.")

    # Optional automatic AB evaluation of generated samples. This does not require manual screening.
    parser.add_argument("--eval-ab", action="store_true", help="After export, compare baseline vs experiment generated sample directories automatically.")
    parser.add_argument("--baseline-dir", type=str, default="", help="Directory containing baseline generated JSON samples.")
    parser.add_argument("--experiment-dir", type=str, default="", help="Directory containing production-rule experiment generated JSON samples.")
    parser.add_argument("--eval-out-dir", type=str, default="", help="Output directory for AB evaluation. Default: <out-dir>/ab_eval_outputs")
    parser.add_argument("--candidate-json", type=str, default="", help="Optional production_rule_candidates_review.json to use for AB coverage. Defaults to current export output.")
    parser.add_argument("--max-eval-files", type=int, default=0, help="Optional cap on JSON files per AB directory. 0 means no cap.")
    parser.add_argument("--max-eval-records", type=int, default=0, help="Optional cap on records per AB directory. 0 means no cap.")

    # Generate temporary baseline/experiment samples in memory and evaluate them without writing generated samples to disk.
    parser.add_argument("--eval-generate-memory", action="store_true", help="Generate baseline and production-biased samples in memory, then evaluate them automatically. No generated samples are written to local storage.")
    parser.add_argument("--generated-samples", type=int, default=500, help="Number of in-memory samples per group for --eval-generate-memory.")
    parser.add_argument("--generation-seed", type=int, default=42, help="Random seed for in-memory generation evaluation.")
    parser.add_argument("--memory-production-weight", type=float, default=0.75, help="Probability that an experiment in-memory sample uses an exported production blueprint instead of baseline random generation.")

    # Feedback judgement chain. Enabled by default in this file: no external baseline/experiment directories are required.
    parser.add_argument("--feedback-loop", dest="feedback_loop", action="store_true", default=True, help="Run the automatic feedback judgement chain after exporting rules. Enabled by default.")
    parser.add_argument("--no-feedback-loop", dest="feedback_loop", action="store_false", help="Only export rules; skip the automatic feedback judgement chain.")
    parser.add_argument("--feedback-samples", type=int, default=600, help="Internal memory-only samples per policy for the feedback judgement chain.")
    args = parser.parse_args()

    source_csv = Path(args.source_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    rows = read_csv_rows(source_csv)

    # Deterministic ranking before filtering.
    rows.sort(key=lambda r: (
        tier_rank(norm_text(r.get("production_tier_v7"))) * -1,
        safe_float(r.get("score")),
        safe_float(r.get("cleaned_support")),
        safe_float(r.get("manual_support")),
        safe_float(r.get("good_support")),
    ), reverse=True)
    # Better rank: tier asc, score desc. Use a second stable sort for clarity.
    rows.sort(key=lambda r: (tier_rank(norm_text(r.get("production_tier_v7"))), -safe_float(r.get("score")), -safe_float(r.get("cleaned_support")), -safe_float(r.get("manual_support"))))

    decisions: List[ExportDecision] = []
    accepted_rows: List[Dict[str, Any]] = []
    for row in rows:
        ok, reason = row_passes_filters(row, args)
        idx_preview = len(accepted_rows) + 1
        decisions.append(ExportDecision(
            rule_id=f"row_{len(decisions)+1:04d}",
            decision="accepted" if ok else "rejected",
            reason=reason,
            production_tier_v7=norm_text(row.get("production_tier_v7")),
            production_family_v7=norm_text(row.get("production_family_v7")),
            score=safe_float(row.get("score")),
            good_support=safe_float(row.get("good_support")),
            bad_support=safe_float(row.get("bad_support")),
            cleaned_support=safe_float(row.get("cleaned_support")),
            manual_support=safe_float(row.get("manual_support")),
        ))
        if ok:
            accepted_rows.append(row)
        if len(accepted_rows) >= args.top_n:
            # Keep scanning is unnecessary for the outputs; decisions are a sample, not full audit.
            break

    candidates = [build_production_candidate(row, i + 1) for i, row in enumerate(accepted_rows[: args.top_n])]

    out_dir.mkdir(parents=True, exist_ok=True)
    draft = compact_rule_cache_draft(candidates)
    profile = generation_profile_patch(candidates)

    write_json(out_dir / "production_rule_candidates_review.json", candidates)
    write_json(out_dir / "production_rule_cache_draft_review_only.json", draft)
    write_json(out_dir / "generation_profile_patch_suggestion.json", profile)
    write_csv(out_dir / "export_decisions_sample.csv", [asdict(d) for d in decisions])
    (out_dir / "README_EXPORT_REPORT.md").write_text(markdown_report(candidates, decisions, args), encoding="utf-8")

    installed_path: Optional[Path] = None
    if args.install_review_copy:
        MORPHEME_NEW_RULE_CACHE.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        installed_path = MORPHEME_NEW_RULE_CACHE / f"production_rule_cache_draft_review_only_{stamp}.json"
        shutil.copy2(out_dir / "production_rule_cache_draft_review_only.json", installed_path)

    print("=" * 92)
    print("Morpheme V7 Production Rule Exporter")
    print("=" * 92)
    print(f"SCRIPT_DIR              = {SCRIPT_DIR}")
    print(f"SOURCE_CSV              = {source_csv}")
    print(f"OUT_DIR                 = {out_dir}")
    print(f"MORPHEME_NEW_RULE_CACHE = {MORPHEME_NEW_RULE_CACHE}")
    print("")
    print(f"Read rows               = {len(rows)}")
    print(f"Accepted candidates     = {len(candidates)}")
    print(f"Include tiers           = {', '.join(args.include_tiers)}")
    print(f"Filters                 = min_good>={args.min_good_support}, max_bad<={args.max_bad_support}, min_score>={args.min_score}")
    print("")
    print("Wrote:")
    print(f"  {out_dir / 'production_rule_candidates_review.json'}")
    print(f"  {out_dir / 'production_rule_cache_draft_review_only.json'}")
    print(f"  {out_dir / 'generation_profile_patch_suggestion.json'}")
    print(f"  {out_dir / 'export_decisions_sample.csv'}")
    print(f"  {out_dir / 'README_EXPORT_REPORT.md'}")
    if installed_path:
        print(f"  Review copy installed to: {installed_path}")
    print("")
    if candidates:
        print("Top exported candidates:")
        for c in candidates[:10]:
            s = c["support"]
            print(f"  - {c['rule_id']} | {c['production_family_v7']} | score={s['score']:.4f} good={s['good_support']:.4f} bad={s['bad_support']:.4f} cleaned={s['cleaned_support']:.4f} manual={s['manual_support']:.4f}")
    else:
        print("No candidates accepted. Try relaxing --max-bad-support or disabling --require-alignment.")

    # Optional automatic AB generation quality evaluation.
    if args.eval_ab:
        run_ab_evaluator(args, candidates=candidates)
    if args.eval_generate_memory:
        run_memory_generation_evaluator(args, candidates=candidates)
    if args.feedback_loop:
        run_feedback_judgement_chain(args, candidates=candidates)

    print("=" * 92)


if __name__ == "__main__":
    main()
