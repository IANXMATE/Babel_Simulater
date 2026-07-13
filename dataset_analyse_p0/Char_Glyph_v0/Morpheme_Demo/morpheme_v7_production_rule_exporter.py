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
    python morpheme_v7_production_rule_exporter.py --top-n 24

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V7 production-ready grammar rule prototypes into review-only graph grammar production drafts.")
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
    print("=" * 92)


if __name__ == "__main__":
    main()
