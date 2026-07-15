"""
Verify Proposition 6: Welfare Ranking Across Equilibria.

For three parameter regimes, find all NPL equilibria and compute approximate
ex-ante welfare under each. Reference welfare ranking (RA Guide):
  (i)  Low psi  -> P^H dominates
  (ii) High psi -> P^M dominates (over-automation failure)
  (iii) High K  -> P^L dominates (arms race avoidance)

Equilibrium search matches Prop3: find_all_equilibria with N_INITS structured
+ random inits; no exclude_low or type filtering during search. All converged
equilibria are reported; user judges welfare ranking from output.

Regime (iii) High K also uses direct P^L verification (Section 2.5): P^L is
NPL-unstable, so find_all_equilibria often misses it; verify_low_adoption_eq +
refine_low_eq_on_reachable recovers the economically relevant profile.

Usage (from project root):
    python Prop6/verify_proposition6.py
    python Prop6/verify_proposition6.py --regime i ii
    python Prop6/verify_proposition6.py --regime iii

Output: Prop6/prop6_welfare_ranking_verification.txt
When --regime selects a subset and the output file already exists, only the
selected regime block(s) and matching summary TABLE row(s) are updated.
"""

import argparse
import os
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import ModelParams, StateSpace, compute_payoffs
from npl import find_all_equilibria, classify_equilibrium, npl_iteration
from verify_equilibrium_existence import (
    DELTA_X_BASE,
    DELTA_X_COMP,
    DEV_MODE,
    BASE_PARAMS,
    N_INITS,
    NPL_MAX_ITER,
    NPL_TOL,
    NPL_DAMPING,
    RESIDUAL_VF_ITER,
    DISTINCT_TOL,
    compute_equilibrium_stats,
    equilibrium_residual,
    equilibrium_residual_on_mask,
    verify_low_adoption_eq,
    refine_low_eq_on_reachable,
    ccp_distance,
)

DELTA_FIXED = 3.0

REGIME_CONFIGS = [
    {
        "id": "i",
        "label": "(i) Low psi (P^H dominates)",
        "expected_dominant": "P^H",
        "delta": DELTA_FIXED,
        "psi": 0.3,
        "K_L": 4.0,
        "K_H": 3.0,
    },
    {
        "id": "ii",
        "label": "(ii) High psi (P^M dominates, over-automation)",
        "expected_dominant": "P^M",
        "delta": DELTA_FIXED,
        "psi": 5.0,
        "K_L": 4.0,
        "K_H": 3.0,
    },
    {
        "id": "iii",
        "label": "(iii) High K (P^L dominates, arms race)",
        "expected_dominant": "P^L",
        "delta": DELTA_FIXED,
        "psi": 1.0,
        "K_L": 10.0,
        "K_H": 9.0,
    },
]

OUTPUT_TXT = os.path.join(SCRIPT_DIR, "prop6_welfare_ranking_verification.txt")


def build_params(regime):
    """ModelParams for one welfare-ranking regime."""
    return ModelParams(
        **BASE_PARAMS,
        delta=regime["delta"],
        psi=regime["psi"],
        K_L=regime["K_L"],
        K_H=regime["K_H"],
        delta_x_base=DELTA_X_BASE,
        delta_x_comp=DELTA_X_COMP,
    )


def enrich_equilibria(ss, params, N_C, N_G, raw_eqs, u_C, u_G):
    """Attach type, stats, residual to each equilibrium (same pattern as Prop3)."""
    enriched = []
    for eq in raw_eqs:
        stats = eq.get("stats") or compute_equilibrium_stats(eq["P_C"], eq["P_G"])
        residual = equilibrium_residual(
            ss, params, eq["P_C"], eq["P_G"], N_C, N_G, u_C, u_G,
            max_vf_iter=RESIDUAL_VF_ITER,
        )
        enriched.append({
            **eq,
            "stats": stats,
            "type": classify_equilibrium(stats),
            "residual": residual,
        })
    return enriched


def obtain_direct_pl(ss, params, N_C, N_G, u_C, u_G):
    """
    Direct P^L verification (Section 2.5), same pipeline as Prop1/Prop2.

    Returns (eq_dict or None, n_reachable, verified).
    """
    low_ok, _, _, reach_mask = verify_low_adoption_eq(
        params, ss, N_C, N_G, u_C=u_C, u_G=u_G, verbose=False,
    )
    n_reach = int(reach_mask.sum())
    if not low_ok:
        return None, n_reach, False

    P_C, P_G = refine_low_eq_on_reachable(
        ss, params, N_C, N_G, u_C, u_G, reach_mask,
    )
    stats = compute_equilibrium_stats(P_C, P_G)
    P_C_br, P_G_br = npl_iteration(
        ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
        max_vf_iter=RESIDUAL_VF_ITER,
    )
    res_reach = equilibrium_residual_on_mask(
        P_C, P_G, P_C_br, P_G_br, reach_mask,
    )
    res_global = equilibrium_residual(
        ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
        max_vf_iter=RESIDUAL_VF_ITER,
    )
    return {
        "P_C": P_C,
        "P_G": P_G,
        "stats": stats,
        "type": classify_equilibrium(stats),
        "residual": res_reach,
        "residual_global": res_global,
        "label": "direct_P^L",
        "source": "direct",
        "n_reachable": n_reach,
    }, n_reach, True


def already_has_profile(eqs, P_C, P_G):
    """True if an existing eq is within DISTINCT_TOL of (P_C, P_G)."""
    for eq in eqs:
        if ccp_distance(P_C, P_G, eq["P_C"], eq["P_G"]) < DISTINCT_TOL:
            return True
    return False


def canonical_type(eq_type):
    """Map classify_equilibrium label to P^L / P^M / P^H, or None."""
    if eq_type.startswith("P^L"):
        return "P^L"
    if eq_type.startswith("P^M"):
        return "P^M"
    if eq_type.startswith("P^H"):
        return "P^H"
    return None


def compute_welfare(eq, u_C, u_G, N_C, N_G, N):
    """Approximate ex-ante welfare under equilibrium CCPs."""
    w_C = np.sum(eq["P_C"] * u_C, axis=1).mean()
    w_G = np.sum(eq["P_G"] * u_G, axis=1).mean()
    w_total = (N_C * w_C + N_G * w_G) / N
    return {"w_C": w_C, "w_G": w_G, "w_total": w_total}


def format_equilibrium_stats(stats):
    """Format adoption / HITL / agent rates for one equilibrium."""
    return [
        f"       C: adopt={stats['adopt_C']:.3f} "
        f"(HITL={stats['hitl_C']:.3f}, agent={stats['agent_C']:.3f})",
        f"       G: adopt={stats['adopt_G']:.3f} "
        f"(HITL={stats['hitl_G']:.3f}, agent={stats['agent_G']:.3f})",
    ]


def best_welfare_by_canonical_type(welfare_rows):
    """Best W_total per canonical equilibrium type (P^L, P^M, P^H)."""
    best = {}
    for row in welfare_rows:
        ctype = row["canonical"]
        if ctype is None:
            continue
        if ctype not in best or row["w_total"] > best[ctype]["w_total"]:
            best[ctype] = row
    return best


def all_types_found(welfare_rows):
    """Comma-separated list of all equilibrium types found."""
    return ", ".join(row["eq_type"] for row in welfare_rows)


def verify_regime_ranking(welfare_rows, expected_dominant):
    """
    Informational: compare best canonical welfare to RA Guide expectation.

    Does not filter equilibria; user judges from full output.
    """
    best = best_welfare_by_canonical_type(welfare_rows)
    if expected_dominant not in best:
        return {
            "ranking_pass": False,
            "expected_found": False,
            "expected_dominant": expected_dominant,
            "best_type": None,
            "best_welfare": np.nan,
            "expected_welfare": np.nan,
            "canonical_found": sorted(best.keys()),
            "all_types": all_types_found(welfare_rows),
        }

    best_type = max(best, key=lambda t: best[t]["w_total"])
    expected_w = best[expected_dominant]["w_total"]
    best_w = best[best_type]["w_total"]

    return {
        "ranking_pass": best_type == expected_dominant,
        "expected_found": True,
        "expected_dominant": expected_dominant,
        "best_type": best_type,
        "best_welfare": best_w,
        "expected_welfare": expected_w,
        "canonical_found": sorted(best.keys()),
        "all_types": all_types_found(welfare_rows),
    }


def run_regime(regime, ss, N_C, N_G, log_lines):
    """Find equilibria and compute welfare for one parameter regime."""
    regime_id = regime["id"]
    label = regime["label"]
    expected = regime["expected_dominant"]
    use_direct_pl = regime_id == "iii"

    print(f"[ regime {regime_id} ]  finding equilibria...", flush=True)

    params = build_params(regime)
    u_C, u_G = compute_payoffs(ss, params)

    log_lines.append(f"[ regime {regime_id}: {label} ]")
    log_lines.append(
        f"  Parameters: delta={regime['delta']}, psi={regime['psi']}, "
        f"K_L={regime['K_L']}, K_H={regime['K_H']}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}"
    )
    log_lines.append(f"  RA Guide reference (informational): expected dominant = {expected}")
    if use_direct_pl:
        log_lines.append(
            f"  Method: find_all_equilibria (NPL) + direct P^L verification "
            f"(Section 2.5); n_inits={N_INITS}, max_iter={NPL_MAX_ITER}, "
            f"tol={NPL_TOL}, damping={NPL_DAMPING}"
        )
    else:
        log_lines.append(
            f"  Method: find_all_equilibria (same as Prop3); "
            f"n_inits={N_INITS}, max_iter={NPL_MAX_ITER}, "
            f"tol={NPL_TOL}, damping={NPL_DAMPING}; no filtering during search"
        )
    log_lines.append("")

    raw_eqs = find_all_equilibria(
        ss, params, N_C, N_G,
        n_inits=N_INITS, tol=NPL_TOL,
        max_iter=NPL_MAX_ITER, damping=NPL_DAMPING,
        verbose=False,
    )
    eqs = enrich_equilibria(ss, params, N_C, N_G, raw_eqs, u_C, u_G)
    n_npl = len(eqs)

    if use_direct_pl:
        print(f"[ regime {regime_id} ]  direct P^L verification...", flush=True)
        pl_eq, n_reach, pl_ok = obtain_direct_pl(
            ss, params, N_C, N_G, u_C, u_G,
        )
        log_lines.append(
            f"  Direct P^L (Section 2.5): "
            f"reachable states={n_reach}, verified={pl_ok}"
        )
        if pl_ok and pl_eq is not None:
            if already_has_profile(eqs, pl_eq["P_C"], pl_eq["P_G"]):
                log_lines.append(
                    "  Direct P^L already present among NPL equilibria "
                    "(skipped duplicate)."
                )
            else:
                eqs.append(pl_eq)
                log_lines.append(
                    "  Appended P^L via direct verification "
                    "(NPL-unstable / typically missed by NPL)."
                )
        else:
            log_lines.append("  WARNING: P^L direct verification FAILED")
        log_lines.append("")

    if not eqs:
        if use_direct_pl:
            log_lines.append(
                f"  No equilibria found "
                f"(NPL converged={n_npl}; direct P^L failed)."
            )
        else:
            log_lines.append("  No equilibria found via NPL.")
        log_lines.append("")
        print(f"[ regime {regime_id} ]  done — no equilibria found", flush=True)
        return {
            "regime_id": regime_id,
            "label": label,
            "expected_dominant": expected,
            "n_equilibria": 0,
            "welfare_rows": [],
            "ranking_check": verify_regime_ranking([], expected),
        }

    welfare_rows = []
    for i, eq in enumerate(eqs, start=1):
        stats = eq["stats"]
        eq_type = eq["type"]
        welfare = compute_welfare(eq, u_C, u_G, N_C, N_G, params.N)
        init_label = eq.get("label", "N/A")
        source = eq.get("source", "npl")

        log_lines.append(f"  Equilibrium {i}: {eq_type}  (init={init_label})")
        log_lines.extend(format_equilibrium_stats(stats))
        log_lines.append(
            f"       W(C)={welfare['w_C']:.3f}  "
            f"W(G)={welfare['w_G']:.3f}  "
            f"W_total={welfare['w_total']:.3f}"
        )
        if source == "direct" and "residual_global" in eq:
            log_lines.append(
                f"       Residual (reachable): {eq['residual']:.2e}"
            )
            log_lines.append(
                f"       Residual (global):    {eq['residual_global']:.2e}"
            )
            log_lines.append(
                "       Note: global residual may exceed 1e-8 off-path because "
                "P^L is NPL-unstable outside the ergodic component."
            )
        else:
            log_lines.append(f"       Residual: {eq['residual']:.2e}")
        log_lines.append("")

        welfare_rows.append({
            "eq_index": i,
            "eq_type": eq_type,
            "canonical": canonical_type(eq_type),
            "w_C": welfare["w_C"],
            "w_G": welfare["w_G"],
            "w_total": welfare["w_total"],
            "residual": eq["residual"],
            "adopt_C": stats["adopt_C"],
            "adopt_G": stats["adopt_G"],
            "agent_C": stats["agent_C"],
            "agent_G": stats["agent_G"],
        })

    ranking_check = verify_regime_ranking(welfare_rows, expected)

    log_lines.append(
        f"  All types found ({len(eqs)} equilibria): {ranking_check['all_types']}"
    )
    if ranking_check["canonical_found"]:
        log_lines.append(
            f"  Best canonical by welfare: {ranking_check['best_type']} "
            f"(W_total={ranking_check['best_welfare']:.3f})"
        )
        log_lines.append(
            f"  Canonical types: {', '.join(ranking_check['canonical_found'])}"
        )
    log_lines.append(
        f"  Reference check ({expected} vs best canonical): "
        f"{'match' if ranking_check['ranking_pass'] else 'no match'} "
        f"(informational only)"
    )
    log_lines.append("")

    print(
        f"[ regime {regime_id} ]  done — "
        f"{len(eqs)} equilibria, types={ranking_check['all_types']}",
        flush=True,
    )

    return {
        "regime_id": regime_id,
        "label": label,
        "expected_dominant": expected,
        "n_equilibria": len(eqs),
        "welfare_rows": welfare_rows,
        "ranking_check": ranking_check,
    }


def run_all_regimes(log_lines, regime_ids=None):
    """Run welfare ranking verification for selected regimes."""
    if regime_ids is None:
        regimes = REGIME_CONFIGS
    else:
        id_set = {r.lower() for r in regime_ids}
        regimes = [r for r in REGIME_CONFIGS if r["id"] in id_set]
        if not regimes:
            raise ValueError(
                f"No matching regimes for {regime_ids!r}; "
                f"choose from {[r['id'] for r in REGIME_CONFIGS]}"
            )

    log_lines.append("PROPOSITION 6: WELFARE RANKING VERIFICATION")
    log_lines.append("=" * 105)
    log_lines.append(
        f"Fixed delta={DELTA_FIXED}, delta_x_base={DELTA_X_BASE}, "
        f"delta_x_comp={DELTA_X_COMP}"
    )
    log_lines.append(
        f"State space (DEV_MODE={DEV_MODE}): "
        f"N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"T_bar={BASE_PARAMS['T_bar']}, tau_bar={BASE_PARAMS['tau_bar']}"
    )
    log_lines.append(
        "Welfare: mean over states of E[u|s] under equilibrium CCPs; "
        "W_total = (N_C*W(C) + N_G*W(G)) / N"
    )
    log_lines.append(
        f"Equilibrium search: find_all_equilibria (same as Prop3); "
        f"n_inits={N_INITS}, max_iter={NPL_MAX_ITER}, "
        f"tol={NPL_TOL}, damping={NPL_DAMPING}; no type filtering"
    )
    log_lines.append("")

    summary_rows = []
    params0 = build_params(regimes[0])
    ss = StateSpace(params0)
    N_C = params0.N // 2
    N_G = params0.N - N_C

    for regime in regimes:
        result = run_regime(regime, ss, N_C, N_G, log_lines)
        rc = result["ranking_check"]
        summary_rows.append({
            "regime_id": result["regime_id"],
            "label": result["label"],
            "expected": result["expected_dominant"],
            "n_equilibria": result["n_equilibria"],
            "all_types": rc.get("all_types", ""),
            "canonical_found": ", ".join(rc["canonical_found"]) or "none",
            "best_type": rc["best_type"] if rc["best_type"] else "N/A",
            "best_welfare": rc["best_welfare"],
            "ranking_pass": rc["ranking_pass"],
        })

    log_lines.append("=" * 105)
    log_lines.append("TABLE: Proposition 6 Verification Summary")
    log_lines.append("-" * 105)
    header = (
        f"{'regime':<8}| {'ref':<6}| {'#eq':<4}| "
        f"{'all types':<40}| {'best(can)':<8}| {'W(best)':<9}"
    )
    log_lines.append(header)
    log_lines.append("-" * 105)

    for row in summary_rows:
        w_best = (
            f"{row['best_welfare']:.3f}"
            if row["best_type"] != "N/A" and not np.isnan(row["best_welfare"])
            else "N/A"
        )
        types_short = row["all_types"] if row["all_types"] else "none"
        if len(types_short) > 40:
            types_short = types_short[:37] + "..."
        log_lines.append(
            f"{row['regime_id']:<8}| "
            f"{row['expected']:<6}| "
            f"{row['n_equilibria']:<4d}| "
            f"{types_short:<40}| "
            f"{row['best_type']:<8}| "
            f"{w_best:<9}"
        )

    return summary_rows


def print_summary_table(summary_rows):
    """Print summary table to console."""
    print()
    print("TABLE: Proposition 6 Summary")
    print("-" * 80)
    print(f"{'regime':<8} {'ref':<6} {'#eq':<4} {'best(can)':<8} {'W(best)':<9}")
    print("-" * 80)
    for row in summary_rows:
        w_best = (
            f"{row['best_welfare']:.3f}"
            if row["best_type"] != "N/A" and not np.isnan(row["best_welfare"])
            else "N/A"
        )
        print(
            f"{row['regime_id']:<8} "
            f"{row['expected']:<6} "
            f"{row['n_equilibria']:<4d} "
            f"{row['best_type']:<8} "
            f"{w_best:<9}"
        )


def format_summary_table_row(row):
    """One row of the verification summary TABLE (fixed column widths)."""
    w_best = (
        f"{row['best_welfare']:.3f}"
        if row["best_type"] != "N/A" and not np.isnan(row["best_welfare"])
        else "N/A"
    )
    types_short = row["all_types"] if row["all_types"] else "none"
    if len(types_short) > 40:
        types_short = types_short[:37] + "..."
    return (
        f"{row['regime_id']:<8}| "
        f"{row['expected']:<6}| "
        f"{row['n_equilibria']:<4d}| "
        f"{types_short:<40}| "
        f"{row['best_type']:<8}| "
        f"{w_best:<9}"
    )


def extract_regime_blocks(log_text):
    """
    Split verification log into preamble, ordered regime blocks, and table tail.

    Returns (preamble, {regime_id: block_text}, table_section).
    """
    # Regime blocks start at "[ regime <id>:" and run until next regime or TABLE.
    parts = re.split(r"(?=\[ regime )", log_text)
    preamble = parts[0]
    regimes = {}
    table_section = ""
    for part in parts[1:]:
        if part.lstrip().startswith("=") or "TABLE:" in part[:80]:
            # Shouldn't happen with lookahead, but keep robust
            table_section += part
            continue
        m = re.match(r"\[ regime (i{1,3}):", part)
        if not m:
            table_section += part
            continue
        rid = m.group(1)
        # If TABLE follows inside this chunk (last regime), split it off
        table_idx = part.find("\n====")
        if table_idx < 0:
            table_idx = part.find("\nTABLE:")
        if table_idx >= 0:
            regimes[rid] = part[:table_idx].rstrip() + "\n\n"
            table_section = part[table_idx:].lstrip("\n")
        else:
            regimes[rid] = part if part.endswith("\n") else part + "\n"
    return preamble, regimes, table_section


def merge_partial_into_output(existing_text, new_log_lines, summary_rows):
    """
    Replace only selected regime blocks and their TABLE rows in an existing file.

    Keeps preamble and untouched regimes exactly as before.
    """
    new_text = "\n".join(new_log_lines) + "\n"
    _, new_regimes, _ = extract_regime_blocks(new_text)
    preamble, old_regimes, old_table = extract_regime_blocks(existing_text)

    for rid, block in new_regimes.items():
        old_regimes[rid] = block

    # Preserve canonical order i, ii, iii
    ordered = []
    for rid in ("i", "ii", "iii"):
        if rid in old_regimes:
            ordered.append(old_regimes[rid].rstrip() + "\n\n")

    # Rebuild TABLE: keep header, replace rows for updated regimes
    updated = {row["regime_id"]: format_summary_table_row(row) for row in summary_rows}
    table_lines = old_table.splitlines()
    out_table = []
    for line in table_lines:
        m = re.match(r"^(i{1,3})\s*\|", line)
        if m and m.group(1) in updated:
            out_table.append(updated[m.group(1)].rstrip())
        else:
            out_table.append(line)
    table_section = "\n".join(out_table)
    if not table_section.endswith("\n"):
        table_section += "\n"

    return preamble.rstrip() + "\n\n" + "".join(ordered) + table_section


def main():
    """Run Proposition 6 verification and write results to file."""
    parser = argparse.ArgumentParser(
        description="Verify Proposition 6: Welfare Ranking Across Equilibria."
    )
    parser.add_argument(
        "--regime",
        type=str,
        nargs="*",
        default=None,
        choices=["i", "ii", "iii"],
        help="Optional subset of regimes (default: all three).",
    )
    args = parser.parse_args()

    regime_ids = args.regime if args.regime else None
    partial = regime_ids is not None

    print("=" * 70)
    print("PROPOSITION 6: Welfare Ranking Verification")
    print("=" * 70)
    print(
        f"Parameters: delta={DELTA_FIXED}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}"
    )
    print(
        f"Regimes: "
        f"{[r['id'] for r in REGIME_CONFIGS] if regime_ids is None else regime_ids}"
    )
    print(
        f"DEV_MODE={DEV_MODE}  "
        f"(N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"tau_bar={BASE_PARAMS['tau_bar']}, N_INITS={N_INITS})"
    )
    print(
        f"NPL: max_iter={NPL_MAX_ITER}, tol={NPL_TOL}, damping={NPL_DAMPING}"
    )
    print(f"Output file: {OUTPUT_TXT}")
    if partial:
        print("Mode: surgical merge (update selected regimes only)")
    print()

    log_lines = []
    summary_rows = run_all_regimes(log_lines, regime_ids=regime_ids)

    if partial and os.path.isfile(OUTPUT_TXT):
        with open(OUTPUT_TXT, "r", encoding="utf-8") as f:
            existing = f.read()
        merged = merge_partial_into_output(existing, log_lines, summary_rows)
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            f.write(merged)
        print(f"Surgically merged regimes {regime_ids} into existing output.")
    else:
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
            f.write("\n")

    print()
    print("=" * 70)
    print(f"Verification complete. Results written to:\n  {OUTPUT_TXT}")
    print("=" * 70)

    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
