"""
Verify Proposition 7: Asymmetric Equilibria among Homogeneous Firms.

Set omega_C_E = omega_G_E and mu_C = mu_G (all firms identical). Show that
multiple equilibria still exist — multiplicity is not an artifact of type
heterogeneity.

Core logic mirrors prop7_asymmetric_homogeneous() in verify_propositions.py.
Equilibrium search matches Prop3/Prop6: find_all_equilibria with N_INITS /
NPL settings from verify_equilibrium_existence.py; no type filtering during
search; all converged equilibria are reported.

N_G=0 workaround (from verify_propositions.py): the transition code expects
N_C >= 1, so we set N_C = N, N_G = 0 while equating type parameters so that
all active firms are identical. Optionally --firm-split keeps N_C = N//2,
N_G = N - N_C with the same homogeneous type parameters.

Usage (from project root):
    python Prop7/verify_proposition7.py
    python Prop7/verify_proposition7.py --n-inits 30
    python Prop7/verify_proposition7.py --firm-split

Output: Prop7/prop7_asymmetric_homogeneous_verification.txt
"""

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import ModelParams, StateSpace, compute_payoffs
from npl import find_all_equilibria, classify_equilibrium
from verify_equilibrium_existence import (
    DELTA_X_BASE,
    DELTA_X_COMP,
    DEV_MODE,
    BASE_PARAMS,
    N_INITS,
    NPL_MAX_ITER,
    NPL_TOL,
    NPL_DAMPING,
    DISTINCT_TOL,
    RESIDUAL_VF_ITER,
    compute_equilibrium_stats,
    equilibrium_residual,
    ccp_distance,
)

# Homogeneous-firm regime (verify_propositions.py prop7)
DELTA_FIXED = 3.0
PSI_FIXED = 2.0
OMEGA_COMMON = 2.0
MU_COMMON = 1.0

OUTPUT_TXT = os.path.join(
    SCRIPT_DIR, "prop7_asymmetric_homogeneous_verification.txt"
)


def build_params():
    """ModelParams with identical C and G type preferences."""
    return ModelParams(
        **BASE_PARAMS,
        delta=DELTA_FIXED,
        psi=PSI_FIXED,
        omega_C_E=OMEGA_COMMON,
        omega_G_E=OMEGA_COMMON,
        mu_C=MU_COMMON,
        mu_G=MU_COMMON,
        delta_x_base=DELTA_X_BASE,
        delta_x_comp=DELTA_X_COMP,
    )


def firm_counts(params, firm_split):
    """
    Firm counts under the homogeneity experiment.

    Default (firm_split=False): N_C = N, N_G = 0 — all mass on one label while
    type params are identical (verify_propositions.py workaround).

    firm_split=True: N_C = N//2, N_G = N - N_C with identical type params.
    """
    if firm_split:
        N_C = params.N // 2
        N_G = params.N - N_C
    else:
        N_C, N_G = params.N, 0
    return N_C, N_G


def enrich_equilibria(ss, params, N_C, N_G, raw_eqs, u_C, u_G):
    """Attach type, stats, residual to each equilibrium (Prop3/Prop6 pattern)."""
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


def format_equilibrium_stats(stats):
    """Format adoption / HITL / agent rates for one equilibrium."""
    return [
        f"       C: adopt={stats['adopt_C']:.3f} "
        f"(HITL={stats['hitl_C']:.3f}, agent={stats['agent_C']:.3f})",
        f"       G: adopt={stats['adopt_G']:.3f} "
        f"(HITL={stats['hitl_G']:.3f}, agent={stats['agent_G']:.3f})",
    ]


def append_distinctness(log_lines, equilibria):
    """Report pairwise CCP max-norm distances when multiple equilibria exist."""
    if len(equilibria) < 2:
        return

    log_lines.append("  Distinctness (max-norm over CCPs):")
    log_lines.append(f"  {'Pair':<40} {'Distance':>12}")
    log_lines.append(f"  {'-' * 40} {'-' * 12}")
    for i in range(len(equilibria)):
        for k in range(i + 1, len(equilibria)):
            d = ccp_distance(
                equilibria[i]["P_C"], equilibria[i]["P_G"],
                equilibria[k]["P_C"], equilibria[k]["P_G"],
            )
            ti = equilibria[i]["type"]
            tk = equilibria[k]["type"]
            pair = f"||Eq{i + 1}:{ti} - Eq{k + 1}:{tk}||"
            if len(pair) > 40:
                pair = pair[:37] + "..."
            log_lines.append(f"  {pair:<40}{d:.4f}")
            distinct = d >= DISTINCT_TOL
            log_lines.append(
                f"    distinct (tol={DISTINCT_TOL}): "
                f"{'YES' if distinct else 'NO'}"
            )
    log_lines.append("")


def verify_multiplicity(n_equilibria):
    """
    Informational check: Prop7 expects multiple equilibria under homogeneity.

    Does not filter equilibria; user judges from full output.
    """
    return {
        "multiplicity": n_equilibria >= 2,
        "n_equilibria": n_equilibria,
        "expected": "multiple equilibria exist even when firms are identical",
    }


def run_verification(log_lines, n_inits, firm_split):
    """Find all equilibria under homogeneous firm parameters."""
    params = build_params()
    ss = StateSpace(params)
    N_C, N_G = firm_counts(params, firm_split)
    u_C, u_G = compute_payoffs(ss, params)

    firm_mode = (
        f"N_C={N_C}, N_G={N_G} "
        f"({'split labels, identical types' if firm_split else 'all-C workaround'})"
    )

    log_lines.append(
        "PROPOSITION 7: ASYMMETRIC EQUILIBRIA AMONG HOMOGENEOUS FIRMS"
    )
    log_lines.append("=" * 105)
    log_lines.append(
        f"Homogeneous types: omega_C_E = omega_G_E = {OMEGA_COMMON}, "
        f"mu_C = mu_G = {MU_COMMON}"
    )
    log_lines.append(
        f"Fixed parameters: delta={DELTA_FIXED}, psi={PSI_FIXED}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}"
    )
    log_lines.append(
        f"State space (DEV_MODE={DEV_MODE}): "
        f"N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"T_bar={BASE_PARAMS['T_bar']}, tau_bar={BASE_PARAMS['tau_bar']}, "
        f"n_states={ss.n_states}"
    )
    log_lines.append(f"Firm counts: {firm_mode}")
    log_lines.append(
        f"Equilibrium search: find_all_equilibria (same as Prop3/Prop6); "
        f"n_inits={n_inits}, max_iter={NPL_MAX_ITER}, "
        f"tol={NPL_TOL}, damping={NPL_DAMPING}; no type filtering"
    )
    log_lines.append(
        f"RA Guide / Prop7 expected (informational): "
        f"multiple equilibria exist even when firms are identical"
    )
    log_lines.append("")
    log_lines.append(
        "Note: true within-population asymmetry among identical firms can "
        "require breaking the symmetric-within-type CCP assumption. "
        "Multiplicity across distinct NPL fixed points with identical type "
        "parameters is the Prop7 claim verified here."
    )
    log_lines.append("")

    print("[ Prop7 ]  finding equilibria...", flush=True)

    raw_eqs = find_all_equilibria(
        ss, params, N_C, N_G,
        n_inits=n_inits, tol=NPL_TOL,
        max_iter=NPL_MAX_ITER, damping=NPL_DAMPING,
        verbose=False,
    )
    eqs = enrich_equilibria(ss, params, N_C, N_G, raw_eqs, u_C, u_G)

    check = verify_multiplicity(len(eqs))

    if not eqs:
        log_lines.append("  No equilibria found via NPL.")
        log_lines.append("")
        log_lines.append(
            f"  Multiplicity check: FAIL "
            f"(expected: {check['expected']})"
        )
        log_lines.append("")
        print("[ Prop7 ]  done — no equilibria found", flush=True)
        return {"n_equilibria": 0, "equilibria": [], "check": check}

    for i, eq in enumerate(eqs, start=1):
        stats = eq["stats"]
        eq_type = eq["type"]
        init_label = eq.get("label", "N/A")

        log_lines.append(f"  Equilibrium {i}: {eq_type}  (init={init_label})")
        log_lines.extend(format_equilibrium_stats(stats))
        log_lines.append(f"       Residual: {eq['residual']:.2e}")
        log_lines.append("")

    append_distinctness(log_lines, eqs)

    types_str = ", ".join(eq["type"] for eq in eqs)
    log_lines.append(
        f"  All types found ({len(eqs)} equilibria): {types_str}"
    )
    log_lines.append(
        f"  Multiplicity check: "
        f"{'PASS' if check['multiplicity'] else 'FAIL'} "
        f"(n_eq={len(eqs)}; expected: {check['expected']})"
    )
    log_lines.append("")

    # Summary table
    log_lines.append("=" * 105)
    log_lines.append("TABLE: Proposition 7 Verification Summary")
    log_lines.append("-" * 105)
    header = (
        f"{'#':<4}| {'type':<28}| {'init':<12}| "
        f"{'adopt(C)':<10}| {'HITL(C)':<10}| {'agent(C)':<10}| "
        f"{'adopt(G)':<10}| {'HITL(G)':<10}| {'agent(G)':<10}| "
        f"{'residual':<12}"
    )
    log_lines.append(header)
    log_lines.append("-" * 105)

    for i, eq in enumerate(eqs, start=1):
        stats = eq["stats"]
        init_label = str(eq.get("label", "N/A"))
        if len(init_label) > 12:
            init_label = init_label[:9] + "..."
        type_label = eq["type"]
        if len(type_label) > 28:
            type_label = type_label[:25] + "..."
        log_lines.append(
            f"{i:<4}| "
            f"{type_label:<28}| "
            f"{init_label:<12}| "
            f"{stats['adopt_C']:<10.3f}| "
            f"{stats['hitl_C']:<10.3f}| "
            f"{stats['agent_C']:<10.3f}| "
            f"{stats['adopt_G']:<10.3f}| "
            f"{stats['hitl_G']:<10.3f}| "
            f"{stats['agent_G']:<10.3f}| "
            f"{eq['residual']:<12.2e}"
        )

    log_lines.append("-" * 105)
    log_lines.append(
        f"Multiplicity: {'YES' if check['multiplicity'] else 'NO'} "
        f"({len(eqs)} distinct equilibria under homogeneous firms)"
    )
    log_lines.append("")

    print(
        f"[ Prop7 ]  done — {len(eqs)} equilibria, "
        f"multiplicity={'YES' if check['multiplicity'] else 'NO'}",
        flush=True,
    )

    return {"n_equilibria": len(eqs), "equilibria": eqs, "check": check}


def print_summary(result):
    """Print a short console summary."""
    eqs = result["equilibria"]
    check = result["check"]
    print()
    print("TABLE: Proposition 7 Summary")
    print("-" * 70)
    print(f"{'#':<4} {'type':<28} {'adopt(C)':<10} {'agent(C)':<10} residual")
    print("-" * 70)
    for i, eq in enumerate(eqs, start=1):
        stats = eq["stats"]
        print(
            f"{i:<4} "
            f"{eq['type']:<28} "
            f"{stats['adopt_C']:<10.3f} "
            f"{stats['agent_C']:<10.3f} "
            f"{eq['residual']:.2e}"
        )
    if not eqs:
        print("(no equilibria)")
    print("-" * 70)
    print(
        f"Multiplicity: {'YES' if check['multiplicity'] else 'NO'} "
        f"(n_eq={result['n_equilibria']})"
    )
    print(f"Expected: {check['expected']}")


def main():
    """Run Proposition 7 verification and write results to file."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify Proposition 7: Asymmetric Equilibria "
            "among Homogeneous Firms."
        )
    )
    parser.add_argument(
        "--n-inits",
        type=int,
        default=None,
        help=f"Number of NPL initializations (default: N_INITS={N_INITS}).",
    )
    parser.add_argument(
        "--firm-split",
        action="store_true",
        help=(
            "Use N_C=N//2, N_G=N-N_C with identical type params "
            "(default: N_C=N, N_G=0 workaround from verify_propositions.py)."
        ),
    )
    args = parser.parse_args()

    n_inits = args.n_inits if args.n_inits is not None else N_INITS

    print("=" * 70)
    print("PROPOSITION 7: Asymmetric Equilibria among Homogeneous Firms")
    print("=" * 70)
    print(
        f"Homogeneous: omega_C_E=omega_G_E={OMEGA_COMMON}, "
        f"mu_C=mu_G={MU_COMMON}"
    )
    print(
        f"Parameters: delta={DELTA_FIXED}, psi={PSI_FIXED}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}"
    )
    print(
        f"DEV_MODE={DEV_MODE}  "
        f"(N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"tau_bar={BASE_PARAMS['tau_bar']}, n_inits={n_inits})"
    )
    print(
        f"NPL: max_iter={NPL_MAX_ITER}, tol={NPL_TOL}, damping={NPL_DAMPING}"
    )
    if args.firm_split:
        print("Firm mode: split N_C/N_G with identical type params")
    else:
        print("Firm mode: N_C=N, N_G=0 (verify_propositions.py workaround)")
    print(f"Output file: {OUTPUT_TXT}")
    print()

    log_lines = []
    result = run_verification(
        log_lines, n_inits=n_inits, firm_split=args.firm_split
    )

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
        f.write("\n")

    print()
    print("=" * 70)
    print(f"Verification complete. Results written to:\n  {OUTPUT_TXT}")
    print("=" * 70)

    print_summary(result)


if __name__ == "__main__":
    main()
