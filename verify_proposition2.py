"""
Verify Proposition 2: Adoption Cascade Dynamics.

From the low-adoption equilibrium P^L, exogenously shock k competitors to upgrade
(HITL). Show that there exists a threshold k = A_bar such that for k < A_bar the
system returns to low adoption, and for k >= A_bar adoption cascades upward.

P^L baseline: identical direct verification pipeline as verify_equilibrium_existence.py
(Section 2.5 — verify_low_adoption_eq + refine_low_eq_on_reachable; no NPL for P^L).

Post-shock: NPL from shocked init (same as prop2_cascade() in verify_propositions.py).
Cascade is detected by adoption-rate jump; P^H is not searched or classified.

Usage (from project root):
    python Prop2/verify_proposition2.py
    python Prop2/verify_proposition2.py --delta 2.0 3.0

Output: Prop2/prop2_adoption_cascade_verification.txt
"""

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import ModelParams, compute_payoffs, A_HITL, A_AGENT
from npl import run_npl, npl_iteration
from verify_equilibrium_existence import (
    PSI_FIXED,
    DELTA_X_BASE,
    DELTA_X_COMP,
    DEV_MODE,
    BASE_PARAMS,
    NPL_MAX_ITER,
    NPL_TOL,
    NPL_DAMPING,
    DISTINCT_TOL,
    make_sweep_context,
    verify_low_adoption_eq,
    refine_low_eq_on_reachable,
    compute_equilibrium_stats,
    ccp_distance,
    equilibrium_residual,
    equilibrium_residual_on_mask,
)

DELTA_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
CASCADE_ADOPT_MARGIN = 0.2

OUTPUT_TXT = os.path.join(SCRIPT_DIR, "prop2_adoption_cascade_verification.txt")


def build_params_for_delta(delta):
    """ModelParams for one delta — same construction as verify_equilibrium_existence.py."""
    return ModelParams(
        **BASE_PARAMS,
        delta=delta,
        psi=PSI_FIXED,
        delta_x_base=DELTA_X_BASE,
        delta_x_comp=DELTA_X_COMP,
    )


def format_equilibrium_stats(stats):
    """Format adoption / HITL / agent rates for one equilibrium."""
    return [
        f"       C: adopt={stats['adopt_C']:.3f} "
        f"(HITL={stats['hitl_C']:.3f}, agent={stats['agent_C']:.3f})",
        f"       G: adopt={stats['adopt_G']:.3f} "
        f"(HITL={stats['hitl_G']:.3f}, agent={stats['agent_G']:.3f})",
    ]


def get_pl_profile(ss, params, N_C, N_G, u_C, u_G, log_lines):
    """
    Obtain verified P^L via direct fixed-point check (Section 2.5).

    Pipeline matches verify_equilibrium_existence.py:
      1. verify_low_adoption_eq (BR check on reachable states)
      2. refine_low_eq_on_reachable (undamped BR on ergodic component)
      3. reachable + global residuals for reporting
    """
    low_exists, _P_cand, _br_results, reach_mask = verify_low_adoption_eq(
        params, ss, N_C, N_G, u_C=u_C, u_G=u_G, verbose=False,
    )
    n_reach = int(reach_mask.sum())
    log_lines.append(
        f"  Reachable states under low-adoption policy: {n_reach}"
    )

    if not low_exists:
        log_lines.append("  WARNING: P^L direct verification FAILED")
        return None, reach_mask, False

    P_C, P_G = refine_low_eq_on_reachable(
        ss, params, N_C, N_G, u_C, u_G, reach_mask,
    )
    stats = compute_equilibrium_stats(P_C, P_G)
    P_C_br, P_G_br = npl_iteration(
        ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
    )
    res_global = equilibrium_residual(
        ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
    )
    res_reach = equilibrium_residual_on_mask(
        P_C, P_G, P_C_br, P_G_br, reach_mask,
    )

    log_lines.append("  P^L (low adoption) baseline — direct verification (Prop1):")
    log_lines.extend(format_equilibrium_stats(stats))
    log_lines.append(f"       Residual (reachable):  {res_reach:.2e}")
    log_lines.append(f"       Residual (global):     {res_global:.2e}")
    log_lines.append(
        "       Note: global residual may exceed 1e-8 off-path because P^L is "
        "NPL-unstable outside the ergodic component."
    )
    return {
        "P_C": P_C, "P_G": P_G,
        "stats": stats,
        "residual": res_reach,
        "residual_global": res_global,
    }, reach_mask, True


def apply_shock(P_C_base, P_G_base, ss, k):
    """
    Exogenous shock: force k competitors to upgrade (HITL) from P^L.

    k=0: no shock — return P^L unchanged; cascade sweep skips NPL (zero forced
    upgraders = baseline control, not a shock experiment).
    k>=1: bias CCPs toward upgrading in states with A_L >= k.  Condition A_L >= k
    (not A_L == k-1) matches the RA cascade protocol: perturb states where at least
    k competitors have already adopted, simulating a coordinated upgrade of k firms.
    """
    P_C = P_C_base.copy()
    P_G = P_G_base.copy()
    if k == 0:
        return P_C, P_G
    for idx in range(ss.n_states):
        _x, A_L, _A_H, _tau, _T = tuple(ss.states[idx])
        if A_L >= k:
            P_C[idx] = [0.1, 0.7, 0.2]
            P_G[idx] = [0.1, 0.3, 0.6]
    return P_C, P_G


def mean_adopt_rate(P_C, P_G):
    """Average adoption rate across firm types."""
    return (
        (P_C[:, A_HITL] + P_C[:, A_AGENT]).mean() +
        (P_G[:, A_HITL] + P_G[:, A_AGENT]).mean()
    ) / 2


def classify_cascade_outcome(P_C, P_G, pl_profile, baseline_adopt):
    """
    Classify post-shock outcome using adoption jump and distance to P^L.

    Cascade = adopt_jump > CASCADE_ADOPT_MARGIN (no P^H reference).
    """
    adopt_after = mean_adopt_rate(P_C, P_G)
    adopt_jump = adopt_after - baseline_adopt
    cascaded = adopt_jump > CASCADE_ADOPT_MARGIN
    d_L = ccp_distance(P_C, P_G, pl_profile["P_C"], pl_profile["P_G"])
    near_L = d_L < DISTINCT_TOL * 3

    if cascaded:
        outcome = "cascade"
    elif near_L:
        outcome = "P^L"
    else:
        outcome = "low"

    return {
        "adopt_after": adopt_after,
        "adopt_jump": adopt_jump,
        "d_to_PL": d_L,
        "outcome": outcome,
        "cascaded": cascaded,
    }


def run_cascade_sweep_for_delta(delta, ss, N_C, N_G, log_lines):
    """Run Proposition 2 verification for one delta."""
    params = build_params_for_delta(delta)
    u_C, u_G = compute_payoffs(ss, params)

    log_lines.append(f"[ delta = {delta} ]")
    log_lines.append(
        f"  Parameters: psi={PSI_FIXED}, N={params.N}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP} "
        f"(ModelParams matches verify_equilibrium_existence.py)"
    )

    pl_profile, _reach_mask, pl_ok = get_pl_profile(
        ss, params, N_C, N_G, u_C, u_G, log_lines,
    )
    if not pl_ok:
        log_lines.append("")
        return {
            "delta": delta,
            "pl_exists": False,
            "A_bar": np.nan,
            "prop2_pass": False,
            "cascade_rows": [],
        }

    baseline_adopt = mean_adopt_rate(pl_profile["P_C"], pl_profile["P_G"])

    log_lines.append("")
    log_lines.append(
        f"  Cascade test: force k competitors to upgrade (HITL); "
        f"baseline adopt={baseline_adopt:.3f}; k=0 is no-shock baseline (no NPL)"
    )
    log_lines.append(
        f"  {'k':>4}  {'adopt_after':>12}  {'jump':>8}  "
        f"{'d(P^L)':>10}  {'outcome':>8}  {'cascade?':>9}"
    )
    log_lines.append("  " + "-" * 58)

    cascade_rows = []
    for k in range(params.N):
        P_C_shock, P_G_shock = apply_shock(
            pl_profile["P_C"], pl_profile["P_G"], ss, k,
        )

        if k == 0:
            # Zero forced upgraders: P^L unchanged; no NPL (baseline control).
            info = classify_cascade_outcome(
                P_C_shock, P_G_shock, pl_profile, baseline_adopt,
            )
            row = {"k": k, "converged": True, **info}
            log_lines.append(
                f"  {k:>4}  {row['adopt_after']:>12.3f}  "
                f"{row['adopt_jump']:>8.3f}  "
                f"{row['d_to_PL']:>10.4f}  "
                f"{'baseline':>8}  "
                f"{'no':>9}"
            )
            cascade_rows.append(row)
            continue

        result = run_npl(
            ss, params, N_C, N_G,
            P_C_init=P_C_shock, P_G_init=P_G_shock,
            max_iter=NPL_MAX_ITER, tol=NPL_TOL, damping=NPL_DAMPING,
            verbose=False,
        )

        if not result["converged"]:
            row = {
                "k": k,
                "converged": False,
                "adopt_after": np.nan,
                "adopt_jump": np.nan,
                "d_to_PL": np.nan,
                "outcome": "no conv",
                "cascaded": False,
            }
            log_lines.append(
                f"  {k:>4}  {'(not conv)':>12}  {'':>8}  "
                f"{'':>10}  {'no conv':>8}  {'?':>9}"
            )
        else:
            info = classify_cascade_outcome(
                result["P_C"], result["P_G"],
                pl_profile, baseline_adopt,
            )
            row = {"k": k, "converged": True, **info}
            log_lines.append(
                f"  {k:>4}  {row['adopt_after']:>12.3f}  "
                f"{row['adopt_jump']:>8.3f}  "
                f"{row['d_to_PL']:>10.4f}  "
                f"{row['outcome']:>8}  "
                f"{'YES' if row['cascaded'] else 'no':>9}"
            )
        cascade_rows.append(row)

    cascaded_ks = [r["k"] for r in cascade_rows if r.get("cascaded")]
    if cascaded_ks:
        A_bar = min(cascaded_ks)
        log_lines.append("")
        log_lines.append(
            f"  Cascade threshold A_bar = {A_bar} "
            f"(k < {A_bar} -> low adoption, k >= {A_bar} -> cascade)"
        )
        below = [r for r in cascade_rows if r["k"] < A_bar]
        above = [r for r in cascade_rows if r["k"] >= A_bar]
        returns_ok = all(not r.get("cascaded") for r in below) if below else True
        cascade_ok = all(r.get("cascaded") for r in above) if above else True
        prop2_pass = returns_ok and cascade_ok
        log_lines.append(
            f"  Proposition 2 check: "
            f"{'PASS' if prop2_pass else 'PARTIAL'} "
            f"(no cascade below A_bar: {'OK' if returns_ok else 'FAIL'}, "
            f"cascade at/above A_bar: {'OK' if cascade_ok else 'FAIL'})"
        )
    else:
        A_bar = np.nan
        log_lines.append("")
        log_lines.append("  Cascade threshold A_bar: NOT OBSERVED (no cascade for any k)")
        prop2_pass = False

    log_lines.append("")
    print(
        f"[ delta = {delta} ]  done — "
        f"A_bar={A_bar if not np.isnan(A_bar) else 'N/A'}, "
        f"P^L={'Yes' if pl_ok else 'No'}",
        flush=True,
    )

    return {
        "delta": delta,
        "pl_exists": pl_ok,
        "baseline_adopt": baseline_adopt,
        "A_bar": A_bar,
        "prop2_pass": prop2_pass if cascaded_ks else False,
        "cascade_rows": cascade_rows,
    }


def run_delta_sweep(log_lines, delta_values=None):
    """Sweep delta and collect Proposition 2 summary rows."""
    if delta_values is None:
        delta_values = DELTA_VALUES

    ss, N_C, N_G, _inits = make_sweep_context()

    log_lines.append("PROPOSITION 2: ADOPTION CASCADE DYNAMICS VERIFICATION")
    log_lines.append("=" * 105)
    log_lines.append(
        f"Fixed parameters: psi={PSI_FIXED}, delta_x_base={DELTA_X_BASE}, "
        f"delta_x_comp={DELTA_X_COMP}"
    )
    log_lines.append(
        f"State space (DEV_MODE={DEV_MODE}): "
        f"N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"T_bar={BASE_PARAMS['T_bar']}, tau_bar={BASE_PARAMS['tau_bar']}, "
        f"n_states={ss.n_states}"
    )
    log_lines.append(
        "P^L baseline: direct verification (Section 2.5) — same pipeline as "
        "verify_equilibrium_existence.py (no NPL for P^L)"
    )
    log_lines.append(
        "Cascade: k=0 no shock (P^L baseline, no NPL); k>=1 shock states with A_L >= k "
        f"toward HITL; NPL from init (max_iter={NPL_MAX_ITER}, tol={NPL_TOL}, "
        f"damping={NPL_DAMPING})"
    )
    log_lines.append(
        f"Cascade criterion: adopt jump > {CASCADE_ADOPT_MARGIN} "
        f"(P^H not searched)"
    )
    log_lines.append("")

    summary_rows = []
    for delta in delta_values:
        print(f"[ delta = {delta} ]  running cascade sweep...", flush=True)
        row = run_cascade_sweep_for_delta(delta, ss, N_C, N_G, log_lines)
        summary_rows.append(row)

    log_lines.append("=" * 105)
    log_lines.append("TABLE: Proposition 2 Verification Summary")
    log_lines.append("-" * 105)
    header = (
        f"{'delta':<7}| {'P^L?':<6}| {'A_bar':<7}| "
        f"{'base adopt':<11}| {'Prop2 OK?':<10}| "
        f"cascade path (k: outcome)"
    )
    log_lines.append(header)
    log_lines.append("-" * 105)

    for row in summary_rows:
        A_str = (
            f"{int(row['A_bar'])}"
            if row["A_bar"] is not None and not np.isnan(row["A_bar"])
            else "N/A"
        )
        base_adopt = (
            f"{row['baseline_adopt']:.3f}"
            if row.get("baseline_adopt") is not None
            else "N/A"
        )
        path = ", ".join(
            f"{r['k']}:{r['outcome']}"
            for r in row.get("cascade_rows", [])
        )
        log_lines.append(
            f"{row['delta']:<7.1f}| "
            f"{str(row['pl_exists']):<6}| "
            f"{A_str:<7}| "
            f"{base_adopt:<11}| "
            f"{str(row.get('prop2_pass', False)):<10}| "
            f"{path}"
        )

    return summary_rows


def print_summary_table(summary_rows):
    """Print summary table to console."""
    print()
    print("TABLE: Proposition 2 Summary")
    print("-" * 55)
    print(f"{'delta':<7} {'P^L?':<6} {'A_bar':<7} {'Prop2 OK?':<10}")
    print("-" * 55)
    for row in summary_rows:
        A_str = (
            f"{int(row['A_bar'])}"
            if row["A_bar"] is not None and not np.isnan(row["A_bar"])
            else "N/A"
        )
        print(
            f"{row['delta']:<7.1f} "
            f"{str(row['pl_exists']):<6} "
            f"{A_str:<7} "
            f"{str(row.get('prop2_pass', False)):<10}"
        )


def main():
    """Run Proposition 2 verification and write results to file."""
    parser = argparse.ArgumentParser(
        description="Verify Proposition 2: Adoption Cascade Dynamics."
    )
    parser.add_argument(
        "--delta",
        type=float,
        nargs="*",
        default=None,
        help="Optional subset of delta values (default: full sweep).",
    )
    args = parser.parse_args()

    delta_values = args.delta if args.delta else DELTA_VALUES

    print("=" * 70)
    print("PROPOSITION 2: Adoption Cascade Dynamics Verification")
    print("=" * 70)
    print(
        f"Parameters: psi={PSI_FIXED}, delta_x_base={DELTA_X_BASE}, "
        f"delta_x_comp={DELTA_X_COMP}"
    )
    print(f"Delta sweep: {delta_values}")
    print(f"DEV_MODE={DEV_MODE}  "
          f"(N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']})")
    print(f"Output file: {OUTPUT_TXT}")
    print()

    log_lines = []
    summary_rows = run_delta_sweep(log_lines, delta_values=delta_values)

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
