"""
Verify Proposition 5: Risk-Type Sorting in Asymmetric Equilibria.

Sweep omega_C_E (conservative error cost) with omega_G_E fixed at 1.0.
Show that C-types shift from agentic (H) toward HITL (L) as omega_C_E rises,
while G-types remain agentic; sorting intensity agent(G) - agent(C) increases
monotonically.

Mirrors prop5_risk_type_sorting() in verify_propositions.py, with
ModelParams aligned to verify_equilibrium_existence.py (delta_x_base,
delta_x_comp, DEV_MODE).

Usage (from project root):
    python Prop5/verify_proposition5.py
    python Prop5/verify_proposition5.py --omega-ce 1.0 3.0 8.0

Output: Prop5/prop5_risk_sorting_verification.txt
"""

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import ModelParams, StateSpace, compute_payoffs, A_NONE, A_HITL, A_AGENT
from npl import run_npl, classify_equilibrium
from verify_equilibrium_existence import (
    DELTA_X_BASE,
    DELTA_X_COMP,
    DEV_MODE,
    BASE_PARAMS,
    NPL_MAX_ITER,
    NPL_TOL,
    NPL_DAMPING,
    RESIDUAL_VF_ITER,
    compute_equilibrium_stats,
    equilibrium_residual,
)

# Fixed parameters (verify_propositions.py prop5 + RA Guide)
DELTA_FIXED = 3.0
PSI_FIXED = 1.5
OMEGA_G_E_FIXED = 1.0
OMEGA_C_E_VALUES = [1.0, 2.0, 3.0, 5.0, 8.0]

# G-types "remain at H": mean agent rate should stay above this floor.
G_AGENT_FLOOR = 0.30
# Tolerance for monotonicity checks (numerical NPL noise).
MONO_TOL = 1e-4

OUTPUT_TXT = os.path.join(SCRIPT_DIR, "prop5_risk_sorting_verification.txt")


def build_params(omega_C_E):
    """ModelParams for one omega_C_E sweep point."""
    return ModelParams(
        **BASE_PARAMS,
        delta=DELTA_FIXED,
        psi=PSI_FIXED,
        omega_C_E=omega_C_E,
        omega_G_E=OMEGA_G_E_FIXED,
        delta_x_base=DELTA_X_BASE,
        delta_x_comp=DELTA_X_COMP,
    )


def build_asymmetric_init(n_states):
    """
    Asymmetric seed from verify_propositions.py prop5_risk_type_sorting().

    C-types biased toward HITL; G-types biased toward agentic deployment.
    """
    P_C_init = np.zeros((n_states, 3))
    P_C_init[:, A_NONE] = 0.1
    P_C_init[:, A_HITL] = 0.7
    P_C_init[:, A_AGENT] = 0.2

    P_G_init = np.zeros((n_states, 3))
    P_G_init[:, A_NONE] = 0.1
    P_G_init[:, A_HITL] = 0.2
    P_G_init[:, A_AGENT] = 0.7

    return P_C_init, P_G_init


def format_equilibrium_stats(stats):
    """Format adoption / HITL / agent rates for one equilibrium."""
    return [
        f"       C: adopt={stats['adopt_C']:.3f} "
        f"(HITL={stats['hitl_C']:.3f}, agent={stats['agent_C']:.3f})",
        f"       G: adopt={stats['adopt_G']:.3f} "
        f"(HITL={stats['hitl_G']:.3f}, agent={stats['agent_G']:.3f})",
    ]


def check_monotone_increasing(values, tol=MONO_TOL):
    """True if each step is >= previous - tol."""
    if len(values) < 2:
        return True
    return all(values[i] >= values[i - 1] - tol for i in range(1, len(values)))


def check_monotone_decreasing(values, tol=MONO_TOL):
    """True if each step is <= previous + tol."""
    if len(values) < 2:
        return True
    return all(values[i] <= values[i - 1] + tol for i in range(1, len(values)))


def verify_sorting_criteria(summary_rows):
    """
    RA Guide key tests for Proposition 5.

    Returns boolean checks and an overall pass flag (all converged rows only).
    """
    converged = [r for r in summary_rows if r["converged"]]
    if len(converged) < 2:
        return {
            "sorting_monotone": False,
            "agent_C_decreasing": False,
            "agent_G_at_H": False,
            "all_pass": False,
        }

    sorting_gaps = [r["sorting"] for r in converged]
    agent_C_vals = [r["agent_C"] for r in converged]
    agent_G_vals = [r["agent_G"] for r in converged]

    checks = {
        "sorting_monotone": check_monotone_increasing(sorting_gaps),
        "agent_C_decreasing": check_monotone_decreasing(agent_C_vals),
        "agent_G_at_H": all(ag >= G_AGENT_FLOOR for ag in agent_G_vals),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def run_omega_ce_sweep(log_lines, omega_ce_values=None):
    """Sweep omega_C_E and collect risk-sorting statistics."""
    if omega_ce_values is None:
        omega_ce_values = OMEGA_C_E_VALUES

    params0 = build_params(omega_ce_values[0])
    ss = StateSpace(params0)
    N_C = params0.N // 2
    N_G = params0.N - N_C
    P_C_init, P_G_init = build_asymmetric_init(ss.n_states)

    log_lines.append("PROPOSITION 5: RISK-TYPE SORTING VERIFICATION")
    log_lines.append("=" * 105)
    log_lines.append(
        f"Fixed parameters: delta={DELTA_FIXED}, psi={PSI_FIXED}, "
        f"omega_G_E={OMEGA_G_E_FIXED}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}"
    )
    log_lines.append(
        f"State space (DEV_MODE={DEV_MODE}): "
        f"N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"T_bar={BASE_PARAMS['T_bar']}, tau_bar={BASE_PARAMS['tau_bar']}, "
        f"n_states={ss.n_states}"
    )
    log_lines.append(
        "Method: NPL from asymmetric init (verify_propositions.py prop5); "
        "sorting gap = agent(G) - agent(C)"
    )
    log_lines.append(
        "Key tests: C shifts H->L as omega_C_E rises; G stays at H; "
        "sorting gap increases monotonically"
    )
    log_lines.append(
        f"NPL settings: max_iter={NPL_MAX_ITER}, tol={NPL_TOL}, "
        f"damping={NPL_DAMPING}"
    )
    log_lines.append("")

    summary_rows = []

    for omega_C_E in omega_ce_values:
        print(f"[ omega_C_E = {omega_C_E} ]  running NPL...", flush=True)
        log_lines.append(f"[ omega_C_E = {omega_C_E} ]")

        params = build_params(omega_C_E)
        u_C, u_G = compute_payoffs(ss, params)

        result = run_npl(
            ss, params, N_C, N_G,
            P_C_init=P_C_init.copy(), P_G_init=P_G_init.copy(),
            max_iter=NPL_MAX_ITER, tol=NPL_TOL, damping=NPL_DAMPING,
            verbose=False,
        )

        if not result["converged"]:
            log_lines.append("  NPL: NOT CONVERGED")
            log_lines.append("")
            summary_rows.append({
                "omega_C_E": omega_C_E,
                "converged": False,
                "agent_C": np.nan,
                "hitl_C": np.nan,
                "agent_G": np.nan,
                "hitl_G": np.nan,
                "sorting": np.nan,
                "residual": np.nan,
                "eq_type": "N/A",
            })
            print(f"[ omega_C_E = {omega_C_E} ]  done — NOT CONVERGED", flush=True)
            continue

        P_C, P_G = result["P_C"], result["P_G"]
        stats = compute_equilibrium_stats(P_C, P_G)
        residual = equilibrium_residual(
            ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
            max_vf_iter=RESIDUAL_VF_ITER,
        )
        eq_type = classify_equilibrium(stats)
        sorting = stats["agent_G"] - stats["agent_C"]

        log_lines.append(f"  Equilibrium: {eq_type}")
        log_lines.extend(format_equilibrium_stats(stats))
        log_lines.append(f"       Sorting gap agent(G)-agent(C): {sorting:.3f}")
        log_lines.append(f"       Residual: {residual:.2e}")
        log_lines.append("")

        summary_rows.append({
            "omega_C_E": omega_C_E,
            "converged": True,
            "agent_C": stats["agent_C"],
            "hitl_C": stats["hitl_C"],
            "agent_G": stats["agent_G"],
            "hitl_G": stats["hitl_G"],
            "sorting": sorting,
            "residual": residual,
            "eq_type": eq_type,
        })

        print(
            f"[ omega_C_E = {omega_C_E} ]  done — "
            f"agent(C)={stats['agent_C']:.3f}, agent(G)={stats['agent_G']:.3f}, "
            f"sorting={sorting:.3f}",
            flush=True,
        )

    checks = verify_sorting_criteria(summary_rows)

    log_lines.append("=" * 105)
    log_lines.append("TABLE: Proposition 5 Verification Summary")
    log_lines.append("-" * 105)
    header = (
        f"{'omega_C_E':<10}| {'agent(C)':<10}| {'HITL(C)':<10}| "
        f"{'agent(G)':<10}| {'HITL(G)':<10}| {'sorting':<10}| "
        f"{'conv?':<6}| {'type'}"
    )
    log_lines.append(header)
    log_lines.append("-" * 105)

    for row in summary_rows:
        if row["converged"]:
            log_lines.append(
                f"{row['omega_C_E']:<10.1f}| "
                f"{row['agent_C']:<10.3f}| {row['hitl_C']:<10.3f}| "
                f"{row['agent_G']:<10.3f}| {row['hitl_G']:<10.3f}| "
                f"{row['sorting']:<10.3f}| "
                f"{'Yes':<6}| {row['eq_type']}"
            )
        else:
            log_lines.append(
                f"{row['omega_C_E']:<10.1f}| "
                f"{'N/A':<10}| {'N/A':<10}| "
                f"{'N/A':<10}| {'N/A':<10}| "
                f"{'N/A':<10}| "
                f"{'No':<6}| N/A"
            )

    log_lines.append("")
    log_lines.append("Key tests (RA Guide):")
    log_lines.append(
        f"  Sorting gap monotone increasing: "
        f"{'PASS' if checks['sorting_monotone'] else 'FAIL'}"
    )
    log_lines.append(
        f"  C agent rate decreasing (H -> L): "
        f"{'PASS' if checks['agent_C_decreasing'] else 'FAIL'}"
    )
    log_lines.append(
        f"  G agent rate stays at H (>= {G_AGENT_FLOOR}): "
        f"{'PASS' if checks['agent_G_at_H'] else 'FAIL'}"
    )
    log_lines.append(
        f"  Overall Proposition 5 check: "
        f"{'PASS' if checks['all_pass'] else 'PARTIAL'}"
    )

    return summary_rows, checks


def print_summary_table(summary_rows, checks):
    """Print summary table to console."""
    print()
    print("TABLE: Proposition 5 Summary")
    print("-" * 85)
    print(
        f"{'omega_C_E':<10} {'agent(C)':<10} {'HITL(C)':<10} "
        f"{'agent(G)':<10} {'HITL(G)':<10} {'sorting':<10} {'conv?'}"
    )
    print("-" * 85)
    for row in summary_rows:
        if row["converged"]:
            print(
                f"{row['omega_C_E']:<10.1f} "
                f"{row['agent_C']:<10.3f} {row['hitl_C']:<10.3f} "
                f"{row['agent_G']:<10.3f} {row['hitl_G']:<10.3f} "
                f"{row['sorting']:<10.3f} Yes"
            )
        else:
            print(
                f"{row['omega_C_E']:<10.1f} "
                f"{'N/A':<10} {'N/A':<10} "
                f"{'N/A':<10} {'N/A':<10} "
                f"{'N/A':<10} No"
            )
    print()
    print(
        f"Prop5 OK? {checks['all_pass']}  "
        f"(sorting mono: {checks['sorting_monotone']}, "
        f"C decreasing: {checks['agent_C_decreasing']}, "
        f"G at H: {checks['agent_G_at_H']})"
    )


def main():
    """Run Proposition 5 verification and write results to file."""
    parser = argparse.ArgumentParser(
        description="Verify Proposition 5: Risk-Type Sorting."
    )
    parser.add_argument(
        "--omega-ce",
        type=float,
        nargs="*",
        default=None,
        dest="omega_ce",
        help="Optional subset of omega_C_E values (default: full sweep).",
    )
    args = parser.parse_args()

    omega_ce_values = args.omega_ce if args.omega_ce else OMEGA_C_E_VALUES

    print("=" * 70)
    print("PROPOSITION 5: Risk-Type Sorting Verification")
    print("=" * 70)
    print(
        f"Parameters: delta={DELTA_FIXED}, psi={PSI_FIXED}, "
        f"omega_G_E={OMEGA_G_E_FIXED}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}"
    )
    print(f"omega_C_E sweep: {omega_ce_values}")
    print(f"DEV_MODE={DEV_MODE}  "
          f"(N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
          f"tau_bar={BASE_PARAMS['tau_bar']})")
    print(f"Output file: {OUTPUT_TXT}")
    print()

    log_lines = []
    summary_rows, checks = run_omega_ce_sweep(
        log_lines, omega_ce_values=omega_ce_values,
    )

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
        f.write("\n")

    print()
    print("=" * 70)
    print(f"Verification complete. Results written to:\n  {OUTPUT_TXT}")
    print("=" * 70)

    print_summary_table(summary_rows, checks)


if __name__ == "__main__":
    main()
