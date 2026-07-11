"""
Verify Proposition 3: Cautious Coordination Equilibrium.

Sweep psi (trust damage intensity) with delta fixed at 3.0 and show that for
large psi a P^M equilibrium emerges: firms adopt HITL but avoid agentic deployment.

Uses the same NPL / find_all_equilibria logic as prop3_cautious_coordination()
in verify_propositions.py, with equilibrium residual values and trust-capital
reporting at each equilibrium (per RA Guide Section on Proposition 3).
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
from npl import (
    npl_iteration,
    find_all_equilibria,
    classify_equilibrium,
    generate_initial_profiles,
)

# ---------------------------------------------------------------------------
# Fixed parameters for Proposition 3 verification
# (mirrors verify_propositions.py prop3 + RA Guide sweep)
# ---------------------------------------------------------------------------
DELTA_FIXED = 3.0
PSI_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

DEV_MODE = True
if DEV_MODE:
    BASE_PARAMS = dict(N=4, x_bar=5, T_bar=3, tau_bar=3)
    N_INITS = 20
else:
    BASE_PARAMS = dict(N=6, x_bar=9, T_bar=4, tau_bar=4)
    N_INITS = 50

NPL_TOL = 1e-6
DISTINCT_TOL = 0.01
RESIDUAL_VF_ITER = 200

OUTPUT_TXT = os.path.join(SCRIPT_DIR, "prop3_cautious_coordination_verification.txt")


def equilibrium_residual(ss, params, P_C, P_G, N_C, N_G, u_C=None, u_G=None,
                         max_vf_iter=RESIDUAL_VF_ITER):
    """Compute ||P - Psi(P)||: max-norm distance from a true fixed point."""
    if u_C is None or u_G is None:
        u_C, u_G = compute_payoffs(ss, params)
    P_C_br, P_G_br = npl_iteration(
        ss, params, P_C, P_G, N_C, N_G, u_C, u_G, max_vf_iter=max_vf_iter
    )
    return max(
        np.max(np.abs(P_C - P_C_br)),
        np.max(np.abs(P_G - P_G_br)),
    )


def compute_equilibrium_stats(P_C, P_G):
    """Average adoption / HITL / agent rates for types C and G."""
    return {
        "adopt_C": (P_C[:, A_HITL] + P_C[:, A_AGENT]).mean(),
        "adopt_G": (P_G[:, A_HITL] + P_G[:, A_AGENT]).mean(),
        "hitl_C": P_C[:, A_HITL].mean(),
        "hitl_G": P_G[:, A_HITL].mean(),
        "agent_C": P_C[:, A_AGENT].mean(),
        "agent_G": P_G[:, A_AGENT].mean(),
    }


def ccp_distance(P_C_a, P_G_a, P_C_b, P_G_b):
    """Max-norm distance between two CCP profiles."""
    return max(
        np.max(np.abs(P_C_a - P_C_b)),
        np.max(np.abs(P_G_a - P_G_b)),
    )


def compute_trust_capital_stats(ss, params, P_C, P_G):
    """
    Trust-capital diagnostics at an equilibrium profile.

    Adoption-weighted mean tau and the share of adoption-weighted mass at
    tau = tau_bar, reported for informational output only.
    """
    tau = ss.states[:, 3].astype(float)
    w_C = P_C[:, A_HITL] + P_C[:, A_AGENT]
    w_G = P_G[:, A_HITL] + P_G[:, A_AGENT]
    w = (w_C + w_G) / 2.0
    w_sum = w.sum()

    if w_sum > 0:
        mean_tau = float(np.average(tau, weights=w))
        at_max_mask = tau >= params.tau_bar
        frac_at_tau_bar = float(w[at_max_mask].sum() / w_sum)
    else:
        mean_tau = float(tau.mean())
        frac_at_tau_bar = float((tau >= params.tau_bar).mean())

    norm_tau = mean_tau / params.tau_bar if params.tau_bar > 0 else 0.0
    return {
        "mean_tau": mean_tau,
        "norm_tau": norm_tau,
        "frac_at_tau_bar": frac_at_tau_bar,
    }


def pick_cautious_eq(equilibria):
    """Return the P^M (cautious) equilibrium if present."""
    for eq in equilibria:
        if "cautious" in classify_equilibrium(eq["stats"]).lower():
            return eq
    return None


def pick_low_eq(equilibria):
    """Return the P^L (low adoption) equilibrium if present."""
    for eq in equilibria:
        if "low" in classify_equilibrium(eq["stats"]).lower():
            return eq
    return None


def pick_high_eq(equilibria):
    """Return the P^H (aggressive) equilibrium if present."""
    for eq in equilibria:
        eq_type = classify_equilibrium(eq["stats"])
        if "aggressive" in eq_type.lower() or "agentic" in eq_type.lower():
            return eq
    return None


def format_equilibrium_stats(stats):
    """Format adoption / HITL / agent rates for one equilibrium."""
    return [
        f"       C: adopt={stats['adopt_C']:.3f} "
        f"(HITL={stats['hitl_C']:.3f}, agent={stats['agent_C']:.3f})",
        f"       G: adopt={stats['adopt_G']:.3f} "
        f"(HITL={stats['hitl_G']:.3f}, agent={stats['agent_G']:.3f})",
    ]


def format_trust_capital(trust, tau_bar):
    """Format trust-capital diagnostics."""
    return [
        f"       Trust capital: mean_tau={trust['mean_tau']:.3f} "
        f"(tau/tau_bar={trust['norm_tau']:.3f}), "
        f"frac_at_tau_bar={trust['frac_at_tau_bar']:.3f}",
        f"       (tau_bar={tau_bar})",
    ]


def verify_pm_criteria(stats):
    """
    Check RA Guide key tests for P^M.

    Returns dict of boolean checks and an overall pass flag.
    """
    total_adopt = (stats["adopt_C"] + stats["adopt_G"]) / 2
    total_hitl = (stats["hitl_C"] + stats["hitl_G"]) / 2
    total_agent = (stats["agent_C"] + stats["agent_G"]) / 2

    checks = {
        "distinct_from_PL": total_adopt >= 0.3,
        "distinct_from_PH": total_agent < 0.2,
        "high_HITL": total_hitl > 0.4,
        "low_agent": total_agent < 0.2,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def enrich_equilibria(ss, params, N_C, N_G, equilibria, u_C, u_G):
    """Attach type, residual, and trust-capital stats to each equilibrium."""
    enriched = []
    for eq in equilibria:
        P_C, P_G = eq["P_C"], eq["P_G"]
        stats = eq.get("stats") or compute_equilibrium_stats(P_C, P_G)
        residual = equilibrium_residual(
            ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
            max_vf_iter=RESIDUAL_VF_ITER,
        )
        trust = compute_trust_capital_stats(ss, params, P_C, P_G)
        enriched.append({
            **eq,
            "stats": stats,
            "type": classify_equilibrium(stats),
            "residual": residual,
            "trust": trust,
        })
    return enriched


def run_psi_sweep(log_lines):
    """Main sweep over psi values with P^M verification."""
    summary_rows = []
    params0 = ModelParams(**BASE_PARAMS, delta=DELTA_FIXED, psi=PSI_VALUES[0])
    ss = StateSpace(params0)
    N_C = params0.N // 2
    N_G = params0.N - N_C

    log_lines.append("PROPOSITION 3: CAUTIOUS COORDINATION EQUILIBRIUM VERIFICATION")
    log_lines.append("=" * 105)
    log_lines.append(f"Fixed parameters: delta={DELTA_FIXED}")
    log_lines.append(
        f"State space (DEV_MODE={DEV_MODE}): "
        f"N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"T_bar={BASE_PARAMS['T_bar']}, tau_bar={BASE_PARAMS['tau_bar']}, "
        f"n_states={ss.n_states}, N_INITS={N_INITS}"
    )
    log_lines.append(
        "Method: find_all_equilibria (same as verify_propositions.py prop3); "
        "report residual ||P - Psi(P)|| and adoption-weighted trust capital"
    )
    log_lines.append(
        "Key test: for large psi, P^M has high HITL and low agent"
    )
    log_lines.append("")

    for psi in PSI_VALUES:
        print(f"[ psi = {psi} ]  finding equilibria...", flush=True)
        log_lines.append(f"[ psi = {psi} ]")

        params = ModelParams(**BASE_PARAMS, delta=DELTA_FIXED, psi=psi)
        u_C, u_G = compute_payoffs(ss, params)

        raw_eqs = find_all_equilibria(
            ss, params, N_C, N_G,
            n_inits=N_INITS, tol=NPL_TOL, verbose=False,
        )
        equilibria = enrich_equilibria(ss, params, N_C, N_G, raw_eqs, u_C, u_G)

        if not equilibria:
            log_lines.append("  No equilibria found")
            log_lines.append("")
            summary_rows.append({
                "psi": psi,
                "n_equilibria": 0,
                "PM_exists": False,
                "PM_residual": np.nan,
                "PM_mean_tau": np.nan,
                "PM_norm_tau": np.nan,
                "PM_checks_pass": False,
                "types": "",
            })
            print(f"[ psi = {psi} ]  done — no equilibria found", flush=True)
            continue

        for j, eq in enumerate(equilibria, start=1):
            log_lines.append(f"  Equilibrium {j}: {eq['type']}")
            log_lines.extend(format_equilibrium_stats(eq["stats"]))
            log_lines.append(f"       Residual: {eq['residual']:.2e}")
            log_lines.extend(format_trust_capital(eq["trust"], params.tau_bar))

        eq_M = pick_cautious_eq(equilibria)
        eq_L = pick_low_eq(equilibria)
        eq_H = pick_high_eq(equilibria)

        if eq_M is not None:
            checks = verify_pm_criteria(eq_M["stats"])
            log_lines.append("")
            log_lines.append("  P^M key tests (RA Guide):")
            log_lines.append(
                f"       Distinct from P^L (adopt >= 0.3): "
                f"{'PASS' if checks['distinct_from_PL'] else 'FAIL'}"
            )
            log_lines.append(
                f"       Distinct from P^H (agent < 0.2):  "
                f"{'PASS' if checks['distinct_from_PH'] else 'FAIL'}"
            )
            log_lines.append(
                f"       High HITL (hitl > 0.4):           "
                f"{'PASS' if checks['high_HITL'] else 'FAIL'}"
            )
            log_lines.append(
                f"       Low agent (agent < 0.2):          "
                f"{'PASS' if checks['low_agent'] else 'FAIL'}"
            )
            log_lines.append(
                f"       Overall P^M verification:         "
                f"{'PASS' if checks['all_pass'] else 'PARTIAL'}"
            )
        else:
            checks = None
            log_lines.append("")
            log_lines.append("  P^M: NOT FOUND at this psi")

        if len(equilibria) >= 2:
            log_lines.append("")
            log_lines.append("  Distinctness (max-norm over CCPs):")
            log_lines.append(f"  {'Pair':<24} {'Distance':>12}")
            log_lines.append(f"  {'-'*24} {'-'*12}")
            for i in range(len(equilibria)):
                for k in range(i + 1, len(equilibria)):
                    d = ccp_distance(
                        equilibria[i]["P_C"], equilibria[i]["P_G"],
                        equilibria[k]["P_C"], equilibria[k]["P_G"],
                    )
                    ti = equilibria[i]["type"]
                    tk = equilibria[k]["type"]
                    pad = max(0, 14 - len(ti) - len(tk))
                    log_lines.append(f"  ||{ti} - {tk}||{' ' * pad}{d:.4f}")

            if eq_M and eq_L:
                d_ML = ccp_distance(
                    eq_M["P_C"], eq_M["P_G"], eq_L["P_C"], eq_L["P_G"]
                )
                log_lines.append(f"  P^M vs P^L distance: {d_ML:.4f}")
            if eq_M and eq_H:
                d_MH = ccp_distance(
                    eq_M["P_C"], eq_M["P_G"], eq_H["P_C"], eq_H["P_G"]
                )
                log_lines.append(f"  P^M vs P^H distance: {d_MH:.4f}")

        log_lines.append("")

        types_str = ", ".join(eq["type"] for eq in equilibria)
        summary_rows.append({
            "psi": psi,
            "n_equilibria": len(equilibria),
            "PM_exists": eq_M is not None,
            "PM_residual": eq_M["residual"] if eq_M else np.nan,
            "PM_mean_tau": eq_M["trust"]["mean_tau"] if eq_M else np.nan,
            "PM_norm_tau": eq_M["trust"]["norm_tau"] if eq_M else np.nan,
            "PM_hitl_C": eq_M["stats"]["hitl_C"] if eq_M else np.nan,
            "PM_agent_C": eq_M["stats"]["agent_C"] if eq_M else np.nan,
            "PM_checks_pass": checks["all_pass"] if checks else False,
            "types": types_str,
        })

        pm_str = "Yes" if eq_M else "No"
        res_str = (
            f"{eq_M['residual']:.2e}"
            if eq_M is not None else "N/A"
        )
        print(
            f"[ psi = {psi} ]  done — "
            f"P^M={pm_str}, n_eq={len(equilibria)}, PM_res={res_str}",
            flush=True,
        )

    log_lines.append("=" * 105)
    log_lines.append(
        f"TABLE: Proposition 3 Verification Summary (delta fixed at {DELTA_FIXED})"
    )
    log_lines.append("-" * 105)
    header = (
        f"{'psi':<7}| {'#Eq':<5}| {'P^M?':<6}| {'PM Residual':<14}| "
        f"{'mean_tau':<10}| {'tau/tau_b':<10}| "
        f"{'HITL(C)':<9}| {'agent(C)':<9}| {'PM OK?':<7}| {'types'}"
    )
    log_lines.append(header)
    log_lines.append("-" * 105)

    for row in summary_rows:
        pm_res = (
            f"{row['PM_residual']:.2e}"
            if row["PM_exists"] and not np.isnan(row["PM_residual"])
            else "N/A"
        )
        mean_tau = (
            f"{row['PM_mean_tau']:.3f}"
            if row["PM_exists"] and not np.isnan(row["PM_mean_tau"])
            else "N/A"
        )
        norm_tau = (
            f"{row['PM_norm_tau']:.3f}"
            if row["PM_exists"] and not np.isnan(row["PM_norm_tau"])
            else "N/A"
        )
        hitl_C = (
            f"{row['PM_hitl_C']:.3f}"
            if row["PM_exists"] and not np.isnan(row["PM_hitl_C"])
            else "N/A"
        )
        agent_C = (
            f"{row['PM_agent_C']:.3f}"
            if row["PM_exists"] and not np.isnan(row["PM_agent_C"])
            else "N/A"
        )
        log_lines.append(
            f"{row['psi']:<7.1f}| "
            f"{row['n_equilibria']:<5}| "
            f"{str(row['PM_exists']):<6}| "
            f"{pm_res:<14}| "
            f"{mean_tau:<10}| {norm_tau:<10}| "
            f"{hitl_C:<9}| {agent_C:<9}| "
            f"{str(row['PM_checks_pass']):<7}| "
            f"{row['types']}"
        )

    return summary_rows


def print_summary_table(summary_rows):
    """Print summary table to console."""
    print()
    print(f"TABLE: Proposition 3 Summary (delta={DELTA_FIXED})")
    print("-" * 95)
    print(
        f"{'psi':<7} {'#Eq':<5} {'P^M?':<6} {'PM Res':<12} "
        f"{'tau/tau_b':<10} {'PM OK?':<7} types"
    )
    print("-" * 95)
    for row in summary_rows:
        pm_res = (
            f"{row['PM_residual']:.2e}"
            if row["PM_exists"] and not np.isnan(row["PM_residual"])
            else "N/A"
        )
        norm_tau = (
            f"{row['PM_norm_tau']:.3f}"
            if row["PM_exists"] and not np.isnan(row["PM_norm_tau"])
            else "N/A"
        )
        print(
            f"{row['psi']:<7.1f} "
            f"{row['n_equilibria']:<5} "
            f"{str(row['PM_exists']):<6} "
            f"{pm_res:<12} "
            f"{norm_tau:<10} "
            f"{str(row['PM_checks_pass']):<7} "
            f"{row['types']}"
        )


def main():
    """Run Proposition 3 verification and write results to file."""
    parser = argparse.ArgumentParser(
        description="Verify Proposition 3: Cautious Coordination Equilibrium."
    )
    parser.parse_args()

    print("=" * 70)
    print("PROPOSITION 3: Cautious Coordination Equilibrium Verification")
    print("=" * 70)
    print(f"Parameters: delta={DELTA_FIXED}")
    print(f"Psi sweep: {PSI_VALUES}")
    print(f"DEV_MODE={DEV_MODE}  "
          f"(N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
          f"N_INITS={N_INITS})")
    print(f"Output file: {OUTPUT_TXT}")
    print()

    log_lines = []
    summary_rows = run_psi_sweep(log_lines)

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
