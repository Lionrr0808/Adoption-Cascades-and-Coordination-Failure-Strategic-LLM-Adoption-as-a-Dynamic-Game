"""
Verify Proposition 4: Trust-Induced Cascade Reversal.

From the aggressive equilibrium P^H, exogenously reduce trust (tau). Show that below
a trust threshold tau_underline, firms switch from H (agentic) to L (HITL / no agent),
cascading the system from P^H toward P^M or P^L.

P^H baseline: NPL from aggressive initialization (same as prop4_cascade_reversal()
in verify_propositions.py), with direct high-adoption verification fallback from
verify_equilibrium_existence.py.

Trust analysis:
  (A) Tau-slice reporting on the fixed P^H profile (core prop4 logic).
  (B) Post-shock NPL sweep: perturb states at tau >= tau_cap toward cautious play,
      then re-run NPL to detect cascade reversal.

Key test: conservative firms (C-type) switch from agentic to cautious before G-types.

Usage (from project root):
    python Prop4/verify_proposition4.py
    python Prop4/verify_proposition4.py --delta 2.0 3.0

Output: Prop4/prop4_trust_reversal_verification.txt
"""

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import ModelParams, compute_payoffs, A_NONE, A_HITL, A_AGENT
from npl import run_npl, classify_equilibrium
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
    RESIDUAL_VF_ITER,
    make_sweep_context,
    compute_equilibrium_stats,
    ccp_distance,
    equilibrium_residual,
    verify_high_adoption_eq,
    refine_high_eq_on_reachable,
    build_high_adoption_candidate,
    reachable_mask_high,
    find_npl_equilibria,
    pick_high_adoption_eq,
    run_npl_with_fallbacks,
)

DELTA_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]

# Below this agent rate, a firm type is considered to have "switched to L".
AGENT_SWITCH_THRESH = 0.30
# Post-shock NPL: agent drop from P^H baseline indicating reversal.
REVERSAL_AGENT_MARGIN = 0.20

OUTPUT_TXT = os.path.join(SCRIPT_DIR, "prop4_trust_reversal_verification.txt")


def build_params_for_delta(delta):
    """ModelParams for one delta — same construction as verify_equilibrium_existence.py."""
    return ModelParams(
        **BASE_PARAMS,
        delta=delta,
        psi=PSI_FIXED,
        delta_x_base=DELTA_X_BASE,
        delta_x_comp=DELTA_X_COMP,
    )


def build_aggressive_init(n_states):
    """High-agentic seed used in verify_propositions.py prop4."""
    P = np.zeros((n_states, 3))
    P[:, A_NONE] = 0.05
    P[:, A_HITL] = 0.10
    P[:, A_AGENT] = 0.85
    return P


def format_equilibrium_stats(stats):
    """Format adoption / HITL / agent rates for one equilibrium."""
    return [
        f"       C: adopt={stats['adopt_C']:.3f} "
        f"(HITL={stats['hitl_C']:.3f}, agent={stats['agent_C']:.3f})",
        f"       G: adopt={stats['adopt_G']:.3f} "
        f"(HITL={stats['hitl_G']:.3f}, agent={stats['agent_G']:.3f})",
    ]


def get_ph_profile(ss, params, N_C, N_G, u_C, u_G, log_lines):
    """
    Obtain P^H via aggressive NPL init (prop4), then fallbacks.

    Priority:
      1. NPL from symmetric aggressive init (verify_propositions prop4)
      2. NPL from uniform init
      3. Direct high-adoption verification (Section 2.5 mirror)
      4. Highest-adoption NPL equilibrium from find_npl_equilibria
    """
    P_aggr = build_aggressive_init(ss.n_states)

    result = run_npl(
        ss, params, N_C, N_G,
        P_C_init=P_aggr.copy(), P_G_init=P_aggr.copy(),
        max_iter=NPL_MAX_ITER, tol=NPL_TOL, damping=NPL_DAMPING,
        verbose=False,
    )
    source = "NPL (aggressive init)"

    if not result["converged"]:
        log_lines.append(
            "  WARNING: aggressive-init NPL did not converge; trying uniform init..."
        )
        result = run_npl(
            ss, params, N_C, N_G,
            max_iter=NPL_MAX_ITER, tol=NPL_TOL, damping=NPL_DAMPING,
            verbose=False,
        )
        source = "NPL (uniform init)"

    if result["converged"]:
        P_C, P_G = result["P_C"], result["P_G"]
        stats = compute_equilibrium_stats(P_C, P_G)
        residual = equilibrium_residual(
            ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
            max_vf_iter=RESIDUAL_VF_ITER,
        )
        eq_type = classify_equilibrium(stats)
        if (stats["adopt_C"] + stats["adopt_G"]) / 2 >= 0.5:
            log_lines.append(f"  P^H baseline — {source}:")
            log_lines.extend(format_equilibrium_stats(stats))
            log_lines.append(f"       Type: {eq_type}")
            log_lines.append(f"       Residual: {residual:.2e}")
            return {
                "P_C": P_C, "P_G": P_G,
                "stats": stats,
                "residual": residual,
                "type": eq_type,
                "source": source,
            }, True

    # Direct verification fallback
    high_ok, P_C, P_G, _br, _mask = verify_high_adoption_eq(
        params, ss, N_C, N_G, u_C=u_C, u_G=u_G, verbose=False,
    )
    if high_ok:
        mask = reachable_mask_high(ss, params)
        P_C, P_G = refine_high_eq_on_reachable(
            ss, params, N_C, N_G, u_C, u_G, mask,
            P_C_init=P_C, P_G_init=P_G,
        )
        stats = compute_equilibrium_stats(P_C, P_G)
        residual = equilibrium_residual(
            ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
            max_vf_iter=RESIDUAL_VF_ITER,
        )
        eq_type = classify_equilibrium(stats)
        log_lines.append("  P^H baseline — direct high-adoption verification:")
        log_lines.extend(format_equilibrium_stats(stats))
        log_lines.append(f"       Type: {eq_type}")
        log_lines.append(f"       Residual: {residual:.2e}")
        return {
            "P_C": P_C, "P_G": P_G,
            "stats": stats,
            "residual": residual,
            "type": eq_type,
            "source": "direct verify",
        }, True

    # NPL equilibrium search fallback
    npl_eqs = find_npl_equilibria(
        ss, params, N_C, N_G, u_C, u_G,
        n_inits=20, exclude_low=True, verbose=False,
    )
    eq_H = pick_high_adoption_eq(npl_eqs)
    if eq_H is not None:
        log_lines.append("  P^H baseline — pick_high_adoption_eq (NPL search):")
        log_lines.extend(format_equilibrium_stats(eq_H["stats"]))
        log_lines.append(f"       Type: {eq_H['type']}")
        log_lines.append(f"       Residual: {eq_H['residual']:.2e}")
        return {
            "P_C": eq_H["P_C"], "P_G": eq_H["P_G"],
            "stats": eq_H["stats"],
            "residual": eq_H["residual"],
            "type": eq_H["type"],
            "source": "NPL search",
        }, True

    log_lines.append("  WARNING: P^H baseline NOT FOUND")
    return None, False


def tau_slice_stats(P_C, P_G, ss, tau_level):
    """Mean adoption / HITL / agent rates at states with tau == tau_level."""
    mask = ss.states[:, 3] == tau_level
    if mask.sum() == 0:
        return None
    return {
        "tau": tau_level,
        "n_states": int(mask.sum()),
        "adopt_C": (P_C[mask, A_HITL] + P_C[mask, A_AGENT]).mean(),
        "adopt_G": (P_G[mask, A_HITL] + P_G[mask, A_AGENT]).mean(),
        "hitl_C": P_C[mask, A_HITL].mean(),
        "hitl_G": P_G[mask, A_HITL].mean(),
        "agent_C": P_C[mask, A_AGENT].mean(),
        "agent_G": P_G[mask, A_AGENT].mean(),
    }


def detect_switch_tau(slice_rows, agent_key):
    """
    Highest tau level where agent rate is below AGENT_SWITCH_THRESH.

    C switches first when switch_tau_C > switch_tau_G (C abandons agentic
    deployment at a higher trust level / earlier when trust falls).
    """
    switch_tau = -1
    for row in slice_rows:
        if row[agent_key] < AGENT_SWITCH_THRESH:
            switch_tau = max(switch_tau, row["tau"])
    return switch_tau


def detect_tau_underline(slice_rows):
    """
    Trust threshold tau_underline: highest tau where mean agent (both types)
    is still at or above AGENT_SWITCH_THRESH.

    Below tau_underline (lower trust), firms have switched away from agentic play.
    """
    tau_underline = -1
    for row in slice_rows:
        mean_agent = (row["agent_C"] + row["agent_G"]) / 2
        if mean_agent >= AGENT_SWITCH_THRESH:
            tau_underline = max(tau_underline, row["tau"])
    return tau_underline


def apply_trust_shock(P_C_base, P_G_base, ss, tau_cap):
    """
    Exogenous trust reduction: perturb states with tau >= tau_cap toward cautious play.

    tau_cap = tau_bar: no shock (P^H unchanged).
    Lower tau_cap: more states shocked toward HITL / away from agentic deployment.
    C-types receive a stronger downward bias than G-types (conservative sorting).
    """
    P_C = P_C_base.copy()
    P_G = P_G_base.copy()
    if tau_cap >= ss.states[:, 3].max():
        return P_C, P_G

    for idx in range(ss.n_states):
        tau = ss.states[idx, 3]
        if tau >= tau_cap:
            P_C[idx] = [0.10, 0.75, 0.15]
            P_G[idx] = [0.10, 0.45, 0.45]
    return P_C, P_G


def classify_reversal_outcome(P_C, P_G, ph_profile):
    """Classify post-shock outcome relative to P^H baseline."""
    stats = compute_equilibrium_stats(P_C, P_G)
    eq_type = classify_equilibrium(stats)
    baseline_agent = (
        ph_profile["stats"]["agent_C"] + ph_profile["stats"]["agent_G"]
    ) / 2
    post_agent = (stats["agent_C"] + stats["agent_G"]) / 2
    agent_drop = baseline_agent - post_agent
    d_H = ccp_distance(P_C, P_G, ph_profile["P_C"], ph_profile["P_G"])

    if agent_drop > REVERSAL_AGENT_MARGIN:
        outcome = "reversal"
    elif d_H < DISTINCT_TOL * 3:
        outcome = "P^H"
    elif "cautious" in eq_type.lower():
        outcome = "P^M"
    elif "low" in eq_type.lower():
        outcome = "P^L"
    else:
        outcome = eq_type

    return {
        "stats": stats,
        "type": eq_type,
        "post_agent": post_agent,
        "agent_drop": agent_drop,
        "d_to_PH": d_H,
        "outcome": outcome,
        "reversed": agent_drop > REVERSAL_AGENT_MARGIN,
    }


def run_tau_slice_analysis(ph_profile, ss, tau_bar, log_lines):
    """Part A: tau-slice reporting on fixed P^H (verify_propositions prop4)."""
    P_C, P_G = ph_profile["P_C"], ph_profile["P_G"]
    slice_rows = []

    log_lines.append("")
    log_lines.append(
        "  Trust slice analysis on P^H (holding profile fixed):"
    )
    log_lines.append(
        f"  {'tau':>4}  {'adopt(C)':>9}  {'agent(C)':>9}  {'HITL(C)':>9}  "
        f"{'adopt(G)':>9}  {'agent(G)':>9}  {'HITL(G)':>9}"
    )
    log_lines.append("  " + "-" * 72)

    for tau_level in range(tau_bar + 1):
        row = tau_slice_stats(P_C, P_G, ss, tau_level)
        if row is None:
            continue
        slice_rows.append(row)
        log_lines.append(
            f"  {row['tau']:>4d}  {row['adopt_C']:>9.3f}  "
            f"{row['agent_C']:>9.3f}  {row['hitl_C']:>9.3f}  "
            f"{row['adopt_G']:>9.3f}  {row['agent_G']:>9.3f}  "
            f"{row['hitl_G']:>9.3f}"
        )

    tau_underline = detect_tau_underline(slice_rows)
    switch_C = detect_switch_tau(slice_rows, "agent_C")
    switch_G = detect_switch_tau(slice_rows, "agent_G")
    c_first = switch_C > switch_G

    log_lines.append("")
    if tau_underline >= 0:
        log_lines.append(
            f"  Trust threshold tau_underline = {tau_underline} "
            f"(agent >= {AGENT_SWITCH_THRESH} above; switch below)"
        )
    else:
        log_lines.append(
            f"  Trust threshold tau_underline: NOT OBSERVED "
            f"(agent < {AGENT_SWITCH_THRESH} at all tau levels)"
        )

    log_lines.append(
        f"  C-type switch tau (agent < {AGENT_SWITCH_THRESH}): "
        f"{switch_C if switch_C >= 0 else 'never'}"
    )
    log_lines.append(
        f"  G-type switch tau (agent < {AGENT_SWITCH_THRESH}): "
        f"{switch_G if switch_G >= 0 else 'never'}"
    )
    log_lines.append(
        f"  Key test — C switches first: "
        f"{'PASS' if c_first else 'FAIL' if switch_C >= 0 and switch_G >= 0 else 'N/A'}"
    )

    return {
        "slice_rows": slice_rows,
        "tau_underline": tau_underline,
        "switch_C": switch_C,
        "switch_G": switch_G,
        "c_switches_first": c_first,
    }


def run_trust_shock_sweep(ph_profile, ss, params, N_C, N_G, tau_bar, log_lines):
    """Part B: NPL after exogenous trust reduction at each tau_cap."""
    baseline_agent = (
        ph_profile["stats"]["agent_C"] + ph_profile["stats"]["agent_G"]
    ) / 2

    log_lines.append("")
    log_lines.append(
        f"  Trust shock + NPL sweep: perturb states with tau >= tau_cap; "
        f"baseline agent={baseline_agent:.3f}; "
        f"tau_cap={tau_bar} is no shock (P^H baseline, no NPL)"
    )
    log_lines.append(
        f"  {'tau_cap':>8}  {'post_agent':>11}  {'agent_drop':>11}  "
        f"{'d(P^H)':>10}  {'outcome':>10}  {'reversal?':>10}"
    )
    log_lines.append("  " + "-" * 68)

    shock_rows = []
    for tau_cap in range(tau_bar, -1, -1):
        P_C_shock, P_G_shock = apply_trust_shock(
            ph_profile["P_C"], ph_profile["P_G"], ss, tau_cap,
        )

        if tau_cap >= tau_bar:
            info = classify_reversal_outcome(P_C_shock, P_G_shock, ph_profile)
            row = {"tau_cap": tau_cap, "converged": True, **info}
            log_lines.append(
                f"  {tau_cap:>8d}  {row['post_agent']:>11.3f}  "
                f"{row['agent_drop']:>11.3f}  "
                f"{row['d_to_PH']:>10.4f}  "
                f"{'baseline':>10}  "
                f"{'no':>10}"
            )
            shock_rows.append(row)
            continue

        result = run_npl_with_fallbacks(
            ss, params, N_C, N_G,
            P_C_shock, P_G_shock,
            verbose=False,
        )

        if not result["converged"]:
            row = {
                "tau_cap": tau_cap,
                "converged": False,
                "post_agent": np.nan,
                "agent_drop": np.nan,
                "d_to_PH": np.nan,
                "outcome": "no conv",
                "reversed": False,
            }
            log_lines.append(
                f"  {tau_cap:>8d}  {'(not conv)':>11}  {'':>11}  "
                f"{'':>10}  {'no conv':>10}  {'?':>10}"
            )
        else:
            info = classify_reversal_outcome(
                result["P_C"], result["P_G"], ph_profile,
            )
            row = {"tau_cap": tau_cap, "converged": True, **info}
            log_lines.append(
                f"  {tau_cap:>8d}  {row['post_agent']:>11.3f}  "
                f"{row['agent_drop']:>11.3f}  "
                f"{row['d_to_PH']:>10.4f}  "
                f"{row['outcome']:>10}  "
                f"{'YES' if row['reversed'] else 'no':>10}"
            )
        shock_rows.append(row)

    reversed_caps = [r["tau_cap"] for r in shock_rows if r.get("reversed")]
    if reversed_caps:
        tau_rev = max(reversed_caps)
        log_lines.append("")
        log_lines.append(
            f"  Reversal threshold (NPL shock): tau_cap <= {tau_rev} "
            f"triggers agent drop > {REVERSAL_AGENT_MARGIN}"
        )
        below = [r for r in shock_rows if r["tau_cap"] > tau_rev]
        at_below = [r for r in shock_rows if r["tau_cap"] <= tau_rev]
        no_rev_ok = all(not r.get("reversed") for r in below) if below else True
        rev_ok = all(r.get("reversed") for r in at_below) if at_below else True
        shock_pass = no_rev_ok and rev_ok
        log_lines.append(
            f"  NPL reversal check: "
            f"{'PASS' if shock_pass else 'PARTIAL'} "
            f"(no reversal above threshold: {'OK' if no_rev_ok else 'FAIL'}, "
            f"reversal at/below: {'OK' if rev_ok else 'FAIL'})"
        )
    else:
        tau_rev = np.nan
        shock_pass = False
        log_lines.append("")
        log_lines.append(
            "  Reversal threshold (NPL shock): NOT OBSERVED"
        )

    return {
        "shock_rows": shock_rows,
        "tau_rev": tau_rev,
        "shock_pass": shock_pass if reversed_caps else False,
    }


def run_delta_sweep_for_prop4(delta, ss, N_C, N_G, log_lines):
    """Run Proposition 4 verification for one delta."""
    params = build_params_for_delta(delta)
    u_C, u_G = compute_payoffs(ss, params)
    tau_bar = params.tau_bar

    log_lines.append(f"[ delta = {delta} ]")
    log_lines.append(
        f"  Parameters: psi={PSI_FIXED}, N={params.N}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP} "
        f"(ModelParams matches verify_equilibrium_existence.py)"
    )

    ph_profile, ph_ok = get_ph_profile(
        ss, params, N_C, N_G, u_C, u_G, log_lines,
    )
    if not ph_ok:
        log_lines.append("")
        return {
            "delta": delta,
            "ph_exists": False,
            "tau_underline": np.nan,
            "switch_C": np.nan,
            "switch_G": np.nan,
            "c_switches_first": False,
            "tau_rev": np.nan,
            "prop4_pass": False,
        }

    slice_info = run_tau_slice_analysis(ph_profile, ss, tau_bar, log_lines)
    shock_info = run_trust_shock_sweep(
        ph_profile, ss, params, N_C, N_G, tau_bar, log_lines,
    )

    slice_pass = slice_info["c_switches_first"]
    shock_pass = shock_info["shock_pass"]
    prop4_pass = slice_pass or shock_pass

    log_lines.append("")
    log_lines.append(
        f"  Proposition 4 check: "
        f"{'PASS' if prop4_pass else 'PARTIAL'} "
        f"(C switches first: {'OK' if slice_pass else 'FAIL'}, "
        f"NPL reversal cascade: {'OK' if shock_pass else 'FAIL/N/A'})"
    )
    log_lines.append("")

    print(
        f"[ delta = {delta} ]  done — "
        f"P^H=Yes, tau_underline={slice_info['tau_underline']}, "
        f"C-first={'Yes' if slice_pass else 'No'}",
        flush=True,
    )

    return {
        "delta": delta,
        "ph_exists": True,
        "ph_agent_C": ph_profile["stats"]["agent_C"],
        "ph_agent_G": ph_profile["stats"]["agent_G"],
        "tau_underline": slice_info["tau_underline"],
        "switch_C": slice_info["switch_C"],
        "switch_G": slice_info["switch_G"],
        "c_switches_first": slice_info["c_switches_first"],
        "tau_rev": shock_info["tau_rev"],
        "prop4_pass": prop4_pass,
    }


def run_delta_sweep(log_lines, delta_values=None):
    """Sweep delta and collect Proposition 4 summary rows."""
    if delta_values is None:
        delta_values = DELTA_VALUES

    ss, N_C, N_G, _inits = make_sweep_context()

    log_lines.append("PROPOSITION 4: TRUST-INDUCED CASCADE REVERSAL VERIFICATION")
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
        "P^H baseline: NPL aggressive init (verify_propositions prop4), "
        "with direct-verify / NPL-search fallbacks"
    )
    log_lines.append(
        f"Tau-slice key test: C switches before G (agent < {AGENT_SWITCH_THRESH})"
    )
    log_lines.append(
        f"NPL shock: perturb tau >= tau_cap; reversal if agent drop > "
        f"{REVERSAL_AGENT_MARGIN} (max_iter={NPL_MAX_ITER}, tol={NPL_TOL}, "
        f"damping={NPL_DAMPING})"
    )
    log_lines.append("")

    summary_rows = []
    for delta in delta_values:
        print(f"[ delta = {delta} ]  running trust reversal sweep...", flush=True)
        row = run_delta_sweep_for_prop4(delta, ss, N_C, N_G, log_lines)
        summary_rows.append(row)

    log_lines.append("=" * 105)
    log_lines.append("TABLE: Proposition 4 Verification Summary")
    log_lines.append("-" * 105)
    header = (
        f"{'delta':<7}| {'P^H?':<6}| {'tau_ul':<7}| "
        f"{'sw_C':<5}| {'sw_G':<5}| {'C 1st?':<7}| "
        f"{'tau_rev':<8}| {'Prop4 OK?':<10}| "
        f"baseline agent (C/G)"
    )
    log_lines.append(header)
    log_lines.append("-" * 105)

    for row in summary_rows:
        tau_ul = (
            f"{int(row['tau_underline'])}"
            if row.get("tau_underline") is not None
            and row["tau_underline"] >= 0
            else "N/A"
        )
        sw_C = (
            f"{int(row['switch_C'])}"
            if row.get("switch_C") is not None and row["switch_C"] >= 0
            else "N/A"
        )
        sw_G = (
            f"{int(row['switch_G'])}"
            if row.get("switch_G") is not None and row["switch_G"] >= 0
            else "N/A"
        )
        tau_rev = (
            f"{int(row['tau_rev'])}"
            if row.get("tau_rev") is not None and not np.isnan(row["tau_rev"])
            else "N/A"
        )
        agent_baseline = (
            f"{row['ph_agent_C']:.3f}/{row['ph_agent_G']:.3f}"
            if row.get("ph_exists")
            else "N/A"
        )
        log_lines.append(
            f"{row['delta']:<7.1f}| "
            f"{str(row['ph_exists']):<6}| "
            f"{tau_ul:<7}| "
            f"{sw_C:<5}| {sw_G:<5}| "
            f"{str(row.get('c_switches_first', False)):<7}| "
            f"{tau_rev:<8}| "
            f"{str(row.get('prop4_pass', False)):<10}| "
            f"{agent_baseline}"
        )

    return summary_rows


def print_summary_table(summary_rows):
    """Print summary table to console."""
    print()
    print("TABLE: Proposition 4 Summary")
    print("-" * 65)
    print(
        f"{'delta':<7} {'P^H?':<6} {'tau_ul':<7} "
        f"{'C 1st?':<7} {'Prop4 OK?':<10}"
    )
    print("-" * 65)
    for row in summary_rows:
        tau_ul = (
            f"{int(row['tau_underline'])}"
            if row.get("tau_underline") is not None
            and row["tau_underline"] >= 0
            else "N/A"
        )
        print(
            f"{row['delta']:<7.1f} "
            f"{str(row['ph_exists']):<6} "
            f"{tau_ul:<7} "
            f"{str(row.get('c_switches_first', False)):<7} "
            f"{str(row.get('prop4_pass', False)):<10}"
        )


def main():
    """Run Proposition 4 verification and write results to file."""
    parser = argparse.ArgumentParser(
        description="Verify Proposition 4: Trust-Induced Cascade Reversal."
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
    print("PROPOSITION 4: Trust-Induced Cascade Reversal Verification")
    print("=" * 70)
    print(
        f"Parameters: psi={PSI_FIXED}, delta_x_base={DELTA_X_BASE}, "
        f"delta_x_comp={DELTA_X_COMP}"
    )
    print(f"Delta sweep: {delta_values}")
    print(f"DEV_MODE={DEV_MODE}  "
          f"(N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
          f"tau_bar={BASE_PARAMS['tau_bar']})")
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
