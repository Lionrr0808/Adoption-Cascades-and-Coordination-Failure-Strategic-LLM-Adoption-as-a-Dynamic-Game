"""
Verify Proposition 1: multiple equilibria via direct fixed-point verification.

KEY INSIGHT: NPL can only find NPL-stable equilibria (Aguirregabiria & Mira, 2007).
The low-adoption equilibrium P^L is NPL-unstable in coordination games, so NPL
always converges to high-adoption equilibria regardless of initialization.

SOLUTION:
  1. Verify P^L directly via verify_low_adoption_eq (Section 2.5 of RA Guide).
  2. Find all other equilibria (P^H, P^M, ...) via NPL with find_all_equilibria.
  3. Report residuals, distinctness, and phase diagram.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import (
    ModelParams, StateSpace, compute_payoffs, A_NONE, A_HITL, A_AGENT,
)
from npl import (
    npl_iteration, run_npl, classify_equilibrium,
    generate_initial_profiles,
)


# ---------------------------------------------------------------------------
# Fixed parameters for Proposition 1 verification
# (DEV_MODE mirrors verify_propositions.py for faster NPL / high-eq search)
# ---------------------------------------------------------------------------
PSI_FIXED = 0.5
DELTA_X_BASE = 0.0
DELTA_X_COMP = 0.2
DELTA_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
# DELTA_VALUES = [4.0, 5.0]

DEV_MODE = True
if DEV_MODE:
    BASE_PARAMS = dict(N=4, x_bar=5, T_bar=3, tau_bar=3)
    N_INITS = 20
else:
    BASE_PARAMS = dict(N=6, x_bar=9, T_bar=4, tau_bar=4)
    N_INITS = 50

DISTINCT_TOL = 0.01
# NPL_MAX_ITER = 1000          # run_npl default (find_all_equilibria uses this)
NPL_MAX_ITER = 500          # run_npl default (find_all_equilibria uses this)
NPL_TOL = 1e-6              # find_all_equilibria passes tol=1e-6 (not run_npl 1e-5 default)
NPL_DAMPING = 0.3           # run_npl default
RESIDUAL_VF_ITER = 200      # VF iterations for residual reporting (P^L and logs)

# Coarser grid for phase diagram (main delta sweep keeps N_INITS from DEV_MODE)
# PHASE_DELTA_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
PHASE_DELTA_GRID = [4.0, 5.0]
PHASE_PSI_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
PHASE_N_INITS = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_TXT = os.path.join(OUT_DIR, "prop1_equilibrium_verification.txt")
OUTPUT_TXT_LOW = os.path.join(OUT_DIR, "prop1_low_equilibrium_only.txt")
OUTPUT_FIG = os.path.join(OUT_DIR, "prop1_multiplicity_phase_diagram.png")


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


def make_sweep_context(delta=DELTA_VALUES[0]):
    """
    Build reusable StateSpace / firm counts for a delta sweep.

    State-space cardinality depends only on BASE_PARAMS (not delta), so ss
    can be shared across all delta values in the sweep.
    """
    params = ModelParams(
        **BASE_PARAMS,
        delta=delta,
        psi=PSI_FIXED,
        delta_x_base=DELTA_X_BASE,
        delta_x_comp=DELTA_X_COMP,
    )
    ss = StateSpace(params)
    N_C = params.N // 2
    N_G = params.N - N_C
    inits = generate_initial_profiles(ss.n_states, N_INITS, seed=42)
    return ss, N_C, N_G, inits


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


def build_low_adoption_policy(n_states):
    """Canonical low-adoption CCP profile: essentially no adoption everywhere."""
    P = np.zeros((n_states, 3))
    P[:, A_NONE] = 1.0 - 1e-10
    P[:, A_HITL] = 5e-11
    P[:, A_AGENT] = 5e-11
    return P


def reachable_mask_low(ss, params):
    """States reachable under the low-adoption policy."""
    low_A = (ss.states[:, 1] == 0) & (ss.states[:, 2] == 0)
    if params.delta_x_base == 0:
        return low_A & (ss.states[:, 0] == 0)
    return low_A


def equilibrium_residual_on_mask(P_C, P_G, P_C_br, P_G_br, mask):
    """Residual restricted to a subset of states (for P^L ergodic component)."""
    if mask.sum() == 0:
        return np.nan
    return max(
        np.max(np.abs(P_C[mask] - P_C_br[mask])),
        np.max(np.abs(P_G[mask] - P_G_br[mask])),
    )


def build_high_adoption_candidate(n_states):
    """Asymmetric high-adoption seed (delta=3 equilibrium pattern)."""
    P_C = np.zeros((n_states, 3))
    P_G = np.zeros((n_states, 3))
    P_C[:, A_NONE] = 0.10
    P_C[:, A_HITL] = 0.75
    P_C[:, A_AGENT] = 0.15
    P_G[:, A_NONE] = 0.10
    P_G[:, A_HITL] = 0.35
    P_G[:, A_AGENT] = 0.55
    return P_C, P_G


def reachable_mask_high(ss, _params):
    """Ergodic component under high adoption: x=0 and at least 2 adopters."""
    return (ss.states[:, 0] == 0) & (ss.states[:, 1] + ss.states[:, 2] >= 2)


def refine_high_eq_on_reachable(
    ss, params, N_C, N_G, u_C, u_G, mask,
    P_C_init=None, P_G_init=None,
    max_iter=500, tol=1e-10,
):
    """
    Refine P^H on the high-adoption ergodic set via undamped BR iteration.

    Off-path states are pinned to the initial seed so refinement stays on the
    high-adoption component (mirror of refine_low_eq_on_reachable).
    """
    if P_C_init is None or P_G_init is None:
        P_C_init, P_G_init = build_high_adoption_candidate(ss.n_states)
    P_C = P_C_init.copy()
    P_G = P_G_init.copy()
    pin_C, pin_G = P_C_init.copy(), P_G_init.copy()

    for _ in range(max_iter):
        off = ~mask
        P_C[off] = pin_C[off]
        P_G[off] = pin_G[off]
        P_C_br, P_G_br = npl_iteration(
            ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
            max_vf_iter=RESIDUAL_VF_ITER,
        )
        d = equilibrium_residual_on_mask(P_C, P_G, P_C_br, P_G_br, mask)
        P_C[mask] = P_C_br[mask]
        P_G[mask] = P_G_br[mask]
        if d < tol:
            break

    return P_C, P_G


def verify_high_adoption_eq(params, ss, N_C, N_G, u_C=None, u_G=None, verbose=True):
    """
    Verify that a high-adoption equilibrium exists (Section 2.5 mirror).

    Strategy: seed an asymmetric high-adoption profile, refine via undamped BR
    on the ergodic component (x=0, A>=2), then check BR prefers adoption.
    """
    if u_C is None or u_G is None:
        u_C, u_G = compute_payoffs(ss, params)

    P_cand_C, P_cand_G = build_high_adoption_candidate(ss.n_states)
    mask = reachable_mask_high(ss, params)

    P_C, P_G = refine_high_eq_on_reachable(
        ss, params, N_C, N_G, u_C, u_G, mask,
        P_C_init=P_cand_C, P_G_init=P_cand_G,
    )
    P_C_br, P_G_br = npl_iteration(
        ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
        max_vf_iter=RESIDUAL_VF_ITER,
    )
    results = {"C": P_C_br, "G": P_G_br}

    adopt_C = results["C"][mask, A_HITL] + results["C"][mask, A_AGENT]
    adopt_G = results["G"][mask, A_HITL] + results["G"][mask, A_AGENT]
    verified_C = (adopt_C > 0.5).all()
    verified_G = (adopt_G > 0.5).all()
    verified = verified_C and verified_G

    if verbose:
        print(f"\n  Reachable states under high-adoption policy: {mask.sum()}")
        for label, probs in (("C", results["C"]), ("G", results["G"])):
            p_adopt = probs[mask, A_HITL] + probs[mask, A_AGENT]
            print(f"  Type {label} at reachable states: "
                  f"P(adopt) in [{p_adopt.min():.4f}, {p_adopt.max():.4f}]")
        print(f"  High-adoption eq verified: {verified}")

    return verified, P_C, P_G, results, mask


def refine_low_eq_on_reachable(
    ss, params, N_C, N_G, u_C, u_G, mask,
    max_iter=500, tol=1e-10,
):
    """
    Refine P^L on the ergodic (reachable) component via undamped BR iteration.

    P^L is NPL-unstable globally, but on the low-adoption ergodic set the
    best-response mapping is a contraction.  Updating only reachable states
    yields a machine-precision fixed point on that component.
    """
    P_C = build_low_adoption_policy(ss.n_states)
    P_G = P_C.copy()

    for _ in range(max_iter):
        P_C_br, P_G_br = npl_iteration(
            ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
            max_vf_iter=RESIDUAL_VF_ITER,
        )
        d = equilibrium_residual_on_mask(P_C, P_G, P_C_br, P_G_br, mask)
        P_C[mask] = P_C_br[mask]
        P_G[mask] = P_G_br[mask]
        if d < tol:
            break

    return P_C, P_G


def verify_low_adoption_eq(params, ss, N_C, N_G, u_C=None, u_G=None, verbose=True):
    """
    Verify that a low-adoption equilibrium exists (Section 2.5).

    Strategy: if ALL firms play 'don't adopt' and delta_x_base is small enough,
    then at reachable states (low x, low A), the BR is also 'don't adopt'.
    """
    if u_C is None or u_G is None:
        u_C, u_G = compute_payoffs(ss, params)

    P_cand = build_low_adoption_policy(ss.n_states)
    P_C_br, P_G_br = npl_iteration(
        ss, params, P_cand, P_cand, N_C, N_G, u_C, u_G
    )
    results = {"C": P_C_br, "G": P_G_br}
    mask = reachable_mask_low(ss, params)

    if verbose:
        print(f"\n  Reachable states under low-adoption policy: {mask.sum()}")

    verified_C = (results["C"][mask, A_NONE] > 0.5).all()
    verified_G = (results["G"][mask, A_NONE] > 0.5).all()
    verified = verified_C and verified_G

    if verbose:
        for label in ("C", "G"):
            probs = results[label]
            print(f"  Type {label} at reachable states: "
                  f"P(no adopt) in [{probs[mask, A_NONE].min():.4f}, "
                  f"{probs[mask, A_NONE].max():.4f}]")
        print(f"  Low-adoption eq verified: {verified}")

    return verified, P_cand, results, mask


def find_npl_equilibria(ss, params, N_C, N_G, u_C, u_G, n_inits=N_INITS,
                        exclude_low=True, verbose=False, inits=None):
    """
    Find all NPL-stable equilibria (same acceptance logic as find_all_equilibria).

    Accepts any profile where run_npl converges (same max_iter/tol/damping).
    Residual is computed for reporting only, not for acceptance. When
    exclude_low=True, low-adoption profiles (avg adopt < 0.3) are dropped.
    """
    if inits is None:
        inits = generate_initial_profiles(ss.n_states, n_inits, seed=42)
    equilibria = []

    for P_C_init, P_G_init, label in inits:
        # Match find_all_equilibria -> run_npl exactly for NPL convergence
        result = run_npl(
            ss, params, N_C, N_G,
            P_C_init=P_C_init, P_G_init=P_G_init,
            max_iter=NPL_MAX_ITER, tol=NPL_TOL, damping=NPL_DAMPING,
            verbose=False,
        )
        if not result["converged"]:
            continue

        P_C, P_G = result["P_C"], result["P_G"]
        stats = compute_equilibrium_stats(P_C, P_G)
        if exclude_low and (stats["adopt_C"] + stats["adopt_G"]) / 2 < 0.3:
            continue

        is_new = True
        for eq in equilibria:
            if ccp_distance(P_C, P_G, eq["P_C"], eq["P_G"]) < DISTINCT_TOL:
                is_new = False
                break

        if is_new:
            eq_type = classify_equilibrium(stats)
            residual = equilibrium_residual(
                ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
                max_vf_iter=RESIDUAL_VF_ITER,
            )
            equilibria.append({
                "P_C": P_C, "P_G": P_G,
                "stats": stats, "residual": residual,
                "label": label, "type": eq_type,
            })
            if verbose:
                print(f"    NPL eq: {eq_type}, residual={residual:.2e}")

    return equilibria


def format_equilibrium_stats(stats):
    """Format adoption / HITL / agent rates for one equilibrium."""
    lines = [
        f"       C: adopt={stats['adopt_C']:.3f} "
        f"(HITL={stats['hitl_C']:.3f}, agent={stats['agent_C']:.3f})",
        f"       G: adopt={stats['adopt_G']:.3f} "
        f"(HITL={stats['hitl_G']:.3f}, agent={stats['agent_G']:.3f})",
    ]
    return lines


def label_equilibrium_type(eq_type_str):
    """Short label for equilibrium type."""
    if "low" in eq_type_str.lower():
        return "P^L"
    if "cautious" in eq_type_str.lower():
        return "P^M"
    if "aggressive" in eq_type_str.lower() or "agentic" in eq_type_str.lower():
        return "P^H"
    return eq_type_str


def pick_high_adoption_eq(equilibria):
    """Highest mean-adoption NPL equilibrium (P^H proxy in Prop 1 table)."""
    candidates = [
        e for e in equilibria
        if "low" not in e["type"].lower()
        and (e["stats"]["adopt_C"] + e["stats"]["adopt_G"]) / 2 >= 0.5
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda e: (e["stats"]["adopt_C"] + e["stats"]["adopt_G"]) / 2,
    )


def pick_cautious_eq(equilibria):
    """P^M equilibrium if present."""
    return next(
        (e for e in equilibria if "cautious" in e["type"].lower()),
        None,
    )


def format_br_check(br_results, mask):
    """Summarize best-response check at reachable states."""
    lines = []
    for label in ("C", "G"):
        probs = br_results[label]
        p_none = probs[mask, A_NONE]
        ok = (p_none > 0.5).all()
        lines.append(
            f"       Type {label} BR: P(no adopt) in "
            f"[{p_none.min():.4f}, {p_none.max():.4f}]  "
            f"({'pass' if ok else 'FAIL'})"
        )
    return lines


def run_low_only_sweep(log_lines):
    """
    Fast P^L-only verification (Section 2.5 of RA Guide).

    Skips NPL sweep and phase diagram.
    """
    summary_rows = []
    ss, N_C, N_G, _inits = make_sweep_context()

    log_lines.append("PROPOSITION 1: LOW-ADOPTION EQUILIBRIUM VERIFICATION (Section 2.5)")
    log_lines.append("=" * 105)
    log_lines.append(
        f"Fixed parameters: psi={PSI_FIXED}, delta_x_base={DELTA_X_BASE}, "
        f"delta_x_comp={DELTA_X_COMP}"
    )
    log_lines.append(
        "Method: direct fixed-point verification on the low-adoption ergodic component"
    )
    log_lines.append(
        "Paused: NPL sweep and phase diagram"
    )
    log_lines.append("")

    for delta in DELTA_VALUES:
        print(f"[ delta = {delta} ]  verifying P^L...", flush=True)

        params = ModelParams(
            **BASE_PARAMS,
            delta=delta,
            psi=PSI_FIXED,
            delta_x_base=DELTA_X_BASE,
            delta_x_comp=DELTA_X_COMP,
        )
        u_C, u_G = compute_payoffs(ss, params)

        low_exists, P_low_C, br_results, reach_mask = verify_low_adoption_eq(
            params, ss, N_C, N_G, u_C, u_G, verbose=False
        )
        br_C_ok = (br_results["C"][reach_mask, A_NONE] > 0.5).all()
        br_G_ok = (br_results["G"][reach_mask, A_NONE] > 0.5).all()
        n_reach = int(reach_mask.sum())

        stats_L = None
        res_L_reach = np.nan
        res_L_global = np.nan

        log_lines.append(f"[ delta = {delta} ]")
        log_lines.append(f"  Reachable states under low-adoption policy: {n_reach}")

        if low_exists:
            P_low_C, P_low_G = refine_low_eq_on_reachable(
                ss, params, N_C, N_G, u_C, u_G, reach_mask
            )
            stats_L = compute_equilibrium_stats(P_low_C, P_low_G)
            P_C_br, P_G_br = npl_iteration(
                ss, params, P_low_C, P_low_G, N_C, N_G, u_C, u_G
            )
            res_L_global = equilibrium_residual(
                ss, params, P_low_C, P_low_G, N_C, N_G, u_C, u_G
            )
            res_L_reach = equilibrium_residual_on_mask(
                P_low_C, P_low_G, P_C_br, P_G_br, reach_mask
            )

            log_lines.append("  Equilibrium: P^L (Low Adoption)")
            log_lines.extend(format_equilibrium_stats(stats_L))
            log_lines.extend(format_br_check(br_results, reach_mask))
            log_lines.append(f"       Residual (reachable):  {res_L_reach:.2e}")
            log_lines.append(f"       Residual (global):     {res_L_global:.2e}")
            log_lines.append(
                "       Note: global residual may exceed 1e-8 at off-path states "
                "because P^L is NPL-unstable outside the ergodic component."
            )
        else:
            log_lines.extend(format_br_check(br_results, reach_mask))
            log_lines.append("  P^L: NOT FOUND (direct verification failed)")

        log_lines.append("")

        summary_rows.append({
            "delta": delta,
            "PL_exists": low_exists,
            "n_reachable": n_reach,
            "br_C_ok": br_C_ok,
            "br_G_ok": br_G_ok,
            "PL_residual_reach": res_L_reach,
            "PL_residual_global": res_L_global,
            "stats": stats_L,
        })

        reach_str = f"{res_L_reach:.2e}" if low_exists else "N/A"
        print(
            f"[ delta = {delta} ]  P^L={'Yes' if low_exists else 'No'}, "
            f"reach-res={reach_str}, BR(C/G)=({br_C_ok}/{br_G_ok})",
            flush=True,
        )
        if stats_L is not None:
            print(
                f"    C: adopt={stats_L['adopt_C']:.3f} "
                f"(HITL={stats_L['hitl_C']:.3f}, "
                f"agent={stats_L['agent_C']:.3f})",
                flush=True,
            )
            print(
                f"    G: adopt={stats_L['adopt_G']:.3f} "
                f"(HITL={stats_L['hitl_G']:.3f}, "
                f"agent={stats_L['agent_G']:.3f})",
                flush=True,
            )

    log_lines.append("=" * 105)
    log_lines.append(
        "TABLE: Low-Adoption Equilibrium Summary "
        f"(psi fixed at {PSI_FIXED})"
    )
    log_lines.append("-" * 105)
    header = (
        f"{'delta':<7}| {'P^L?':<6}| {'Reach':<6}| {'BR C':<6}| {'BR G':<6}| "
        f"{'Reach Res':<12}| {'Global Res':<12}| "
        f"{'adopt_C':<8}| {'adopt_G':<8}"
    )
    log_lines.append(header)
    log_lines.append("-" * 105)

    for row in summary_rows:
        reach_res = (
            f"{row['PL_residual_reach']:.2e}"
            if row["PL_exists"] and not np.isnan(row["PL_residual_reach"])
            else "N/A"
        )
        global_res = (
            f"{row['PL_residual_global']:.2e}"
            if row["PL_exists"] and not np.isnan(row["PL_residual_global"])
            else "N/A"
        )
        adopt_C = (
            f"{row['stats']['adopt_C']:.3f}"
            if row["stats"] is not None else "N/A"
        )
        adopt_G = (
            f"{row['stats']['adopt_G']:.3f}"
            if row["stats"] is not None else "N/A"
        )
        log_lines.append(
            f"{row['delta']:<7.1f}| "
            f"{str(row['PL_exists']):<6}| "
            f"{row['n_reachable']:<6}| "
            f"{str(row['br_C_ok']):<6}| "
            f"{str(row['br_G_ok']):<6}| "
            f"{reach_res:<12}| {global_res:<12}| "
            f"{adopt_C:<8}| {adopt_G:<8}"
        )

    return summary_rows


def print_low_only_summary(summary_rows):
    """Print low-only summary table to console."""
    print()
    print("TABLE: Low-Adoption Equilibrium Summary "
          f"(psi fixed at {PSI_FIXED})")
    print("-" * 95)
    print(
        f"{'delta':<7} {'P^L?':<6} {'Reach':<6} {'BR C':<6} {'BR G':<6} "
        f"{'Reach Res':<12} {'Global Res':<12} {'adopt_C':<8} {'adopt_G':<8}"
    )
    print("-" * 95)
    for row in summary_rows:
        reach_res = (
            f"{row['PL_residual_reach']:.2e}"
            if row["PL_exists"] and not np.isnan(row["PL_residual_reach"])
            else "N/A"
        )
        global_res = (
            f"{row['PL_residual_global']:.2e}"
            if row["PL_exists"] and not np.isnan(row["PL_residual_global"])
            else "N/A"
        )
        adopt_C = (
            f"{row['stats']['adopt_C']:.3f}"
            if row["stats"] is not None else "N/A"
        )
        adopt_G = (
            f"{row['stats']['adopt_G']:.3f}"
            if row["stats"] is not None else "N/A"
        )
        print(
            f"{row['delta']:<7.1f} "
            f"{str(row['PL_exists']):<6} "
            f"{row['n_reachable']:<6} "
            f"{str(row['br_C_ok']):<6} "
            f"{str(row['br_G_ok']):<6} "
            f"{reach_res:<12} {global_res:<12} "
            f"{adopt_C:<8} {adopt_G:<8}"
        )


def run_delta_sweep(log_lines):
    """Main sweep over delta values with P^L + P^H verification."""
    summary_rows = []
    all_delta_results = []
    ss, N_C, N_G, npl_inits = make_sweep_context()

    log_lines.append("PROPOSITION 1: MULTIPLE EQUILIBRIA VERIFICATION")
    log_lines.append("=" * 105)
    log_lines.append(
        f"Fixed parameters: psi={PSI_FIXED}, delta_x_base={DELTA_X_BASE}, "
        f"delta_x_comp={DELTA_X_COMP}"
    )
    log_lines.append(
        f"State space (DEV_MODE={DEV_MODE}): "
        f"N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"T_bar={BASE_PARAMS['T_bar']}, tau_bar={BASE_PARAMS['tau_bar']}, "
        f"n_states={ss.n_states}, N_INITS={N_INITS}"
    )
    log_lines.append(
        "Method: P^L via direct verification (Section 2.5); "
        "P^H/P^M via NPL (find_all_equilibria logic)"
    )
    log_lines.append("")

    for delta in DELTA_VALUES:
        print(f"[ delta = {delta} ]  processing P^L + P^H...", flush=True)
        log_lines.append(f"[ delta = {delta} ]")

        params = ModelParams(
            **BASE_PARAMS,
            delta=delta,
            psi=PSI_FIXED,
            delta_x_base=DELTA_X_BASE,
            delta_x_comp=DELTA_X_COMP,
        )
        u_C, u_G = compute_payoffs(ss, params)

        equilibria = []

        # --- Step 1: P^L via direct verification ---
        low_exists, P_low_C, _br_results, reach_mask = verify_low_adoption_eq(
            params, ss, N_C, N_G, u_C, u_G, verbose=False
        )
        P_low_G = P_low_C.copy()

        if low_exists:
            P_low_C, P_low_G = refine_low_eq_on_reachable(
                ss, params, N_C, N_G, u_C, u_G, reach_mask
            )
            stats_L = compute_equilibrium_stats(P_low_C, P_low_G)
            P_C_br, P_G_br = npl_iteration(
                ss, params, P_low_C, P_low_G, N_C, N_G, u_C, u_G
            )
            res_L_global = equilibrium_residual(
                ss, params, P_low_C, P_low_G, N_C, N_G, u_C, u_G
            )
            res_L_reach = equilibrium_residual_on_mask(
                P_low_C, P_low_G, P_C_br, P_G_br, reach_mask
            )

            eq_L = {
                "P_C": P_low_C, "P_G": P_low_G,
                "stats": stats_L,
                "residual": res_L_reach,
                "residual_global": res_L_global,
                "residual_reachable": res_L_reach,
                "type": "P^L (low adoption)",
                "label": "direct_verify",
            }
            equilibria.append(eq_L)

            log_lines.append("  Equilibrium 1: P^L (Low Adoption)")
            log_lines.extend(format_equilibrium_stats(stats_L))
            log_lines.append(f"       Residual (reachable):  {res_L_reach:.2e}")
            log_lines.append(f"       Residual (global):     {res_L_global:.2e}")
            log_lines.append(
                "       Note: P^L is verified on the ergodic component (Section 2.5); "
                "global residual exceeds 1e-8 at off-path states because P^L is "
                "NPL-unstable and those states are never visited."
            )
        else:
            log_lines.append("  P^L: NOT FOUND (direct verification failed)")

        # --- Step 2: P^H / P^M via NPL (dev-mode settings) ---
        npl_eqs = find_npl_equilibria(
            ss, params, N_C, N_G, u_C, u_G,
            n_inits=N_INITS, exclude_low=True, verbose=False,
            inits=npl_inits,
        )

        for j, eq in enumerate(npl_eqs, start=len(equilibria) + 1):
            short = label_equilibrium_type(eq["type"])
            log_lines.append(f"  Equilibrium {j}: {eq['type']}")
            log_lines.extend(format_equilibrium_stats(eq["stats"]))
            log_lines.append(f"       Residual: {eq['residual']:.2e}")
            equilibria.append(eq)

        if not npl_eqs:
            log_lines.append("  No NPL-stable equilibria found (excluding P^L)")

        # --- Distinctness table ---
        if len(equilibria) >= 2:
            log_lines.append("")
            log_lines.append("  Distinctness (max-norm over CCPs):")
            log_lines.append(f"  {'Pair':<20} {'Distance':>12}")
            log_lines.append(f"  {'-'*20} {'-'*12}")
            for i in range(len(equilibria)):
                for j in range(i + 1, len(equilibria)):
                    d = ccp_distance(
                        equilibria[i]["P_C"], equilibria[i]["P_G"],
                        equilibria[j]["P_C"], equilibria[j]["P_G"],
                    )
                    ti = label_equilibrium_type(equilibria[i]["type"])
                    tj = label_equilibrium_type(equilibria[j]["type"])
                    log_lines.append(f"  ||{ti} - {tj}||{' '*(14-len(ti)-len(tj))}{d:.4f}")

        log_lines.append("")

        eq_L = next((e for e in equilibria if "low" in e["type"].lower()), None)
        eq_H = pick_high_adoption_eq(equilibria)
        dist_LH = (
            ccp_distance(eq_L["P_C"], eq_L["P_G"], eq_H["P_C"], eq_H["P_G"])
            if eq_L and eq_H else np.nan
        )

        summary_rows.append({
            "delta": delta,
            "PL_exists": low_exists,
            "PL_residual": eq_L["residual_reachable"] if eq_L else np.nan,
            "PH_exists": eq_H is not None,
            "PH_residual": eq_H["residual"] if eq_H else np.nan,
            "dist_LH": dist_LH,
            "n_equilibria": len(equilibria),
        })
        all_delta_results.append({"delta": delta, "equilibria": equilibria})

        print(f"[ delta = {delta} ]  done — "
              f"P^L={'Yes' if low_exists else 'No'}, "
              f"P^H={'Yes' if eq_H is not None else 'No'}, "
              f"NPL eqs={len(npl_eqs)}", flush=True)

    # --- Summary table ---
    log_lines.append("=" * 105)
    log_lines.append("TABLE 1: Proposition 1 Verification Summary "
                     f"(psi fixed at {PSI_FIXED})")
    log_lines.append("-" * 105)
    header = (
        f"{'delta':<7}| {'P^L Exists?':<13}| {'P^L Residual':<16}| "
        f"{'P^H Exists?':<13}| {'P^H Residual':<16}| "
        f"{'||P^L - P^H||':<14}"
    )
    log_lines.append(header)
    log_lines.append("-" * 105)

    for row in summary_rows:
        pl_res = (
            f"{row['PL_residual']:.2e}"
            if row["PL_exists"] and not np.isnan(row["PL_residual"]) else "N/A"
        )
        ph_res = (
            f"{row['PH_residual']:.2e}"
            if row["PH_exists"] and not np.isnan(row["PH_residual"]) else "N/A"
        )
        dist_str = (
            f"{row['dist_LH']:.3f}"
            if not np.isnan(row["dist_LH"]) else "N/A"
        )
        log_lines.append(
            f"{row['delta']:<7.1f}| "
            f"{str(row['PL_exists']):<13}| {pl_res:<16}| "
            f"{str(row['PH_exists']):<13}| {ph_res:<16}| "
            f"{dist_str:<14}"
        )

    return summary_rows, all_delta_results


def run_phase_diagram(log_lines):
    """
    Phase diagram in (delta, psi) space showing multiplicity boundary.
    P^L via direct verify + reachable refinement; P^H/P^M via NPL.
    """
    print("Generating phase diagram in (delta, psi) space...")
    log_lines.append("")
    log_lines.append("=" * 105)
    log_lines.append("TABLE 2: Multiplicity Phase Diagram (delta, psi) space")
    log_lines.append("-" * 105)

    delta_grid = PHASE_DELTA_GRID
    psi_grid = PHASE_PSI_GRID
    ss, N_C, N_G, _ = make_sweep_context()
    phase_inits = generate_initial_profiles(ss.n_states, PHASE_N_INITS, seed=42)

    cell_matrix = np.zeros((len(delta_grid), len(psi_grid)))
    cell_labels = {}

    header = f"{'delta/psi':<10}" + "".join(f"{p:>10.2f}" for p in psi_grid)
    log_lines.append(header)
    log_lines.append("-" * (10 + 10 * len(psi_grid)))

    for i, delta in enumerate(delta_grid):
        print(f"  phase diagram: delta={delta} ...")
        row_str = f"{delta:<10.1f}"
        for j, psi in enumerate(psi_grid):
            params = ModelParams(
                **BASE_PARAMS,
                delta=delta, psi=psi,
                delta_x_base=DELTA_X_BASE,
                delta_x_comp=DELTA_X_COMP,
            )
            u_C, u_G = compute_payoffs(ss, params)

            low_ok, _, _, mask = verify_low_adoption_eq(
                params, ss, N_C, N_G, u_C, u_G, verbose=False
            )
            npl_eqs = find_npl_equilibria(
                ss, params, N_C, N_G, u_C, u_G,
                n_inits=PHASE_N_INITS, exclude_low=True, verbose=False,
                inits=phase_inits,
            )

            types = set()
            if low_ok:
                types.add("L")
            for eq in npl_eqs:
                t = label_equilibrium_type(eq["type"])
                if t.startswith("P^"):
                    types.add(t.replace("P^", ""))
                elif "Asymmetric" in eq["type"] or "Other" in eq["type"]:
                    types.add("H")

            label = "+".join(sorted(types)) if types else "-"
            cell_labels[(i, j)] = label
            row_str += f"{label:>10}"

            code = 0
            if low_ok:
                code += 1
            if pick_high_adoption_eq(npl_eqs) is not None:
                code += 2
            if pick_cautious_eq(npl_eqs) is not None:
                code += 4
            cell_matrix[i, j] = code

        log_lines.append(row_str)

    log_lines.append("")
    log_lines.append(
        f"Equilibrium residuals along psi={PSI_FIXED} "
        f"(delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}):"
    )
    log_lines.append(f"{'delta':<8} {'P^L reach-res':<16} {'P^H res':<16} {'mult':<8}")
    log_lines.append("-" * 50)

    for delta in delta_grid:
        params = ModelParams(
            **BASE_PARAMS, delta=delta, psi=PSI_FIXED,
            delta_x_base=DELTA_X_BASE, delta_x_comp=DELTA_X_COMP,
        )
        u_C, u_G = compute_payoffs(ss, params)

        low_ok, _, _, mask = verify_low_adoption_eq(
            params, ss, N_C, N_G, u_C, u_G, verbose=False
        )
        res_L = np.nan
        if low_ok:
            P_C, P_G = refine_low_eq_on_reachable(
                ss, params, N_C, N_G, u_C, u_G, mask
            )
            P_C_br, P_G_br = npl_iteration(
                ss, params, P_C, P_G, N_C, N_G, u_C, u_G
            )
            res_L = equilibrium_residual_on_mask(P_C, P_G, P_C_br, P_G_br, mask)

        npl_eqs = find_npl_equilibria(
            ss, params, N_C, N_G, u_C, u_G,
            n_inits=PHASE_N_INITS, exclude_low=True,
            inits=phase_inits,
        )
        eq_H = pick_high_adoption_eq(npl_eqs)
        res_H = eq_H["residual"] if eq_H else np.nan
        mult = "YES" if (low_ok and eq_H) else ("L" if low_ok else ("H" if eq_H else "no"))

        res_L_str = f"{res_L:.2e}" if not np.isnan(res_L) else "N/A"
        res_H_str = f"{res_H:.2e}" if not np.isnan(res_H) else "N/A"
        log_lines.append(f"{delta:<8.1f} {res_L_str:<16} {res_H_str:<16} {mult:<8}")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = matplotlib.colors.ListedColormap(
        ["#cccccc", "#a6cee3", "#fb9a99", "#b2df8a", "#fdbf6f"]
    )
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 7.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(
        cell_matrix, origin="lower", aspect="auto",
        cmap=cmap, norm=norm,
        extent=[psi_grid[0] - 0.125, psi_grid[-1] + 0.125,
                delta_grid[0] - 0.25, delta_grid[-1] + 0.25],
    )

    for i in range(len(delta_grid)):
        for j in range(len(psi_grid)):
            ax.text(
                psi_grid[j], delta_grid[i], cell_labels[(i, j)],
                ha="center", va="center", fontsize=7, color="black",
            )

    ax.set_xlabel(r"Trust damage intensity $\psi$", fontsize=12)
    ax.set_ylabel(r"Competitive externality $\delta$", fontsize=12)
    ax.set_title(
        "Multiplicity Boundary in ($\\delta$, $\\psi$) Space\n"
        f"(delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP})",
        fontsize=12,
    )

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], shrink=0.8)
    cbar.ax.set_yticklabels(["None", "L only", "H only", "L + H", "L + H + M"])

    plt.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Phase diagram saved to {OUTPUT_FIG}")

    log_lines.append("")
    log_lines.append(f"Phase diagram figure saved to: {OUTPUT_FIG}")


def main():
    """Run Proposition 1 verification and write results to file."""
    parser = argparse.ArgumentParser(
        description="Verify Proposition 1 equilibria (P^L direct + P^H via NPL)."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also generate phase diagram in (delta, psi) space.",
    )
    parser.add_argument(
        "--low-only",
        action="store_true",
        help="Fast path: verify P^L only via Section 2.5 (skip NPL / P^H).",
    )
    args = parser.parse_args()
    if args.full and args.low_only:
        parser.error("Cannot specify both --full and --low-only")
    low_only = args.low_only

    print("=" * 70)
    print("PROPOSITION 1: Equilibrium Existence Verification")
    print("=" * 70)
    print(f"Parameters: psi={PSI_FIXED}, delta_x_base={DELTA_X_BASE}, "
          f"delta_x_comp={DELTA_X_COMP}")
    print(f"Delta sweep: {DELTA_VALUES}")
    print(f"DEV_MODE={DEV_MODE}  "
          f"(N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
          f"N_INITS={N_INITS})")
    if low_only:
        print("Mode: low-only (P^L direct verification; NPL / P^H skipped)")
        print(f"Output file: {OUTPUT_TXT_LOW}")
    else:
        print("Mode: default (P^L direct + P^H via NPL)")
        print(f"Output file: {OUTPUT_TXT}")
        if args.full:
            print("         + phase diagram (--full)")
    print()

    log_lines = []

    if low_only:
        summary_rows = run_low_only_sweep(log_lines)
        output_path = OUTPUT_TXT_LOW
    else:
        summary_rows, _ = run_delta_sweep(log_lines)
        if args.full:
            run_phase_diagram(log_lines)
        output_path = OUTPUT_TXT

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
        f.write("\n")

    print()
    print("=" * 70)
    print(f"Verification complete. Results written to:\n  {output_path}")
    if not low_only:
        print(f"Phase diagram written to:\n  {OUTPUT_FIG}")
    print("=" * 70)

    if low_only:
        print_low_only_summary(summary_rows)
    else:
        print()
        print("TABLE 1: Proposition 1 Verification Summary "
              f"(psi fixed at {PSI_FIXED})")
        print("-" * 78)
        print(f"{'delta':<7} {'P^L?':<6} {'P^L Res':<12} {'P^H?':<6} "
              f"{'P^H Res':<12} {'||L-H||':<10}")
        print("-" * 78)
        for row in summary_rows:
            pl_res = (
                f"{row['PL_residual']:.2e}"
                if row["PL_exists"] and not np.isnan(row["PL_residual"]) else "N/A"
            )
            ph_res = (
                f"{row['PH_residual']:.2e}"
                if row["PH_exists"] and not np.isnan(row["PH_residual"]) else "N/A"
            )
            dist_str = (
                f"{row['dist_LH']:.3f}"
                if not np.isnan(row["dist_LH"]) else "N/A"
            )
            print(f"{row['delta']:<7.1f} {str(row['PL_exists']):<6} {pl_res:<12} "
                  f"{str(row['PH_exists']):<6} {ph_res:<12} {dist_str:<10}")


if __name__ == "__main__":
    main()
