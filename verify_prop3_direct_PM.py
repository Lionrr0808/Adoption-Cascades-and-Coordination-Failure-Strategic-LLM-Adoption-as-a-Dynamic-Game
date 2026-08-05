"""
Proposition 3 — Direct verification of cautious equilibrium P^M
(with tau* ≈ tau_bar), WITHOUT multi-init NPL discovery.

Thought experiment (user request):
  Competitors (all other firms) do not adopt → state slice A_L = A_H = 0.
  The firm under study may still play a HITL-heavy CCP (P^M candidate).

Method (Section 2.5 / Prop1 spirit):
  1. Own-type seed: HITL-heavy CCP candidate P^M_cand with psi-dependent
     agent mass — P(H)=p_agent(psi) decreases in psi so damage intensity
     is pressed harder at large psi (tau* stays nearer tau_bar).
     HITL absorbs the remainder; NONE stays small/fixed.
  2. Competitor CCP locked at pure NONE for all mean-field / transitions
     (so E[A_L], E[A_H] stay at 0 — "others don't adopt").
  3. Refine own CCP with undamped BR iterations on the A_L=A_H=0 mask,
     always building transitions from the NONE competitor profile.
     Residual ‖P_own − Ψ(P_comp=NONE)‖ on that mask only.
  4. Classify / tau* / P(H) / CCP rates from the **refined** own CCP
     (primary). Seed schedule logged as diagnostic only (no Seed|A0 residual).
     BR vs NONE on A=0 can collapse mask rows toward NONE.

Consistency note:
  Own HITL-heavy ≠ competitor NONE. This is intentionally asymmetric —
  not a symmetric NPL fixed point of the full game. Full-game HITL by all
  firms would push A away from 0. Here we only ask whether HITL is a BR
  for the studied firm when others are locked at no adoption.

State space: same DEV_MODE grid as Prop1/Prop3
  (N=4, x_bar=5, T_bar=3, tau_bar=3, n_states=960).
Delta fixed at 3.0; sweep psi (default Prop3 grid).

Run from this folder or from simulation/:
  python prop3_direct_PM/verify_prop3_direct_PM.py
  python prop3_direct_PM/verify_prop3_direct_PM.py --smoke
  python prop3_direct_PM/verify_prop3_direct_PM.py --psi 5 8
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
PROP3_DIR = os.path.join(ROOT_DIR, "Prop3")
if PROP3_DIR not in sys.path:
    sys.path.insert(0, PROP3_DIR)

from model import (  # noqa: E402
    ModelParams,
    StateSpace,
    compute_payoffs,
    A_NONE,
    A_HITL,
    A_AGENT,
)
from npl import npl_iteration, classify_equilibrium  # noqa: E402
from verify_equilibrium_existence import (  # noqa: E402
    DEV_MODE,
    BASE_PARAMS,
    DELTA_X_BASE,
    DELTA_X_COMP,
    RESIDUAL_VF_ITER,
    compute_equilibrium_stats,
    equilibrium_residual,
    equilibrium_residual_on_mask,
    format_equilibrium_stats,
)
from verify_proposition3 import (  # noqa: E402
    DELTA_FIXED,
    PSI_VALUES as PROP3_PSI_VALUES,
    compute_trust_capital_stats,
    format_trust_capital,
    verify_pm_criteria,
    build_params as prop3_build_params,
)

# ---------------------------------------------------------------------------
# Direct-verification knobs (NOT NPL multi-init search)
# ---------------------------------------------------------------------------
# Own-type seed (firm under study): HITL-heavy P^M candidate with
# psi-dependent agent mass. NONE fixed; HITL absorbs the remainder.
#
#   p_agent(psi) = max(P_OWN_AGENT_MIN, P_OWN_AGENT_0 / (1 + c * psi))
#
# At psi=0: agent ≈ 0.04 (HITL ≈ 0.88). At psi=8 with c=0.875:
# agent ≈ 0.005 so eq.23 damage psi*P(H) grows slower and tau* stays
# nearer tau_bar. Agent adoption therefore falls more strongly in psi.
P_OWN_NONE = 0.08
P_OWN_AGENT_0 = 0.04          # agent at psi = 0
P_OWN_AGENT_PSI_C = 0.875     # c in p0 / (1 + c*psi); psi=8 → ~0.005
P_OWN_AGENT_MIN = 1e-4        # floor (keep tiny but positive)

# Competitor CCP locked at pure NONE ("others don't adopt").
# Tiny HITL/AGENT mass avoids log(0) in choice probs if ever used as own CCP.
P_COMP_NONE = 1.0 - 1e-10
P_COMP_HITL = 5e-11
P_COMP_AGENT = 5e-11

# BR refinement on A=0 mask (undamped); transitions always from competitor
# NONE profile. Primary CCP / tau* / type come from the refined profile.
REFINE_MAX_ITER = 80
REFINE_TOL = 1e-8

# Residual "approximate equilibrium" threshold (report always; PASS/FAIL flag)
RESIDUAL_PASS_TOL = 1e-4

# tau* ≈ tau_bar: absolute gap and ratio tolerances
TAU_GAP_TOL = 1e-2          # |tau* - tau_bar| (on [0, tau_bar]-clipped value)
TAU_RATIO_TOL = 1e-2        # |tau*/tau_bar - 1|

OUTPUT_TXT = os.path.join(
    SCRIPT_DIR, "prop3_direct_PM_verification.txt"
)


def build_params(psi: float) -> ModelParams:
    """Same ModelParams construction as Prop3 (delta fixed, DEV_MODE grid)."""
    return prop3_build_params(psi)


def own_seed_agent(psi: float) -> float:
    """
    Psi-dependent own-firm AGENT (P(H)) seed mass.

    p_agent(psi) = max(p_min, p0 / (1 + c * psi))

    Decreasing in psi: presses P(H) at large psi so agent adoption falls
    more strongly and tau* stays nearer tau_bar.
    """
    p = float(P_OWN_AGENT_0) / (1.0 + float(P_OWN_AGENT_PSI_C) * float(psi))
    return max(float(P_OWN_AGENT_MIN), p)


def own_seed_rates(psi: float):
    """
    Own HITL-heavy seed mixture at this psi.

    NONE fixed; AGENT = own_seed_agent(psi); HITL = 1 - NONE - AGENT.
    """
    p_none = float(P_OWN_NONE)
    p_agent = own_seed_agent(psi)
    p_hitl = 1.0 - p_none - p_agent
    if p_hitl < 0.0:
        raise ValueError(
            f"own seed rates invalid at psi={psi}: "
            f"NONE={p_none}, AGENT={p_agent} leave HITL={p_hitl}"
        )
    return p_none, p_hitl, p_agent


def build_own_PM_candidate(n_states: int, psi: float):
    """
    HITL-heavy CCP for the firm under study (candidate P^M).

    Agent mass decreases with psi (see own_seed_agent). Both types use the
    same cautious mixture. Competitors are represented separately via
    build_competitor_none_ccp (not this seed).
    """
    p_none, p_hitl, p_agent = own_seed_rates(psi)
    assert abs(p_none + p_hitl + p_agent - 1.0) < 1e-12
    P = np.zeros((n_states, 3))
    P[:, A_NONE] = p_none
    P[:, A_HITL] = p_hitl
    P[:, A_AGENT] = p_agent
    return P.copy(), P.copy()


def build_competitor_none_ccp(n_states: int):
    """
    Pure-NONE CCP for all other firms (mean-field / transitions).

    Keeps competitor adoption at zero so A_L, A_H stay at 0 in expectation.
    """
    assert abs(P_COMP_NONE + P_COMP_HITL + P_COMP_AGENT - 1.0) < 1e-12
    P = np.zeros((n_states, 3))
    P[:, A_NONE] = P_COMP_NONE
    P[:, A_HITL] = P_COMP_HITL
    P[:, A_AGENT] = P_COMP_AGENT
    return P.copy(), P.copy()


def reachable_mask_others_none(ss: StateSpace, params: ModelParams):
    """
    States with no competitor adoption: A_L = 0 and A_H = 0.

    When delta_x_base == 0, also restrict to x = 0 (same convention as
    Prop1 low-adoption reachable mask — competitive gap does not drift).
    """
    low_A = (ss.states[:, 1] == 0) & (ss.states[:, 2] == 0)
    if params.delta_x_base == 0:
        return low_A & (ss.states[:, 0] == 0)
    return low_A


def refine_own_on_mask_competitors_none(
    ss,
    params,
    N_C,
    N_G,
    u_C,
    u_G,
    mask,
    P_own_C_init,
    P_own_G_init,
    P_comp_C,
    P_comp_G,
    max_iter=REFINE_MAX_ITER,
    tol=REFINE_TOL,
):
    """
    Undamped BR iteration for the studied firm on the A_L=A_H=0 mask.

    Transitions / mean-field always use (P_comp_C, P_comp_G) = pure NONE so
    competitors stay at no adoption. Own CCP is updated only on `mask`;
    off-mask own CCP stays pinned to the HITL-heavy seed.

    This is NOT find_npl_equilibria / multi-init discovery.
    """
    P_own_C = P_own_C_init.copy()
    P_own_G = P_own_G_init.copy()
    pin_C, pin_G = P_own_C_init.copy(), P_own_G_init.copy()
    last_d = np.inf
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        off = ~mask
        P_own_C[off] = pin_C[off]
        P_own_G[off] = pin_G[off]
        # BR under locked competitor NONE profile (not under own HITL CCP)
        P_C_br, P_G_br = npl_iteration(
            ss, params, P_comp_C, P_comp_G, N_C, N_G, u_C, u_G,
            max_vf_iter=RESIDUAL_VF_ITER,
        )
        last_d = equilibrium_residual_on_mask(
            P_own_C, P_own_G, P_C_br, P_G_br, mask,
        )
        P_own_C[mask] = P_C_br[mask]
        P_own_G[mask] = P_G_br[mask]
        if last_d < tol:
            break

    return P_own_C, P_own_G, int(n_iter), float(last_d)


def check_tau_star_equals_tau_bar(trust: dict, tau_bar: float):
    """
    Success criterion: tau* ≈ tau_bar (eq.23), i.e. damage intensity ≈ 0.

    Uses clipped industry tau* from compute_trust_capital_stats.
    """
    tau_star = float(trust["tau_star_theory"])
    ratio = float(trust["norm_tau_theory"])
    gap = abs(tau_star - float(tau_bar))
    ratio_gap = abs(ratio - 1.0)
    ok = (gap <= TAU_GAP_TOL) or (ratio_gap <= TAU_RATIO_TOL)
    return {
        "ok": bool(ok),
        "tau_star": tau_star,
        "tau_bar": float(tau_bar),
        "gap": float(gap),
        "ratio": ratio,
        "ratio_gap": float(ratio_gap),
        "intensity": float(trust["PH_pT"]),
        "P_H": float(trust["P_H"]),
    }


def verify_direct_PM_at_psi(ss, params, N_C, N_G, u_C, u_G, verbose=True):
    """
    Direct-verify one candidate P^M at fixed (delta, psi).

    Own seed = HITL-heavy with psi-dependent agent; refine on A=0 under
    competitor CCP locked at NONE. Primary CCP / tau* / P(H) / type from
    the refined own profile. Seed|A0 residual is not computed.
    """
    psi = float(params.psi)
    p_none, p_hitl, p_agent = own_seed_rates(psi)
    mask = reachable_mask_others_none(ss, params)
    P_seed_C, P_seed_G = build_own_PM_candidate(ss.n_states, psi)
    P_comp_C, P_comp_G = build_competitor_none_ccp(ss.n_states)

    P_C, P_G, n_refine, d_refine = refine_own_on_mask_competitors_none(
        ss, params, N_C, N_G, u_C, u_G, mask,
        P_seed_C, P_seed_G, P_comp_C, P_comp_G,
    )

    # Residual of refined own CCP vs BR under competitor NONE (primary)
    P_C_br, P_G_br = npl_iteration(
        ss, params, P_comp_C, P_comp_G, N_C, N_G, u_C, u_G,
        max_vf_iter=RESIDUAL_VF_ITER,
    )
    res_reach = equilibrium_residual_on_mask(P_C, P_G, P_C_br, P_G_br, mask)

    # Global residual under the same locked-competitor BR operator
    # (not symmetric NPL residual of the HITL profile against itself)
    res_global = max(
        np.max(np.abs(P_C - P_C_br)),
        np.max(np.abs(P_G - P_G_br)),
    )
    # Also report classical symmetric residual of own profile (diagnostic)
    res_sym_global = equilibrium_residual(
        ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
        max_vf_iter=RESIDUAL_VF_ITER,
    )

    stats = compute_equilibrium_stats(P_C, P_G)
    stats_seed = compute_equilibrium_stats(P_seed_C, P_seed_G)
    stats_mask = {
        "adopt_C": float(
            (P_C[mask, A_HITL] + P_C[mask, A_AGENT]).mean()
        ) if mask.any() else float("nan"),
        "adopt_G": float(
            (P_G[mask, A_HITL] + P_G[mask, A_AGENT]).mean()
        ) if mask.any() else float("nan"),
        "hitl_C": float(P_C[mask, A_HITL].mean()) if mask.any() else float("nan"),
        "hitl_G": float(P_G[mask, A_HITL].mean()) if mask.any() else float("nan"),
        "agent_C": float(P_C[mask, A_AGENT].mean()) if mask.any() else float("nan"),
        "agent_G": float(P_G[mask, A_AGENT].mean()) if mask.any() else float("nan"),
    }

    # Primary reporting CCP for tau*/P(H)/type: refined own profile.
    eq_type = classify_equilibrium(stats)
    trust = compute_trust_capital_stats(
        ss, params, P_C, P_G, N_C, N_G,
    )
    pm_checks = verify_pm_criteria(stats, trust)
    tau_check = check_tau_star_equals_tau_bar(trust, params.tau_bar)

    # Primary residual for PASS: refined own vs BR(NONE) on A=0
    residual_ok = (
        not np.isnan(res_reach) and res_reach <= RESIDUAL_PASS_TOL
    )
    is_cautious_label = "cautious" in eq_type.lower()
    is_pm = bool(pm_checks["all_pass"] and is_cautious_label)
    success = bool(is_pm and tau_check["ok"] and residual_ok)

    if verbose:
        print(
            f"    seed P(H)=agent={p_agent:.4f} "
            f"(psi-schedule; NONE={p_none:.2f}, HITL={p_hitl:.4f})"
        )
        print(
            f"    refine_iters={n_refine}, d_refine={d_refine:.2e}, "
            f"n_reach={int(mask.sum())} (A_L=A_H=0)"
        )
        print(
            f"    refined type={eq_type}; "
            f"HITL={((stats['hitl_C']+stats['hitl_G'])/2):.3f}, "
            f"agent={((stats['agent_C']+stats['agent_G'])/2):.3f}"
        )
        print(
            f"    residual refined|A=0={res_reach:.2e}, "
            f"vs BR(NONE) global={res_global:.2e}; "
            f"sym-NPL global={res_sym_global:.2e}; "
            f"tau*/tau_bar={tau_check['ratio']:.4f}, "
            f"|tau*-tb|={tau_check['gap']:.4f}"
        )
        print(
            f"    PM criteria={'PASS' if pm_checks['all_pass'] else 'FAIL'}; "
            f"tau*=tb={'PASS' if tau_check['ok'] else 'FAIL'}; "
            f"residual={'PASS' if residual_ok else 'FAIL'}; "
            f"OVERALL={'PASS' if success else 'FAIL'}"
        )

    return {
        "P_C": P_C,
        "P_G": P_G,
        "P_seed_C": P_seed_C,
        "P_seed_G": P_seed_G,
        "P_comp_C": P_comp_C,
        "P_comp_G": P_comp_G,
        "seed_rates": (p_none, p_hitl, p_agent),
        "reach_mask": mask,
        "n_reachable": int(mask.sum()),
        "n_refine": n_refine,
        "d_refine": d_refine,
        "stats": stats,
        "stats_seed": stats_seed,
        "stats_mask": stats_mask,
        "type": eq_type,
        "trust": trust,
        "pm_checks": pm_checks,
        "tau_check": tau_check,
        "residual_reachable": float(res_reach),
        "residual_global": float(res_global),
        "residual_sym_global": float(res_sym_global),
        "residual_ok": residual_ok,
        "is_pm": is_pm,
        "success": success,
    }


def run_psi_sweep(psi_values, log_lines):
    """Sweep psi with fixed delta; direct P^M candidate at each psi."""
    params0 = build_params(psi_values[0])
    ss = StateSpace(params0)
    N_C = params0.N // 2
    N_G = params0.N - N_C

    log_lines.append(
        "PROPOSITION 3: DIRECT VERIFICATION OF P^M (tau* ≈ tau_bar)"
    )
    log_lines.append("=" * 105)
    log_lines.append(
        f"Fixed parameters: delta={DELTA_FIXED}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}"
    )
    log_lines.append(
        f"State space (DEV_MODE={DEV_MODE}): "
        f"N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"T_bar={BASE_PARAMS['T_bar']}, tau_bar={BASE_PARAMS['tau_bar']}, "
        f"n_states={ss.n_states}"
    )
    log_lines.append(
        "Method: DIRECT verification (NOT multi-init NPL discovery). "
        "Own-type HITL-heavy CCP with psi-dependent agent "
        f"p_agent=max({P_OWN_AGENT_MIN:g}, {P_OWN_AGENT_0}/(1+{P_OWN_AGENT_PSI_C}*psi)), "
        f"NONE={P_OWN_NONE} fixed, HITL=1-NONE-agent; "
        "competitor CCP locked at pure NONE "
        f"(NONE={P_COMP_NONE:.1e}, HITL={P_COMP_HITL:.1e}, "
        f"AGENT={P_COMP_AGENT:.1e}) for mean-field / transitions; "
        f"refine own CCP with <= {REFINE_MAX_ITER} undamped BR iters on mask "
        "(A_L=0, A_H=0"
        + (", x=0" if DELTA_X_BASE == 0 else "")
        + "); "
        "BR operator = npl_iteration(P_comp=NONE); "
        "primary CCP / type / P(H) / tau* via Prop3 compute_trust_capital_stats "
        "on refined own CCP (eq.23); residual Ref|A0 only (no Seed|A0)."
    )
    log_lines.append(
        "Consistency: own HITL-heavy ≠ competitor NONE (asymmetric thought "
        "experiment). Not a symmetric full-game NPL fixed point — HITL by "
        "all firms would move A away from 0. BR refine vs NONE may collapse "
        "mask rows toward no-adoption (P^L-like)."
    )
    log_lines.append(
        "Success (P^M + tau*=tau_bar): classify cautious + verify_pm_criteria "
        f"on refined own CCP (high HITL, low agent) AND |tau*-tau_bar|<={TAU_GAP_TOL} "
        f"(or |tau*/tau_bar-1|<={TAU_RATIO_TOL}) AND residual of refined own "
        f"vs BR(NONE) on A=0 mask (Ref|A0) <= {RESIDUAL_PASS_TOL}."
    )
    log_lines.append("")

    summary_rows = []
    for psi in psi_values:
        print(f"[ psi = {psi} ]  direct P^M candidate (others NONE)...", flush=True)
        log_lines.append(f"[ psi = {psi} ]")

        params = build_params(psi)
        u_C, u_G = compute_payoffs(ss, params)
        res = verify_direct_PM_at_psi(
            ss, params, N_C, N_G, u_C, u_G, verbose=True,
        )

        p_none, p_hitl, p_agent = res["seed_rates"]
        log_lines.append(
            f"  Candidate P^M (own HITL seed → BR refine on A_L=A_H=0; "
            f"competitors locked NONE): {res['type']}"
        )
        log_lines.append(
            f"  Own seed schedule at psi={psi}: "
            f"P(H)=agent={p_agent:.4f}, HITL={p_hitl:.4f}, NONE={p_none:.2f} "
            f"[p_agent=max({P_OWN_AGENT_MIN:g}, "
            f"{P_OWN_AGENT_0}/(1+{P_OWN_AGENT_PSI_C}*psi))]"
        )
        log_lines.append(
            "  Refined own CCP rates (primary; studied firm / P^M criteria):"
        )
        log_lines.extend(format_equilibrium_stats(res["stats"]))
        log_lines.append(
            "  Own seed CCP rates (diagnostic; pre-refine schedule):"
        )
        log_lines.extend(format_equilibrium_stats(res["stats_seed"]))
        log_lines.append(
            f"       Own seed: NONE={p_none:.2f}, HITL={p_hitl:.4f}, "
            f"AGENT={p_agent:.4f} (both types; psi-dependent AGENT)"
        )
        log_lines.append(
            f"       Competitor CCP (locked): NONE≈1, HITL≈0, AGENT≈0 "
            f"(both types; mean-field only)"
        )
        log_lines.append(
            f"       Refine: iters={res['n_refine']}, "
            f"last_reach_d={res['d_refine']:.2e}, "
            f"n_reachable={res['n_reachable']} (A_L=A_H=0)"
        )
        log_lines.append(
            f"       Residual refined vs BR(NONE) on A=0 (primary Ref|A0): "
            f"{res['residual_reachable']:.2e}"
        )
        log_lines.append(
            f"       Residual vs BR(NONE) global:        "
            f"{res['residual_global']:.2e}"
        )
        log_lines.append(
            f"       Symmetric NPL residual (own vs own, diagnostic): "
            f"{res['residual_sym_global']:.2e}"
        )
        log_lines.append(
            "       Note: primary CCP / tau* / type / Ref|A0 residual are from "
            "the refined own profile vs BR under competitors locked at NONE, "
            "restricted to A_L=A_H=0. Seed schedule is diagnostic only "
            "(no Seed|A0 residual). Refine vs NONE may drift toward "
            "no-adoption (P^L-like)."
        )
        log_lines.extend(
            format_trust_capital(res["trust"], params.tau_bar, res["type"])
        )

        checks = res["pm_checks"]
        tau_c = res["tau_check"]
        log_lines.append("")
        log_lines.append(
            "  P^M key tests (direct; Prop1-style CCP rates on refined own):"
        )
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
            f"       Low agent CCP (agent < 0.2):      "
            f"{'PASS' if checks['low_agent'] else 'FAIL'}"
        )
        log_lines.append(
            f"       Classify as cautious:             "
            f"{'PASS' if res['is_pm'] else 'FAIL'} ({res['type']})"
        )
        log_lines.append(
            f"       tau* ≈ tau_bar:                   "
            f"{'PASS' if tau_c['ok'] else 'FAIL'} "
            f"(tau*={tau_c['tau_star']:.4f}, tau_bar={tau_c['tau_bar']:.4f}, "
            f"gap={tau_c['gap']:.4e}, ratio={tau_c['ratio']:.6f}, "
            f"P(H)={tau_c['P_H']:.4f}, intensity={tau_c['intensity']:.4e})"
        )
        log_lines.append(
            f"       Residual tol ({RESIDUAL_PASS_TOL:.0e}):           "
            f"{'PASS' if res['residual_ok'] else 'FAIL'} "
            f"(Ref|A0={res['residual_reachable']:.2e}, "
            f"vs BR(NONE) global={res['residual_global']:.2e})"
        )
        log_lines.append(
            f"       Overall direct P^M (tau*=tb):     "
            f"{'PASS' if res['success'] else 'FAIL'}"
        )
        log_lines.append("")

        summary_rows.append({
            "psi": psi,
            "type": res["type"],
            "is_pm": res["is_pm"],
            "pm_checks_pass": checks["all_pass"],
            "tau_ok": tau_c["ok"],
            "residual_ok": res["residual_ok"],
            "success": res["success"],
            "res_reach": res["residual_reachable"],
            "res_global": res["residual_global"],
            "tau_star": tau_c["tau_star"],
            "tau_ratio": tau_c["ratio"],
            "tau_gap": tau_c["gap"],
            "P_H": tau_c["P_H"],
            "P_L": float(res["trust"]["P_L"]),
            "seed_agent": p_agent,
            "hitl_C": float(res["stats"]["hitl_C"]),
            "hitl_G": float(res["stats"]["hitl_G"]),
            "agent_C": float(res["stats"]["agent_C"]),
            "agent_G": float(res["stats"]["agent_G"]),
            "n_refine": res["n_refine"],
        })

    log_lines.append("=" * 130)
    log_lines.append(
        "SUMMARY (direct P^M; others NONE; A_L=A_H=0; CCP/tau*/type = refined)"
    )
    log_lines.append(
        f"{'psi':<7}| {'type':<18}| {'PM?':<5}| {'tau*=tb':<8}| "
        f"{'ResOK':<6}| {'OK?':<5}| {'Ref|A0':<12}| "
        f"{'tau*':<10}| {'tau*/tb':<9}| {'P(H)':<8}| "
        f"{'HITL_C':<8}| {'agent_C':<8}| {'HITL_G':<8}| {'agent_G':<8}"
    )
    log_lines.append("-" * 130)
    for row in summary_rows:
        log_lines.append(
            f"{row['psi']:<7.1f}| {row['type']:<18}| "
            f"{str(row['is_pm']):<5}| {str(row['tau_ok']):<8}| "
            f"{str(row['residual_ok']):<6}| {str(row['success']):<5}| "
            f"{row['res_reach']:<12.2e}| "
            f"{row['tau_star']:<10.4f}| {row['tau_ratio']:<9.4f}| "
            f"{row['P_H']:<8.4f}| "
            f"{row['hitl_C']:<8.3f}| {row['agent_C']:<8.3f}| "
            f"{row['hitl_G']:<8.3f}| {row['agent_G']:<8.3f}"
        )
    log_lines.append("")
    n_pass = sum(1 for r in summary_rows if r["success"])
    log_lines.append(
        f"Direct P^M + tau*=tau_bar PASS count: {n_pass}/{len(summary_rows)}"
    )
    return summary_rows


def print_summary_table(summary_rows):
    print()
    print("=" * 90)
    print("SUMMARY — direct P^M (tau* ≈ tau_bar; others NONE; refined CCP)")
    print("=" * 90)
    hdr = (
        f"{'psi':<7} {'OK?':<6} {'Ref|A0':<12} "
        f"{'tau*':<10} {'tau*/tb':<9} {'P(H)':<8} {'seed_a':<8} {'type'}"
    )
    print(hdr)
    print("-" * 90)
    for row in summary_rows:
        print(
            f"{row['psi']:<7.1f} {str(row['success']):<6} "
            f"{row['res_reach']:<12.2e} "
            f"{row['tau_star']:<10.4f} {row['tau_ratio']:<9.4f} "
            f"{row['P_H']:<8.4f} {row['seed_agent']:<8.4f} {row['type']}"
        )


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Direct verification of Prop3 P^M with tau*≈tau_bar "
            "(own HITL seed; competitors locked NONE; no multi-init NPL)."
        )
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run: single psi=5.0 only (import + one direct verify).",
    )
    p.add_argument(
        "--psi",
        type=float,
        nargs="+",
        default=None,
        help="Psi values to verify (default: Prop3 grid). Ignored if --smoke.",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write output txt (stdout only).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        psi_values = [5.0]
    elif args.psi is not None:
        psi_values = list(args.psi)
    else:
        psi_values = list(PROP3_PSI_VALUES)

    print("=" * 70)
    print("PROPOSITION 3: Direct P^M verification (tau* ≈ tau_bar)")
    print("  Own HITL seed; competitors locked at NONE (A_L=A_H=0)")
    print("=" * 70)
    print(
        f"Parameters: delta={DELTA_FIXED}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}"
    )
    print(f"Psi values: {psi_values}")
    print(
        f"DEV_MODE={DEV_MODE}  "
        f"(N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"T_bar={BASE_PARAMS['T_bar']}, tau_bar={BASE_PARAMS['tau_bar']})"
    )
    print(
        "Method: own HITL CCP (psi-dependent agent) → BR refine under "
        "competitor NONE on A=0 mask (NO multi-init NPL) → eq.23 tau* "
        "on refined CCP"
    )
    print(
        f"Own seed schedule: p_agent=max({P_OWN_AGENT_MIN:g}, "
        f"{P_OWN_AGENT_0}/(1+{P_OWN_AGENT_PSI_C}*psi)), "
        f"NONE={P_OWN_NONE} fixed, HITL=1-NONE-agent"
    )
    print(
        "  e.g. psi=0 → agent="
        f"{own_seed_agent(0.0):.4f}; psi=8 → agent="
        f"{own_seed_agent(8.0):.4f}"
    )
    print(
        f"Competitor (locked): NONE≈1, HITL≈0, AGENT≈0"
    )
    if not args.no_write:
        print(f"Output file: {OUTPUT_TXT}")
    print()

    log_lines = []
    summary_rows = run_psi_sweep(psi_values, log_lines)

    if not args.no_write:
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
            f.write("\n")
        print()
        print("=" * 70)
        print(f"Results written to:\n  {OUTPUT_TXT}")
        print("=" * 70)

    print_summary_table(summary_rows)
    return summary_rows


if __name__ == "__main__":
    main()
