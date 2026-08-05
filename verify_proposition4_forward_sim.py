"""
Proposition 4 — sequential forward simulation along an exogenous trust path.

Complement to Prop4/verify_proposition4.py (tau-slice / NPL shock). Trust follows

    tau_t = max(0, tau_bar - t),   t = 0, 1, ..., T_max

so with tau_bar=3: t=0→3, t=1→2, t=2→1, t=3→0.

At EVERY t (including t=0), every state's τ coordinate is forced to τ_t:
  - Flow payoffs: g_τ = τ_t / τ̄ for all states (compute_payoffs_forced_tau)
  - Transitions: mean-field CCP looked up at (x,A_L,A_H,τ_t,T); next τ pinned
    to τ_t. Only (x, A_L, A_H, T) vary. No model.py change.

Procedure (per delta):
  t=0  — Multi-init NPL P^H under forced τ=τ̄ for all states (local forced-τ
         NPL, not heterogeneous state-τ payoffs). Record overall CCP means.
         Store competitor adoption environment = P^H_0.
  t>0  — Warm-start own CCP from CCP_{t-1}; force τ=τ_t on payoffs +
         transitions; competitor CCPs frozen at P^H_0. Dampened NPL to
         convergence. Rates = overall means of updated own CCP.

Honesty note: sequential NPL along an exogenous tau path with frozen
competitors — not a full closed-loop rational-expectations path.

Usage (from simulation/):
    python prop4_forward_sim/verify_proposition4_forward_sim.py
    python prop4_forward_sim/verify_proposition4_forward_sim.py --delta 2.0 3.0
    python prop4_forward_sim/verify_proposition4_forward_sim.py --n-br 100
    python prop4_forward_sim/verify_proposition4_forward_sim.py --plot-only

Output (defaults under this folder):
    prop4_forward_sim/prop4_forward_sim_verification.txt
    prop4_forward_sim/figures/prop4_forward_rates_vs_t.{png,pdf}
    prop4_forward_sim/figures/prop4_forward_rates_vs_t_delta{d}.{png,pdf}
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # simulation/
PROP4_DIR = os.path.join(ROOT_DIR, "Prop4")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import (
    A_AGENT,
    A_HITL,
    A_NONE,
)
from npl import (
    choice_probs_from_values,
    classify_equilibrium,
    ex_ante_value,
    generate_initial_profiles,
)
from verify_equilibrium_existence import (
    BASE_PARAMS,
    DELTA_X_BASE,
    DELTA_X_COMP,
    DEV_MODE,
    DISTINCT_TOL,
    NPL_DAMPING,
    NPL_MAX_ITER,
    NPL_TOL,
    PSI_FIXED,
    RESIDUAL_VF_ITER,
    ccp_distance,
    compute_equilibrium_stats,
    make_sweep_context,
    pick_high_adoption_eq,
)


def _load_prop4_module():
    """Load Prop4/verify_proposition4.py helpers (Prop4/ is not a package)."""
    path = os.path.join(PROP4_DIR, "verify_proposition4.py")
    name = "verify_proposition4_for_forward_sim"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Prop4 script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_prop4 = _load_prop4_module()
AGENT_SWITCH_THRESH = _prop4.AGENT_SWITCH_THRESH
DELTA_VALUES = _prop4.DELTA_VALUES
MU_C = _prop4.MU_C
build_params_for_delta = _prop4.build_params_for_delta
detect_switch_tau = _prop4.detect_switch_tau
build_aggressive_init = _prop4.build_aggressive_init
tau_slice_stats = _prop4.tau_slice_stats
format_equilibrium_stats = _prop4.format_equilibrium_stats

OUTPUT_TXT = os.path.join(SCRIPT_DIR, "prop4_forward_sim_verification.txt")
FIGDIR = Path(SCRIPT_DIR) / "figures"

# t>0: dampened NPL to convergence vs frozen P^H_0 (not a single weak BR).
DEFAULT_N_BR = NPL_MAX_ITER


# ---------------------------------------------------------------------------
# Tau schedule
# ---------------------------------------------------------------------------

def tau_schedule(tau_bar: int, t_max: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Exogenous trust path: tau_t = max(0, tau_bar - t).

    Returns list of (t, tau_t) for t = 0 .. t_max inclusive.
    Default t_max = tau_bar so the path is t=0..tau_bar (tau: tau_bar .. 0).
    """
    tau_bar = int(tau_bar)
    if t_max is None:
        t_max = tau_bar
    return [(t, max(0, tau_bar - t)) for t in range(int(t_max) + 1)]


# ---------------------------------------------------------------------------
# Rate measurement (CCP means — no single-firm particle)
# ---------------------------------------------------------------------------

def overall_ccp_rates(P_C, P_G) -> Dict[str, float]:
    """Mean agent/HITL over all states (Prop4 compute_equilibrium_stats style)."""
    stats = compute_equilibrium_stats(P_C, P_G)
    return {
        "agent_C": float(stats["agent_C"]),
        "hitl_C": float(stats["hitl_C"]),
        "agent_G": float(stats["agent_G"]),
        "hitl_G": float(stats["hitl_G"]),
        "rate_mode": "overall",
    }


def tau_slice_rates(P_C, P_G, ss, tau_level: int) -> Optional[Dict[str, float]]:
    """Mean agent/HITL on states with tau == tau_level (diagnostic only)."""
    row = tau_slice_stats(P_C, P_G, ss, tau_level)
    if row is None:
        return None
    return {
        "agent_C": float(row["agent_C"]),
        "hitl_C": float(row["hitl_C"]),
        "agent_G": float(row["agent_G"]),
        "hitl_G": float(row["hitl_G"]),
        "n_states": int(row["n_states"]),
        "rate_mode": f"tau_slice_{int(tau_level)}",
    }


# ---------------------------------------------------------------------------
# Forced-tau payoffs + transitions (no model.py change)
# ---------------------------------------------------------------------------

def compute_payoffs_forced_tau(ss, params, tau_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mirror of model.compute_payoffs, but flow trust uses exogenous tau_t for
    every state (pin g_tau = tau_t / tau_bar). Other state dims (x, A_L, A_H, T)
    keep their natural values. Does not modify model.py.
    """
    s = ss.states
    x = s[:, 0].astype(float)
    A_L = s[:, 1].astype(float)
    A_H = s[:, 2].astype(float)
    T = s[:, 4]

    A_eff = A_L + params.alpha * A_H
    g_tau = (
        float(tau_t) / float(params.tau_bar)
        if params.tau_bar > 0
        else 0.0
    )
    c_x = params.theta1_intercept + params.theta1_slope * x
    c_0 = params.C(0)
    h_A = np.log(1.0 + A_eff)
    xi_T = np.array([params.xi(int(t)) for t in T])

    n = ss.n_states
    u_C = np.zeros((n, 3))
    u_G = np.zeros((n, 3))

    u_C[:, A_NONE] = -c_x - params.delta * h_A + params.mu_C * g_tau
    u_G[:, A_NONE] = -c_x - params.delta * h_A + params.mu_G * g_tau
    u_C[:, A_HITL] = -params.K_L - c_0 + params.mu_C * g_tau
    u_G[:, A_HITL] = -params.K_L - c_0 + params.mu_G * g_tau
    u_C[:, A_AGENT] = -params.K_H - c_0 + params.lam * g_tau - params.omega_C_E * xi_T
    u_G[:, A_AGENT] = -params.K_H - c_0 + params.lam * g_tau - params.omega_G_E * xi_T

    return u_C, u_G


def build_sparse_transitions_forced_tau(
    ss, params, P_C, P_G, N_C, N_G, tau_t: int,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """
    Local mirror of model.build_sparse_transitions with calendar τ forced.

    For every grid state (x, A_L, A_H, tau_label, T):
      - Mean-field CCP is read at (x, A_L, A_H, τ_t, T), not tau_label
      - Next-period τ is pinned to τ_t (exogenous schedule; endogenous trust
        law bypassed within the period VF)

    So only (x, A_L, A_H, T) vary; τ is the period's τ_t for all states.
    Does not modify model.py.
    """
    tau_t = int(tau_t)
    n = ss.n_states
    rows = [[] for _ in range(3)]
    cols = [[] for _ in range(3)]
    vals = [[] for _ in range(3)]

    for idx in range(n):
        x, A_L, A_H, _tau_label, T = tuple(ss.states[idx])
        A_eff = A_L + params.alpha * A_H

        # Competitor CCP at forced-τ slice of the same (x, A_L, A_H, T)
        s_mf = (int(x), int(A_L), int(A_H), tau_t, int(T))
        idx_mf = ss.state_index.get(s_mf, idx)
        p_C_L = P_C[idx_mf, A_HITL]
        p_C_H = P_C[idx_mf, A_AGENT]
        p_G_L = P_G[idx_mf, A_HITL]
        p_G_H = P_G[idx_mf, A_AGENT]

        n_C_comp = max(0, N_C - 1)
        n_G_comp = N_G
        E_AL = n_C_comp * p_C_L + n_G_comp * p_G_L
        E_AH = n_C_comp * p_C_H + n_G_comp * p_G_H

        A_max = params.N - 1
        AL_lo = min(int(np.floor(E_AL)), A_max)
        AL_hi = min(AL_lo + 1, A_max)
        wAL_hi = E_AL - AL_lo if AL_lo < AL_hi else 0.0

        AH_lo = min(int(np.floor(E_AH)), A_max)
        AH_hi = min(AH_lo + 1, A_max)
        wAH_hi = E_AH - AH_lo if AH_lo < AH_hi else 0.0

        A_grid = []
        for AL_n, wL in [(AL_lo, 1.0 - wAL_hi), (AL_hi, wAL_hi)]:
            if wL < 1e-12:
                continue
            for AH_n, wH in [(AH_lo, 1.0 - wAH_hi), (AH_hi, wAH_hi)]:
                if wH < 1e-12 or AL_n + AH_n > A_max:
                    continue
                A_grid.append((AL_n, AH_n, wL * wH))

        if T < params.T_bar:
            T_next_list = [(T, 1.0 - params.phi), (T + 1, params.phi)]
        else:
            T_next_list = [(params.T_bar, 1.0)]

        # Pin next τ to calendar τ_t (only other dims evolve)
        tau_list = [(tau_t, 1.0)]

        for a_i in range(3):
            if a_i == A_NONE:
                p_up = min(0.95, params.delta_x_base + params.delta_x_comp * A_eff)
                if x < params.x_bar:
                    x_list = [(x, 1.0 - p_up), (x + 1, p_up)]
                else:
                    x_list = [(params.x_bar, 1.0)]
            else:
                x_list = [(0, 1.0)]

            for x_n, wx in x_list:
                for AL_n, AH_n, wA in A_grid:
                    for tau_n, wt in tau_list:
                        for T_n, wT in T_next_list:
                            s_next = (x_n, AL_n, AH_n, tau_n, T_n)
                            if s_next in ss.state_index:
                                prob = wx * wA * wt * wT
                                if prob > 1e-15:
                                    rows[a_i].append(idx)
                                    cols[a_i].append(ss.state_index[s_next])
                                    vals[a_i].append(prob)

    F = []
    for a_i in range(3):
        mat = sparse.csr_matrix(
            (vals[a_i], (rows[a_i], cols[a_i])),
            shape=(n, n),
        )
        row_sums = np.array(mat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        mat = sparse.diags(1.0 / row_sums) @ mat
        F.append(mat)

    return F[0], F[1], F[2]


# ---------------------------------------------------------------------------
# NPL / VF under forced τ (symmetric + frozen-competitor)
# ---------------------------------------------------------------------------

def solve_vf_warm(
    u: np.ndarray,
    F_0,
    F_L,
    F_H,
    beta: float,
    W_init: Optional[np.ndarray] = None,
    max_vf_iter: int = 200,
    vf_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value-function iteration warm-started from W_init.
    Returns (W, choice_probs).
    """
    n = u.shape[0]
    W = np.zeros(n) if W_init is None else W_init.copy()
    for _ in range(max_vf_iter):
        v = np.column_stack([
            u[:, 0] + beta * F_0.dot(W),
            u[:, 1] + beta * F_L.dot(W),
            u[:, 2] + beta * F_H.dot(W),
        ])
        W_new = ex_ante_value(v)
        if np.max(np.abs(W_new - W)) < vf_tol:
            W = W_new
            break
        W = W_new
    v = np.column_stack([
        u[:, 0] + beta * F_0.dot(W),
        u[:, 1] + beta * F_L.dot(W),
        u[:, 2] + beta * F_H.dot(W),
    ])
    return W, choice_probs_from_values(v)


def run_npl_forced_tau(
    ss, params, N_C, N_G, tau_t: int,
    P_C_init=None, P_G_init=None,
    max_iter: int = NPL_MAX_ITER,
    tol: float = NPL_TOL,
    damping: float = NPL_DAMPING,
    max_vf_iter: int = RESIDUAL_VF_ITER,
    vf_tol: float = 1e-8,
    verbose: bool = False,
) -> dict:
    """
    Symmetric dampened NPL with forced-τ payoffs and forced-τ transitions.
    Replaces model.run_npl / compute_payoffs for the forward-sim calendar.
    """
    n = ss.n_states
    P_C = np.ones((n, 3)) / 3.0 if P_C_init is None else P_C_init.copy()
    P_G = np.ones((n, 3)) / 3.0 if P_G_init is None else P_G_init.copy()
    u_C, u_G = compute_payoffs_forced_tau(ss, params, tau_t)
    history: List[float] = []
    converged = False
    W_C = W_G = None
    k = -1

    for k in range(max_iter):
        F_0, F_L, F_H = build_sparse_transitions_forced_tau(
            ss, params, P_C, P_G, N_C, N_G, tau_t,
        )
        W_C, P_C_br = solve_vf_warm(
            u_C, F_0, F_L, F_H, params.beta, W_C, max_vf_iter, vf_tol,
        )
        W_G, P_G_br = solve_vf_warm(
            u_G, F_0, F_L, F_H, params.beta, W_G, max_vf_iter, vf_tol,
        )
        P_C_new = damping * P_C_br + (1.0 - damping) * P_C
        P_G_new = damping * P_G_br + (1.0 - damping) * P_G
        diff = max(
            float(np.max(np.abs(P_C_new - P_C))),
            float(np.max(np.abs(P_G_new - P_G))),
        )
        history.append(diff)
        if verbose and (k % 10 == 0 or diff < tol):
            ac = (P_C_new[:, 1] + P_C_new[:, 2]).mean()
            ag = (P_G_new[:, 1] + P_G_new[:, 2]).mean()
            print(f"  NPL-forced-τ {k:3d}: dP={diff:.2e}  adopt(C/G)={ac:.3f}/{ag:.3f}")
        P_C, P_G = P_C_new, P_G_new
        if diff < tol:
            converged = True
            break

    return {
        "P_C": P_C, "P_G": P_G, "W_C": W_C, "W_G": W_G,
        "u_C": u_C, "u_G": u_G,
        "converged": converged, "n_iter": k + 1, "history": history,
    }


def get_ph_profile_forced_tau(
    ss, params, N_C, N_G, tau_t: int, log_lines: List[str],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Multi-init P^H under forced τ=τ_t for all states (payoffs + transitions).

    Priority mirrors Prop4 get_ph_profile, but uses run_npl_forced_tau so the
    equilibrium is NOT computed under heterogeneous state-τ payoffs.
    """
    P_aggr = build_aggressive_init(ss.n_states)
    result = run_npl_forced_tau(
        ss, params, N_C, N_G, tau_t,
        P_C_init=P_aggr.copy(), P_G_init=P_aggr.copy(),
        verbose=False,
    )
    source = "NPL forced-τ (aggressive init)"

    if not result["converged"]:
        log_lines.append(
            "  WARNING: aggressive-init forced-τ NPL did not converge; "
            "trying uniform init..."
        )
        result = run_npl_forced_tau(
            ss, params, N_C, N_G, tau_t, verbose=False,
        )
        source = "NPL forced-τ (uniform init)"

    if result["converged"]:
        P_C, P_G = result["P_C"], result["P_G"]
        stats = compute_equilibrium_stats(P_C, P_G)
        if (stats["adopt_C"] + stats["adopt_G"]) / 2 >= 0.5:
            eq_type = classify_equilibrium(stats)
            log_lines.append(f"  P^H baseline — {source} (τ={tau_t} forced all states):")
            log_lines.extend(format_equilibrium_stats(stats))
            log_lines.append(f"       Type: {eq_type}")
            log_lines.append(f"       NPL iters: {result['n_iter']}")
            return {
                "P_C": P_C, "P_G": P_G,
                "W_C": result["W_C"], "W_G": result["W_G"],
                "u_C": result["u_C"], "u_G": result["u_G"],
                "stats": stats,
                "type": eq_type,
                "source": source,
            }, True

    # Multi-init search under forced τ
    log_lines.append(
        "  Aggressive/uniform forced-τ NPL not high-adoption; "
        "scanning initial profiles..."
    )
    inits = generate_initial_profiles(ss.n_states, n_inits=20, seed=42)
    equilibria = []
    for P_C_init, P_G_init, label in inits:
        res = run_npl_forced_tau(
            ss, params, N_C, N_G, tau_t,
            P_C_init=P_C_init, P_G_init=P_G_init, verbose=False,
        )
        if not res["converged"]:
            continue
        P_C, P_G = res["P_C"], res["P_G"]
        stats = compute_equilibrium_stats(P_C, P_G)
        if (stats["adopt_C"] + stats["adopt_G"]) / 2 < 0.3:
            continue
        is_new = True
        for eq in equilibria:
            if ccp_distance(P_C, P_G, eq["P_C"], eq["P_G"]) < DISTINCT_TOL:
                is_new = False
                break
        if is_new:
            equilibria.append({
                "P_C": P_C, "P_G": P_G,
                "W_C": res["W_C"], "W_G": res["W_G"],
                "u_C": res["u_C"], "u_G": res["u_G"],
                "stats": stats,
                "residual": 0.0,
                "label": label,
                "type": classify_equilibrium(stats),
            })

    eq_H = pick_high_adoption_eq(equilibria)
    if eq_H is not None:
        log_lines.append(
            f"  P^H baseline — forced-τ multi-init search (τ={tau_t}):"
        )
        log_lines.extend(format_equilibrium_stats(eq_H["stats"]))
        log_lines.append(f"       Type: {eq_H['type']}")
        log_lines.append(f"       Init label: {eq_H.get('label', '?')}")
        return {
            "P_C": eq_H["P_C"], "P_G": eq_H["P_G"],
            "W_C": eq_H.get("W_C"), "W_G": eq_H.get("W_G"),
            "u_C": eq_H.get("u_C"), "u_G": eq_H.get("u_G"),
            "stats": eq_H["stats"],
            "type": eq_H["type"],
            "source": "forced-τ NPL search",
        }, True

    log_lines.append("  WARNING: P^H baseline NOT FOUND under forced-τ NPL")
    return None, False


def npl_own_vs_frozen_comp(
    ss,
    params,
    P_own_C: np.ndarray,
    P_own_G: np.ndarray,
    P_comp_C: np.ndarray,
    P_comp_G: np.ndarray,
    N_C: int,
    N_G: int,
    u_C: np.ndarray,
    u_G: np.ndarray,
    tau_t: int,
    W_C: Optional[np.ndarray] = None,
    W_G: Optional[np.ndarray] = None,
    max_iter: int = NPL_MAX_ITER,
    tol: float = NPL_TOL,
    damping: float = NPL_DAMPING,
    max_vf_iter: int = RESIDUAL_VF_ITER,
    vf_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float], int]:
    """
    Dampened NPL for own CCP facing frozen competitor CCPs, under forced τ_t.

    Transitions / mean-field always from (P_comp_C, P_comp_G) with τ pinned to
    τ_t — typically P^H_0 — so E[A_L], E[A_H] stay at the t=0 competitor
    environment (looked up on the τ_t slice). Own CCP warm-started from
    P_own_* (usually CCP_{t-1}).
    """
    F_0, F_L, F_H = build_sparse_transitions_forced_tau(
        ss, params, P_comp_C, P_comp_G, N_C, N_G, tau_t,
    )

    P_C = P_own_C.copy()
    P_G = P_own_G.copy()
    diffs: List[float] = []
    n_done = 0

    for k in range(max(1, int(max_iter))):
        W_C, P_C_br = solve_vf_warm(
            u_C, F_0, F_L, F_H, params.beta, W_C, max_vf_iter, vf_tol,
        )
        W_G, P_G_br = solve_vf_warm(
            u_G, F_0, F_L, F_H, params.beta, W_G, max_vf_iter, vf_tol,
        )
        P_C_new = damping * P_C_br + (1.0 - damping) * P_C
        P_G_new = damping * P_G_br + (1.0 - damping) * P_G
        diff = max(
            float(np.max(np.abs(P_C_new - P_C))),
            float(np.max(np.abs(P_G_new - P_G))),
        )
        diffs.append(diff)
        P_C, P_G = P_C_new, P_G_new
        n_done = k + 1
        if diff < tol:
            break

    return P_C, P_G, W_C, W_G, diffs, n_done


def init_W_from_ccp(
    ss, params, P_C, P_G, N_C, N_G, u_C, u_G, tau_t: int,
    max_vf_iter: int = RESIDUAL_VF_ITER, vf_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ex-ante values consistent with an initial CCP under forced τ."""
    F_0, F_L, F_H = build_sparse_transitions_forced_tau(
        ss, params, P_C, P_G, N_C, N_G, tau_t,
    )
    W_C, _ = solve_vf_warm(u_C, F_0, F_L, F_H, params.beta, None, max_vf_iter, vf_tol)
    W_G, _ = solve_vf_warm(u_G, F_0, F_L, F_H, params.beta, None, max_vf_iter, vf_tol)
    return W_C, W_G


# ---------------------------------------------------------------------------
# Sequential forward path (own CCP linked; competitors frozen at P^H_0)
# ---------------------------------------------------------------------------

def run_sequential_path(
    ss,
    params,
    N_C: int,
    N_G: int,
    u_C0,
    u_G0,
    P_C_init: np.ndarray,
    P_G_init: np.ndarray,
    schedule: List[Tuple[int, int]],
    n_br: int = DEFAULT_N_BR,
    damping: float = NPL_DAMPING,
    tol: float = NPL_TOL,
    W_C0: Optional[np.ndarray] = None,
    W_G0: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Sequential forward path with frozen competitor adoption at t=0 CCP.

    Every t (incl. 0): τ forced to τ_t on payoffs + transitions.
    t=0: keep forced-τ NPL P^H as own CCP; rates = overall means.
    t>0: own CCP warm-started from CCP_{t-1}; NPL vs frozen P_comp = P^H_0.
    """
    P_comp_C = P_C_init.copy()
    P_comp_G = P_G_init.copy()

    P_C = P_C_init.copy()
    P_G = P_G_init.copy()
    tau0 = int(schedule[0][1]) if schedule else int(params.tau_bar)
    if W_C0 is not None and W_G0 is not None:
        W_C, W_G = W_C0.copy(), W_G0.copy()
    else:
        W_C, W_G = init_W_from_ccp(
            ss, params, P_comp_C, P_comp_G, N_C, N_G, u_C0, u_G0, tau0,
        )

    rows: List[Dict[str, Any]] = []

    for t, tau_t in schedule:
        if t == 0:
            diffs = [0.0]
            n_iters = 0
            ccp_source = "NPL_PH_forced_tau"
            rates = overall_ccp_rates(P_C, P_G)
        else:
            u_C_t, u_G_t = compute_payoffs_forced_tau(ss, params, tau_t)
            P_C, P_G, W_C, W_G, diffs, n_iters = npl_own_vs_frozen_comp(
                ss, params,
                P_own_C=P_C, P_own_G=P_G,
                P_comp_C=P_comp_C, P_comp_G=P_comp_G,
                N_C=N_C, N_G=N_G,
                u_C=u_C_t, u_G=u_G_t,
                tau_t=tau_t,
                W_C=W_C, W_G=W_G,
                max_iter=n_br, tol=tol, damping=damping,
            )
            ccp_source = "NPL_frozen_P0"
            rates = overall_ccp_rates(P_C, P_G)

        rows.append({
            "t": int(t),
            "tau_t": int(tau_t),
            "agent_C": rates["agent_C"],
            "hitl_C": rates["hitl_C"],
            "agent_G": rates["agent_G"],
            "hitl_G": rates["hitl_G"],
            "rate_mode": rates.get("rate_mode", "?"),
            "dP": float(diffs[-1]) if diffs else np.nan,
            "n_br": int(n_iters),
            "ccp_source": ccp_source,
        })

    return {
        "path_rows": rows,
        "P_C_final": P_C,
        "P_G_final": P_G,
        "W_C_final": W_C,
        "W_G_final": W_G,
        "P_comp_C": P_comp_C,
        "P_comp_G": P_comp_G,
    }


def run_delta_forward(
    delta: float,
    ss,
    N_C: int,
    N_G: int,
    log_lines: List[str],
    t_max: Optional[int],
    n_br: int,
    damping: float,
    tol: float = NPL_TOL,
) -> Dict[str, Any]:
    """t=0: forced-τ NPL P^H; t>0: NPL vs frozen P^H_0 with forced-τ payoffs+F."""
    params = build_params_for_delta(delta)
    tau_bar = int(params.tau_bar)
    schedule = tau_schedule(tau_bar, t_max=t_max)
    tau0 = int(schedule[0][1])  # = tau_bar at t=0

    log_lines.append(f"[ delta = {delta} ]")
    log_lines.append(
        f"  Parameters: psi={PSI_FIXED}, N={params.N}, mu_C={MU_C}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}, "
        f"tau_bar={tau_bar}"
    )
    log_lines.append(
        f"  Tau schedule: tau_t = max(0, {tau_bar} - t); "
        f"path = {[(t, tau) for t, tau in schedule]}"
    )
    log_lines.append(
        "  Method: EVERY t forces τ=τ_t on ALL states (payoffs g_tau=τ_t/τ̄ + "
        "transitions: mf at (x,A_L,A_H,τ_t,T), next τ pinned to τ_t; only "
        "(x,A_L,A_H,T) vary). "
        f"t=0 = multi-init NPL P^H under forced τ={tau0}; "
        f"t>0 = dampened NPL (max_iter={n_br}, tol={tol}, damping={damping}) "
        "warm-started from CCP_{t-1}; competitor CCP frozen at P^H_0. "
        "Rates: overall CCP means of updated own CCP at every t."
    )
    log_lines.append(
        "  Freeze: build_sparse_transitions_forced_tau / mean-field use "
        "P_comp = P^H_0 for all t>0 (E[A_L], E[A_H] fixed at t=0 competitor "
        "environment, looked up on the τ_t slice). Own CCP updates; W linked."
    )
    log_lines.append(
        "  Rates: overall mean over all states (compute_equilibrium_stats) "
        "at every t — consistent (forced-τ makes tau-labeled copies "
        "economically identical). τ=τ_t slice logged as diagnostic only."
    )

    ph_profile, ph_ok = get_ph_profile_forced_tau(
        ss, params, N_C, N_G, tau0, log_lines,
    )
    if not ph_ok:
        log_lines.append("  Forward sim SKIPPED (no P^H from forced-τ NPL).")
        log_lines.append("")
        return {
            "delta": delta,
            "ph_exists": False,
            "path_rows": [],
            "switch_C": np.nan,
            "switch_G": np.nan,
            "c_switches_first": False,
        }

    P_C0, P_G0 = ph_profile["P_C"], ph_profile["P_G"]
    u_C0 = ph_profile.get("u_C")
    u_G0 = ph_profile.get("u_G")
    if u_C0 is None or u_G0 is None:
        u_C0, u_G0 = compute_payoffs_forced_tau(ss, params, tau0)
    overall0 = overall_ccp_rates(P_C0, P_G0)
    log_lines.append(
        f"  P^H overall CCP means (t=0 primary; forced τ={tau0} all states): "
        f"agent_C={overall0['agent_C']:.3f}, HITL_C={overall0['hitl_C']:.3f}, "
        f"agent_G={overall0['agent_G']:.3f}, HITL_G={overall0['hitl_G']:.3f}"
    )
    log_lines.append(
        "  Competitor adoption environment: frozen at this P^H_0 for all t>0."
    )

    # Diagnostic: tau=tau0 labeled slice (should ≈ overall under forced-τ sync)
    slice0 = tau_slice_rates(P_C0, P_G0, ss, tau0)
    if slice0 is not None:
        log_lines.append(
            f"  P^H tau={tau0} label slice (diagnostic; expect ≈ overall): "
            f"agent_C={slice0['agent_C']:.3f}, HITL_C={slice0['hitl_C']:.3f}, "
            f"agent_G={slice0['agent_G']:.3f}, HITL_G={slice0['hitl_G']:.3f} "
            f"(n_states={slice0['n_states']})"
        )

    path_out = run_sequential_path(
        ss, params, N_C, N_G, u_C0, u_G0,
        P_C0, P_G0, schedule,
        n_br=n_br, damping=damping, tol=tol,
        W_C0=ph_profile.get("W_C"), W_G0=ph_profile.get("W_G"),
    )
    path_rows = path_out["path_rows"]

    slice_like = [
        {
            "tau": r["tau_t"],
            "agent_C": r["agent_C"],
            "agent_G": r["agent_G"],
        }
        for r in path_rows
        if not np.isnan(r["agent_C"])
    ]
    seen = {}
    for r in slice_like:
        seen[r["tau"]] = r
    slice_rows = [seen[tau] for tau in sorted(seen.keys())]
    switch_C = detect_switch_tau(slice_rows, "agent_C")
    switch_G = detect_switch_tau(slice_rows, "agent_G")
    c_first = switch_C > switch_G if switch_C >= 0 and switch_G >= 0 else False

    log_lines.append("")
    log_lines.append(
        "  Forward path rates (t=0=forced-τ NPL P^H overall; "
        "t>0=NPL vs frozen P^H_0, forced-τ payoffs+transitions, overall means):"
    )
    log_lines.append(
        f"  {'t':>3}  {'tau_t':>5}  "
        f"{'agent_C':>8}  {'HITL_C':>8}  {'agent_G':>8}  {'HITL_G':>8}  "
        f"{'rate_mode':>16}  {'dP':>8}  {'n_iter':>6}  {'src':>18}"
    )
    log_lines.append("  " + "-" * 114)
    for r in path_rows:
        log_lines.append(
            f"  {r['t']:>3d}  {r['tau_t']:>5d}  "
            f"{r['agent_C']:>8.3f}  {r['hitl_C']:>8.3f}  "
            f"{r['agent_G']:>8.3f}  {r['hitl_G']:>8.3f}  "
            f"{str(r.get('rate_mode', '?')):>16}  "
            f"{r['dP']:>8.2e}  {int(r.get('n_br', 0)):>6d}  "
            f"{r.get('ccp_source', '?'):>18}"
        )

    log_lines.append("")
    log_lines.append(
        f"  C-type switch tau along path (agent < {AGENT_SWITCH_THRESH}): "
        f"{switch_C if switch_C >= 0 else 'never'}"
    )
    log_lines.append(
        f"  G-type switch tau along path (agent < {AGENT_SWITCH_THRESH}): "
        f"{switch_G if switch_G >= 0 else 'never'}"
    )
    log_lines.append(
        f"  Key test — C switches first (path): "
        f"{'PASS' if c_first else 'FAIL' if switch_C >= 0 and switch_G >= 0 else 'N/A'}"
    )
    log_lines.append("")

    print(
        f"[ delta = {delta} ]  sequential forward done — "
        f"n_br_max={n_br} (t>0), T_end={schedule[-1][0]}, "
        f"C-first={'Yes' if c_first else 'No'}",
        flush=True,
    )

    return {
        "delta": delta,
        "ph_exists": True,
        "path_rows": path_rows,
        "switch_C": switch_C,
        "switch_G": switch_G,
        "c_switches_first": c_first,
        "ph_agent_C": ph_profile["stats"]["agent_C"],
        "ph_agent_G": ph_profile["stats"]["agent_G"],
    }


def run_forward_sweep(
    log_lines: List[str],
    delta_values=None,
    t_max: Optional[int] = None,
    n_br: int = DEFAULT_N_BR,
    damping: float = NPL_DAMPING,
    tol: float = NPL_TOL,
) -> List[Dict[str, Any]]:
    if delta_values is None:
        delta_values = DELTA_VALUES

    ss, N_C, N_G, _inits = make_sweep_context()
    tau_bar = int(BASE_PARAMS["tau_bar"])

    log_lines.append(
        "PROPOSITION 4 FORWARD SIM: forced-τ all states every t; "
        "t=0 NPL P^H + t>0 NPL (frozen P^H_0 comps)"
    )
    log_lines.append("=" * 105)
    log_lines.append(
        f"Fixed parameters: psi={PSI_FIXED}, mu_C={MU_C}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}"
    )
    log_lines.append(
        f"State space (DEV_MODE={DEV_MODE}): "
        f"N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"T_bar={BASE_PARAMS['T_bar']}, tau_bar={tau_bar}, "
        f"n_states={ss.n_states}"
    )
    log_lines.append(
        f"Tau schedule formula: tau_t = max(0, tau_bar - t) "
        f"with tau_bar={tau_bar}"
    )
    log_lines.append(
        "Forced τ: at every t, ALL states use τ=τ_t for flow payoffs "
        "(g_tau=τ_t/τ̄) and transitions (mf CCP at (x,A_L,A_H,τ_t,T); "
        "next τ pinned to τ_t). Only (x, A_L, A_H, T) vary. Grid still "
        "has tau labels, but they are economically identical under force."
    )
    log_lines.append(
        "Firm init: NONE (no Prop3-style (0,2,0,0) / reachable-mask lock). "
        "Rates are CCP means, not a single tracked firm state."
    )
    log_lines.append(
        "Method (PRIMARY): t=0 = multi-init NPL P^H under forced τ=τ̄ "
        "(get_ph_profile_forced_tau; overall CCP means; NO extra BR). "
        f"t>0 = dampened NPL (max_iter={n_br}, tol={tol}, damping={damping}) "
        "warm-started from CCP_{t-1}; competitor CCP frozen at P^H_0; "
        "forced-τ payoffs + forced-τ transitions; rates = overall means "
        "of updated own CCP. NOT heterogeneous state-τ at t=0. "
        "NOT overall→tau-slice rate switch. NOT linked BR updating comps."
    )
    log_lines.append(
        "Honesty: sequential NPL along an exogenous tau path with frozen "
        "competitors — not a full closed-loop RE path (full RE would require "
        "agents to correctly anticipate the entire future tau schedule inside "
        "the continuation value)."
    )
    log_lines.append(
        f"NPL settings (t>0): max_iter={n_br}, tol={tol}, damping={damping}"
    )
    log_lines.append(
        f"Horizon: t_max={'tau_bar (t=0..tau_bar)' if t_max is None else t_max}"
    )
    log_lines.append("")

    summary_rows = []
    for delta in delta_values:
        print(f"[ delta = {delta} ]  running sequential forward sim...", flush=True)
        row = run_delta_forward(
            delta, ss, N_C, N_G, log_lines,
            t_max=t_max, n_br=n_br, damping=damping, tol=tol,
        )
        summary_rows.append(row)

    log_lines.append("=" * 105)
    log_lines.append("TABLE: Proposition 4 Forward-Sim Summary")
    log_lines.append("-" * 105)
    header = (
        f"{'delta':<7}| {'P^H?':<6}| {'sw_C':<5}| {'sw_G':<5}| "
        f"{'C 1st?':<7}| baseline agent (C/G)"
    )
    log_lines.append(header)
    log_lines.append("-" * 105)
    for row in summary_rows:
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
        agent_baseline = (
            f"{row['ph_agent_C']:.3f}/{row['ph_agent_G']:.3f}"
            if row.get("ph_exists")
            else "N/A"
        )
        log_lines.append(
            f"{row['delta']:<7.1f}| "
            f"{str(row['ph_exists']):<6}| "
            f"{sw_C:<5}| {sw_G:<5}| "
            f"{str(row.get('c_switches_first', False)):<7}| "
            f"{agent_baseline}"
        )

    return summary_rows


# ---------------------------------------------------------------------------
# Parse + plot (no NPL; reads OUTPUT_TXT)
# ---------------------------------------------------------------------------

def parse_forward_txt(text: str) -> Dict[str, Any]:
    """Parse prop4_forward_sim_verification.txt into blocks with path rows."""
    block_pat = re.compile(
        r"\[\s*delta\s*=\s*([0-9.]+)\s*\](.*?)(?=\[\s*delta\s*=|\n={10,}|\Z)",
        re.S | re.I,
    )

    blocks = []
    for m in block_pat.finditer(text):
        d = float(m.group(1))
        body = m.group(2)
        path_rows = []
        in_path = False
        for line in body.splitlines():
            if "Forward path rates" in line:
                in_path = True
                continue
            if in_path and (
                "C-type switch" in line
                or "G-type switch" in line
                or "Key test" in line
                or line.strip().startswith("Parameters")
            ):
                in_path = False
            if not in_path:
                continue
            parts = line.split()
            try:
                # Primary: t tau agent_C HITL_C agent_G HITL_G  [optional extras]
                if len(parts) >= 6:
                    t = int(parts[0])
                    tau_t = int(parts[1])
                    vals = list(map(float, parts[2:6]))
                    path_rows.append({
                        "t": t,
                        "tau_t": tau_t,
                        "agent_C": vals[0],
                        "hitl_C": vals[1],
                        "agent_G": vals[2],
                        "hitl_G": vals[3],
                    })
            except ValueError:
                continue
        blocks.append({"delta": d, "path": path_rows})
    return {"blocks": blocks}


def plot_forward_results(
    data: Dict[str, Any],
    figdir: Optional[Path] = None,
    overlay_slice: bool = False,  # kept for CLI compat; unused (old Option A removed)
) -> List[Path]:
    """
    Plot C/G agent & HITL CCP-mean rates vs t for each delta.

    Saves under prop4_forward_sim/figures/prop4_forward_*.
    """
    del overlay_slice  # old Option-A overlay removed
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = Path(figdir) if figdir is not None else FIGDIR
    figdir.mkdir(parents=True, exist_ok=True)

    blocks = sorted(
        [b for b in data.get("blocks", []) if b.get("path")],
        key=lambda b: b["delta"],
    )
    if not blocks:
        print("  [warn] Prop4 forward: no path blocks to plot")
        return []

    rate_lab = (
        r"overall CCP means; forced-$\tau_t$ all states; "
        r"$t$>0 NPL vs frozen $P^H_0$"
    )

    colors = {
        "C": "#9467bd",
        "G": "#ff7f0e",
        "HITL_C": "#17becf",
        "HITL_G": "#c44e52",
    }

    paths: List[Path] = []
    n = len(blocks)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), sharey=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for i, blk in enumerate(blocks):
        ax = axes_flat[i]
        rows = blk["path"]
        ts = [r["t"] for r in rows]
        ax.plot(
            ts, [r["agent_C"] for r in rows], "o-",
            color=colors["C"], label="agent(C)",
        )
        ax.plot(
            ts, [r["hitl_C"] for r in rows], "s--",
            color=colors["HITL_C"], alpha=0.85, label="HITL(C)",
        )
        ax.plot(
            ts, [r["agent_G"] for r in rows], "D-",
            color=colors["G"], label="agent(G)",
        )
        ax.plot(
            ts, [r["hitl_G"] for r in rows], "^--",
            color=colors["HITL_G"], alpha=0.85, label="HITL(G)",
        )
        ax.axhline(0.3, color="gray", linestyle=":", linewidth=1)
        tau_ann = ", ".join(f"t{r['t']}→τ{r['tau_t']}" for r in rows[:4])
        if len(rows) > 4:
            tau_ann += ", …"
        ax.set_title(rf"$\delta$={blk['delta']}" + f"\n({tau_ann})", fontsize=10)
        ax.set_xlabel(r"$t$")
        if i % ncols == 0:
            ax.set_ylabel("CCP mean rate")
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        r"Prop 4 forward: C/G agent & HITL vs $t$"
        "\n"
        rf"({rate_lab}; $t$=0 forced-$\tau$ NPL $P^H$; comps frozen; "
        r"$\tau_t=\max(0,\bar\tau-t)$)",
        y=1.04,
    )
    fig.tight_layout()
    stem = "prop4_forward_rates_vs_t"
    for ext in ("png", "pdf"):
        p = figdir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(p)
        print(f"  wrote {p}")
    plt.close(fig)

    for blk in blocks:
        rows = blk["path"]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ts = [r["t"] for r in rows]
        ax.plot(ts, [r["agent_C"] for r in rows], "o-", color=colors["C"], label="agent(C)")
        ax.plot(ts, [r["hitl_C"] for r in rows], "s--", color=colors["HITL_C"], label="HITL(C)")
        ax.plot(ts, [r["agent_G"] for r in rows], "D-", color=colors["G"], label="agent(G)")
        ax.plot(ts, [r["hitl_G"] for r in rows], "^--", color=colors["HITL_G"], label="HITL(G)")
        ax.axhline(0.3, color="gray", linestyle=":", linewidth=1, label="agent=0.3")
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ts)
        ax2.set_xticklabels([str(r["tau_t"]) for r in rows])
        ax2.set_xlabel(r"$\tau_t$ (imposed)")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel("CCP mean rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            rf"Prop 4 forward vs $t$ "
            rf"($\delta$={blk['delta']}; {rate_lab})"
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        d_tag = str(blk["delta"]).replace(".", "p")
        stem_d = f"prop4_forward_rates_vs_t_delta{d_tag}"
        for ext in ("png", "pdf"):
            p = figdir / f"{stem_d}.{ext}"
            fig.savefig(p, bbox_inches="tight", dpi=150)
            paths.append(p)
            print(f"  wrote {p}")
        plt.close(fig)

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prop4 forward simulation: at every t force τ=τ_t on ALL states "
            "(payoffs + transitions; only x,A_L,A_H,T vary). "
            "t=0 = multi-init NPL P^H under forced τ=τ̄; "
            "t>0 = NPL vs frozen P^H_0 along tau_t = max(0, tau_bar - t)."
        )
    )
    parser.add_argument(
        "--delta", type=float, nargs="*", default=None,
        help="Optional subset of delta values (default: Prop4 full grid).",
    )
    parser.add_argument(
        "--t-max", type=int, default=None,
        help="Max t inclusive (default: tau_bar, i.e. t=0..tau_bar).",
    )
    parser.add_argument(
        "--n-br", type=int, default=DEFAULT_N_BR, dest="n_br",
        help=(
            "Max dampened NPL iterations per period for t>0 "
            f"(default {DEFAULT_N_BR} = NPL_MAX_ITER; stops early at --tol)."
        ),
    )
    parser.add_argument(
        "--tol", type=float, default=NPL_TOL,
        help=f"NPL convergence tol for t>0 (default {NPL_TOL}).",
    )
    parser.add_argument(
        "--damping", type=float, default=NPL_DAMPING,
        help=f"NPL damping (default {NPL_DAMPING}).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="After verification, plot from in-memory / written txt.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Only parse OUTPUT_TXT and plot (no NPL / no forward sim).",
    )
    parser.add_argument(
        "--plot-slice", action="store_true",
        help="(Deprecated) old Option-A slice overlay; ignored.",
    )
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.isfile(OUTPUT_TXT):
            print(f"No results file at {OUTPUT_TXT}; run without --plot-only first.")
            sys.exit(1)
        with open(OUTPUT_TXT, "r", encoding="utf-8") as f:
            text = f.read()
        if "NOT RUN" in text[:500].upper() and "[ delta =" not in text.lower():
            print(
                f"Results file is a NOT-RUN placeholder: {OUTPUT_TXT}\n"
                "Run without --plot-only to produce sequential forward results."
            )
            sys.exit(1)
        data = parse_forward_txt(text)
        print(f"Plotting from {OUTPUT_TXT}")
        plot_forward_results(data)
        return

    delta_values = args.delta if args.delta else DELTA_VALUES

    print("=" * 70)
    print(
        "PROPOSITION 4: Forward Sim "
        "(forced-τ all states; t=0 NPL P^H; t>0 frozen P^H_0)"
    )
    print("=" * 70)
    print(
        f"Parameters: psi={PSI_FIXED}, mu_C={MU_C}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}"
    )
    print(f"Delta sweep: {delta_values}")
    print(
        f"DEV_MODE={DEV_MODE}  "
        f"(N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"tau_bar={BASE_PARAMS['tau_bar']})"
    )
    print(f"Schedule: tau_t = max(0, tau_bar - t)")
    print("Forced τ: all states use τ_t for payoffs + transitions every t")
    print("Rates: overall CCP means at every t (consistent)")
    print("t>0: competitor CCP frozen at P^H_0")
    print(f"n_br_max={args.n_br}, tol={args.tol}, damping={args.damping}")
    print(f"Output file: {OUTPUT_TXT}")
    print()

    log_lines: List[str] = []
    summary_rows = run_forward_sweep(
        log_lines,
        delta_values=delta_values,
        t_max=args.t_max,
        n_br=args.n_br,
        damping=args.damping,
        tol=args.tol,
    )

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
        f.write("\n")

    print()
    print("=" * 70)
    print(f"Forward verification complete. Results written to:\n  {OUTPUT_TXT}")
    print("=" * 70)

    if args.plot:
        with open(OUTPUT_TXT, "r", encoding="utf-8") as f:
            text = f.read()
        data = parse_forward_txt(text)
        if not any(b.get("path") for b in data["blocks"]):
            data = {
                "blocks": [
                    {"delta": r["delta"], "path": r.get("path_rows", [])}
                    for r in summary_rows
                    if r.get("path_rows")
                ],
            }
        plot_forward_results(data)


if __name__ == "__main__":
    main()
