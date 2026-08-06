"""
Proposition 2 — sequential forward simulation along an exogenous agent-adoption
shock path k(t).

Complement to Prop2/verify_proposition2.py (static cascade sweep over k).
Competitor agent count follows

    k(t) = min(t, N-1),   t = 0, 1, ..., T_max

so with DEV_MODE N=4 (A_max=3): t=0→0, t=1→1, t=2→2, t=3→3.

At EVERY t (including t=0 for the path environment), industry competitor
counts are forced so that A_H = k(t) and A_L = 0 on all states:
  - Flow payoffs: A_eff = alpha * k(t) for all states (compute_payoffs_forced_AH)
  - Transitions: mean-field CCP looked up at (x, 0, k(t), τ_fixed, T);
    next (A_L, A_H) pinned to (0, k(t)); τ pinned to τ_fixed.
  - τ and other ModelParams stay fixed (τ_fixed = τ̄). No model.py change.

Procedure (per delta):
  t=0  — Prop1-style direct verification of P^L
         (verify_low_adoption_eq + refine_low_eq_on_reachable; NOT random NPL).
         Record CCP at tracked firm state s_0 = (0, 0, 0, τ̄, 0).
  t>0  — Warm-start own CCP (and W if available) from CCP_{t-1}; force
         A_H=k(t), A_L=0, τ=τ̄ on payoffs + transitions; dampened NPL to
         convergence. Rates = CCP at tracked firm s_t = (0, 0, k(t), τ̄, 0).

Honesty note: sequential NPL along an exogenous k(t) / A_H path — not a full
closed-loop RE path; analogous to Prop2's one-shot shock k as a time path.

Usage (from simulation/):
    python prop2_forward_sim/verify_proposition2_forward_sim.py
    python prop2_forward_sim/verify_proposition2_forward_sim.py --delta 2.0 3.0
    python prop2_forward_sim/verify_proposition2_forward_sim.py --n-br 100
    python prop2_forward_sim/verify_proposition2_forward_sim.py --plot-only

Output (defaults under this folder):
    prop2_forward_sim/prop2_forward_sim_verification.txt
    prop2_forward_sim/figures/prop2_forward_rates_vs_t.{png,pdf}
    prop2_forward_sim/figures/prop2_forward_rates_vs_t_delta{d}.{png,pdf}
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
PROP2_DIR = os.path.join(ROOT_DIR, "Prop2")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import (
    A_AGENT,
    A_HITL,
    A_NONE,
    compute_payoffs,
)
from npl import (
    choice_probs_from_values,
    ex_ante_value,
)
from verify_equilibrium_existence import (
    BASE_PARAMS,
    DELTA_X_BASE,
    DELTA_X_COMP,
    DEV_MODE,
    NPL_DAMPING,
    NPL_MAX_ITER,
    NPL_TOL,
    PSI_FIXED,
    RESIDUAL_VF_ITER,
    compute_equilibrium_stats,
    make_sweep_context,
)


def _load_prop2_module():
    """Load Prop2/verify_proposition2.py helpers (Prop2/ is not a package)."""
    path = os.path.join(PROP2_DIR, "verify_proposition2.py")
    name = "verify_proposition2_for_forward_sim"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Prop2 script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_prop2 = _load_prop2_module()
DELTA_VALUES = _prop2.DELTA_VALUES  # [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
build_params_for_delta = _prop2.build_params_for_delta
format_equilibrium_stats = _prop2.format_equilibrium_stats
get_pl_profile = _prop2.get_pl_profile

OUTPUT_TXT = os.path.join(SCRIPT_DIR, "prop2_forward_sim_verification.txt")
FIGDIR = Path(SCRIPT_DIR) / "figures"

DEFAULT_N_BR = NPL_MAX_ITER

# Tracked firm: fixed (x, A_L, tau, T); A_H follows k(t).
TRACKED_X = 0
TRACKED_A_L = 0
TRACKED_T = 0


# ---------------------------------------------------------------------------
# k(t) / A_H schedule
# ---------------------------------------------------------------------------

def k_schedule(N: int, t_max: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Exogenous agent-adopter shock path: k(t) = min(t, N-1).

    Returns list of (t, k_t) for t = 0 .. t_max inclusive.
    Default t_max = N-1 so the path is t=0..N-1 (k: 0 .. N-1).
    """
    A_max = int(N) - 1
    if t_max is None:
        t_max = A_max
    return [(t, min(t, A_max)) for t in range(int(t_max) + 1)]


def tracked_state(k_t: int, tau_fixed: int) -> Tuple[int, int, int, int, int]:
    """Single tracked firm state (x, A_L, A_H, tau, T) at calendar t."""
    return (
        int(TRACKED_X),
        int(TRACKED_A_L),
        int(k_t),
        int(tau_fixed),
        int(TRACKED_T),
    )


# ---------------------------------------------------------------------------
# Rate measurement (tracked-firm CCP + overall diagnostic)
# ---------------------------------------------------------------------------

def ccp_at_state(P_C, P_G, ss, state: Tuple[int, ...]) -> Optional[Dict[str, float]]:
    """CCP agent/HITL/NONE for C and G at one state index."""
    idx = ss.state_index.get(tuple(state))
    if idx is None:
        return None
    return {
        "agent_C": float(P_C[idx, A_AGENT]),
        "hitl_C": float(P_C[idx, A_HITL]),
        "none_C": float(P_C[idx, A_NONE]),
        "agent_G": float(P_G[idx, A_AGENT]),
        "hitl_G": float(P_G[idx, A_HITL]),
        "none_G": float(P_G[idx, A_NONE]),
        "rate_mode": f"tracked_{state}",
    }


def overall_ccp_rates(P_C, P_G) -> Dict[str, float]:
    """Mean agent/HITL over all states (diagnostic)."""
    stats = compute_equilibrium_stats(P_C, P_G)
    return {
        "agent_C": float(stats["agent_C"]),
        "hitl_C": float(stats["hitl_C"]),
        "agent_G": float(stats["agent_G"]),
        "hitl_G": float(stats["hitl_G"]),
        "rate_mode": "overall",
    }


# ---------------------------------------------------------------------------
# Forced-A_H payoffs + transitions (no model.py change)
# ---------------------------------------------------------------------------

def compute_payoffs_forced_AH(
    ss, params, k_t: int, tau_fixed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mirror of model.compute_payoffs, but industry adoption uses exogenous
    (A_L, A_H) = (0, k_t) for every state (pin A_eff = alpha * k_t) and
    flow trust uses fixed tau_fixed (g_tau = tau_fixed / tau_bar).
    Other dims (x, T) keep natural values. Does not modify model.py.
    """
    s = ss.states
    x = s[:, 0].astype(float)
    T = s[:, 4]

    A_eff = float(params.alpha) * float(k_t)
    g_tau = (
        float(tau_fixed) / float(params.tau_bar)
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


def build_sparse_transitions_forced_AH(
    ss, params, P_C, P_G, N_C, N_G, k_t: int, tau_fixed: int,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """
    Local mirror of model.build_sparse_transitions with calendar A_H forced.

    For every grid state (x, A_L_label, A_H_label, tau_label, T):
      - Mean-field CCP is read at (x, 0, k_t, tau_fixed, T)
      - Next-period (A_L, A_H) pinned to (0, k_t)
      - Next-period τ pinned to tau_fixed

    So only (x, T) evolve endogenously within the period VF; competitor
    agent count is the period's k(t). Does not modify model.py.
    """
    k_t = int(k_t)
    tau_fixed = int(tau_fixed)
    n = ss.n_states
    rows = [[] for _ in range(3)]
    cols = [[] for _ in range(3)]
    vals = [[] for _ in range(3)]
    A_eff = float(params.alpha) * float(k_t)
    A_grid = [(0, k_t, 1.0)]  # pin next (A_L, A_H)
    tau_list = [(tau_fixed, 1.0)]

    for idx in range(n):
        x, _A_L_lab, _A_H_lab, _tau_lab, T = tuple(ss.states[idx])

        s_mf = (int(x), 0, k_t, tau_fixed, int(T))
        idx_mf = ss.state_index.get(s_mf, idx)
        p_C_L = P_C[idx_mf, A_HITL]
        p_C_H = P_C[idx_mf, A_AGENT]
        p_G_L = P_G[idx_mf, A_HITL]
        p_G_H = P_G[idx_mf, A_AGENT]

        # Keep E_AL / E_AH only for documentation parity; A next is pinned.
        _ = (p_C_L, p_C_H, p_G_L, p_G_H, N_C, N_G)

        if T < params.T_bar:
            T_next_list = [(T, 1.0 - params.phi), (T + 1, params.phi)]
        else:
            T_next_list = [(params.T_bar, 1.0)]

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
# NPL / VF under forced A_H
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
    """Value-function iteration warm-started from W_init. Returns (W, CCP)."""
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


def run_npl_forced_AH(
    ss, params, N_C, N_G, k_t: int, tau_fixed: int,
    P_C_init=None, P_G_init=None,
    W_C_init: Optional[np.ndarray] = None,
    W_G_init: Optional[np.ndarray] = None,
    max_iter: int = NPL_MAX_ITER,
    tol: float = NPL_TOL,
    damping: float = NPL_DAMPING,
    max_vf_iter: int = RESIDUAL_VF_ITER,
    vf_tol: float = 1e-8,
    verbose: bool = False,
) -> dict:
    """
    Symmetric dampened NPL with forced-A_H payoffs and forced-A_H transitions.
    Warm-starts from P_*_init / W_*_init when provided.
    """
    n = ss.n_states
    P_C = np.ones((n, 3)) / 3.0 if P_C_init is None else P_C_init.copy()
    P_G = np.ones((n, 3)) / 3.0 if P_G_init is None else P_G_init.copy()
    u_C, u_G = compute_payoffs_forced_AH(ss, params, k_t, tau_fixed)
    history: List[float] = []
    converged = False
    W_C = None if W_C_init is None else W_C_init.copy()
    W_G = None if W_G_init is None else W_G_init.copy()
    k_iter = -1

    for k_iter in range(max_iter):
        F_0, F_L, F_H = build_sparse_transitions_forced_AH(
            ss, params, P_C, P_G, N_C, N_G, k_t, tau_fixed,
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
        if verbose and (k_iter % 10 == 0 or diff < tol):
            ac = (P_C_new[:, 1] + P_C_new[:, 2]).mean()
            ag = (P_G_new[:, 1] + P_G_new[:, 2]).mean()
            print(
                f"  NPL-forced-A_H {k_iter:3d}: dP={diff:.2e}  "
                f"adopt(C/G)={ac:.3f}/{ag:.3f}  k={k_t}"
            )
        P_C, P_G = P_C_new, P_G_new
        if diff < tol:
            converged = True
            break

    return {
        "P_C": P_C, "P_G": P_G, "W_C": W_C, "W_G": W_G,
        "u_C": u_C, "u_G": u_G,
        "converged": converged, "n_iter": k_iter + 1, "history": history,
    }


def init_W_from_ccp(
    ss, params, P_C, P_G, N_C, N_G, u_C, u_G, k_t: int, tau_fixed: int,
    max_vf_iter: int = RESIDUAL_VF_ITER, vf_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ex-ante values consistent with an initial CCP under forced A_H."""
    F_0, F_L, F_H = build_sparse_transitions_forced_AH(
        ss, params, P_C, P_G, N_C, N_G, k_t, tau_fixed,
    )
    W_C, _ = solve_vf_warm(u_C, F_0, F_L, F_H, params.beta, None, max_vf_iter, vf_tol)
    W_G, _ = solve_vf_warm(u_G, F_0, F_L, F_H, params.beta, None, max_vf_iter, vf_tol)
    return W_C, W_G


# ---------------------------------------------------------------------------
# Sequential forward path
# ---------------------------------------------------------------------------

def run_sequential_path(
    ss,
    params,
    N_C: int,
    N_G: int,
    P_C_init: np.ndarray,
    P_G_init: np.ndarray,
    schedule: List[Tuple[int, int]],
    tau_fixed: int,
    n_br: int = DEFAULT_N_BR,
    damping: float = NPL_DAMPING,
    tol: float = NPL_TOL,
    W_C0: Optional[np.ndarray] = None,
    W_G0: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Sequential forward path along k(t) with linked CCP warm-starts.

    t=0: keep direct-verify P^L as own CCP; rates at tracked s_0.
    t>0: dampened forced-A_H NPL warm-started from CCP_{t-1} (and W).
    """
    P_C = P_C_init.copy()
    P_G = P_G_init.copy()
    k0 = int(schedule[0][1]) if schedule else 0
    u_C0, u_G0 = compute_payoffs_forced_AH(ss, params, k0, tau_fixed)
    if W_C0 is not None and W_G0 is not None:
        W_C, W_G = W_C0.copy(), W_G0.copy()
    else:
        W_C, W_G = init_W_from_ccp(
            ss, params, P_C, P_G, N_C, N_G, u_C0, u_G0, k0, tau_fixed,
        )

    rows: List[Dict[str, Any]] = []

    for t, k_t in schedule:
        s_t = tracked_state(k_t, tau_fixed)
        if t == 0:
            diffs = [0.0]
            n_iters = 0
            ccp_source = "PL_direct_verify"
            rates = ccp_at_state(P_C, P_G, ss, s_t)
            if rates is None:
                rates = overall_ccp_rates(P_C, P_G)
                rates["rate_mode"] = "overall_fallback"
        else:
            result = run_npl_forced_AH(
                ss, params, N_C, N_G, k_t, tau_fixed,
                P_C_init=P_C, P_G_init=P_G,
                W_C_init=W_C, W_G_init=W_G,
                max_iter=n_br, tol=tol, damping=damping,
            )
            P_C, P_G = result["P_C"], result["P_G"]
            W_C, W_G = result["W_C"], result["W_G"]
            diffs = result["history"]
            n_iters = int(result["n_iter"])
            ccp_source = (
                "NPL_forced_AH_linked"
                if result["converged"]
                else "NPL_forced_AH_unconv"
            )
            rates = ccp_at_state(P_C, P_G, ss, s_t)
            if rates is None:
                rates = overall_ccp_rates(P_C, P_G)
                rates["rate_mode"] = "overall_fallback"

        overall = overall_ccp_rates(P_C, P_G)
        rows.append({
            "t": int(t),
            "k_t": int(k_t),
            "A_H": int(k_t),
            "A_L": 0,
            "tau_fixed": int(tau_fixed),
            "tracked_state": s_t,
            "agent_C": rates["agent_C"],
            "hitl_C": rates["hitl_C"],
            "agent_G": rates["agent_G"],
            "hitl_G": rates["hitl_G"],
            "none_C": rates.get("none_C", np.nan),
            "none_G": rates.get("none_G", np.nan),
            "rate_mode": rates.get("rate_mode", "?"),
            "overall_agent_C": overall["agent_C"],
            "overall_hitl_C": overall["hitl_C"],
            "overall_agent_G": overall["agent_G"],
            "overall_hitl_G": overall["hitl_G"],
            "dP": float(diffs[-1]) if diffs else np.nan,
            "n_br": int(n_iters),
            "ccp_source": ccp_source,
            "converged": bool(t == 0 or "unconv" not in ccp_source),
        })

    return {
        "path_rows": rows,
        "P_C_final": P_C,
        "P_G_final": P_G,
        "W_C_final": W_C,
        "W_G_final": W_G,
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
    """t=0: P^L direct verify; t>0: forced-A_H NPL linked from CCP_{t-1}."""
    params = build_params_for_delta(delta)
    tau_fixed = int(params.tau_bar)
    schedule = k_schedule(params.N, t_max=t_max)
    u_C, u_G = compute_payoffs(ss, params)  # natural payoffs for P^L verify

    log_lines.append(f"[ delta = {delta} ]")
    log_lines.append(
        f"  Parameters: psi={PSI_FIXED}, N={params.N}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}, "
        f"tau_bar={tau_fixed}, alpha={params.alpha}"
    )
    log_lines.append(
        f"  k/A_H schedule: k(t)=min(t, N-1); "
        f"path = {[(t, k) for t, k in schedule]}"
    )
    log_lines.append(
        f"  Tracked firm state: s_t=(x={TRACKED_X}, A_L={TRACKED_A_L}, "
        f"A_H=k(t), tau={tau_fixed}, T={TRACKED_T})"
    )
    log_lines.append(
        "  Method: EVERY t forces (A_L,A_H)=(0,k(t)) and τ=τ̄ on ALL states "
        "(payoffs A_eff=α·k(t), g_τ=1; transitions: mf at (x,0,k(t),τ̄,T), "
        "next (A_L,A_H,τ) pinned). "
        "t=0 = Prop1-style P^L direct verify "
        "(verify_low_adoption_eq + refine_low_eq_on_reachable; NOT NPL high eq). "
        f"t>0 = dampened forced-A_H NPL (max_iter={n_br}, tol={tol}, "
        f"damping={damping}) warm-started from CCP_{{t-1}} (+W). "
        "Rates: tracked-firm CCP agent/HITL (C/G) at s_t."
    )
    log_lines.append(
        "  Prop2 link: classic Prop2 shocks CCP when A_L>=k toward HITL; "
        "here k(t) is a time path that forces competitor agent count A_H=k(t) "
        "(agent-adopting firms among others), with A_L=0 and τ fixed."
    )

    pl_profile, _reach_mask, pl_ok = get_pl_profile(
        ss, params, N_C, N_G, u_C, u_G, log_lines,
    )
    if not pl_ok:
        log_lines.append("  Forward sim SKIPPED (no P^L from direct verification).")
        log_lines.append("")
        return {
            "delta": delta,
            "pl_exists": False,
            "path_rows": [],
        }

    P_C0, P_G0 = pl_profile["P_C"], pl_profile["P_G"]
    s0 = tracked_state(schedule[0][1], tau_fixed)
    tracked0 = ccp_at_state(P_C0, P_G0, ss, s0)
    overall0 = overall_ccp_rates(P_C0, P_G0)
    if tracked0 is not None:
        log_lines.append(
            f"  P^L tracked-firm CCP at {s0}: "
            f"agent_C={tracked0['agent_C']:.3f}, HITL_C={tracked0['hitl_C']:.3f}, "
            f"agent_G={tracked0['agent_G']:.3f}, HITL_G={tracked0['hitl_G']:.3f}"
        )
    log_lines.append(
        f"  P^L overall CCP means (diagnostic): "
        f"agent_C={overall0['agent_C']:.3f}, HITL_C={overall0['hitl_C']:.3f}, "
        f"agent_G={overall0['agent_G']:.3f}, HITL_G={overall0['hitl_G']:.3f}"
    )

    # Seed W under forced k=0 environment for smooth t=1 warm-start
    u_f0, u_g0 = compute_payoffs_forced_AH(ss, params, schedule[0][1], tau_fixed)
    W_C0, W_G0 = init_W_from_ccp(
        ss, params, P_C0, P_G0, N_C, N_G, u_f0, u_g0,
        schedule[0][1], tau_fixed,
    )

    path_out = run_sequential_path(
        ss, params, N_C, N_G,
        P_C0, P_G0, schedule, tau_fixed,
        n_br=n_br, damping=damping, tol=tol,
        W_C0=W_C0, W_G0=W_G0,
    )
    path_rows = path_out["path_rows"]

    log_lines.append("")
    log_lines.append(
        "  Forward path rates (t=0=P^L direct verify at tracked s_t; "
        "t>0=forced-A_H NPL linked from CCP_{t-1}; rates=tracked-firm CCP):"
    )
    log_lines.append(
        f"  {'t':>3}  {'k_t':>4}  {'A_H':>4}  "
        f"{'agent_C':>8}  {'HITL_C':>8}  {'agent_G':>8}  {'HITL_G':>8}  "
        f"{'rate_mode':>28}  {'dP':>8}  {'n_iter':>6}  {'src':>22}"
    )
    log_lines.append("  " + "-" * 130)
    for r in path_rows:
        log_lines.append(
            f"  {r['t']:>3d}  {r['k_t']:>4d}  {r['A_H']:>4d}  "
            f"{r['agent_C']:>8.3f}  {r['hitl_C']:>8.3f}  "
            f"{r['agent_G']:>8.3f}  {r['hitl_G']:>8.3f}  "
            f"{str(r.get('rate_mode', '?')):>28}  "
            f"{r['dP']:>8.2e}  {int(r.get('n_br', 0)):>6d}  "
            f"{r.get('ccp_source', '?'):>22}"
        )

    # Cascade-style jump from t=0 tracked adopt
    adopt0 = 0.5 * (
        path_rows[0]["agent_C"] + path_rows[0]["hitl_C"]
        + path_rows[0]["agent_G"] + path_rows[0]["hitl_G"]
    ) if path_rows else np.nan
    adopt_end = 0.5 * (
        path_rows[-1]["agent_C"] + path_rows[-1]["hitl_C"]
        + path_rows[-1]["agent_G"] + path_rows[-1]["hitl_G"]
    ) if path_rows else np.nan
    jump = adopt_end - adopt0 if path_rows else np.nan
    log_lines.append("")
    log_lines.append(
        f"  Tracked adopt (mean C/G HITL+agent): t0={adopt0:.3f}, "
        f"t_end={adopt_end:.3f}, jump={jump:.3f}"
    )
    log_lines.append("")

    print(
        f"[ delta = {delta} ]  sequential forward done — "
        f"n_br_max={n_br} (t>0), T_end={schedule[-1][0]}, "
        f"k_end={schedule[-1][1]}, jump={jump:.3f}",
        flush=True,
    )

    return {
        "delta": delta,
        "pl_exists": True,
        "path_rows": path_rows,
        "pl_agent_C": pl_profile["stats"]["agent_C"],
        "pl_agent_G": pl_profile["stats"]["agent_G"],
        "adopt0": adopt0,
        "adopt_end": adopt_end,
        "jump": jump,
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
    N = int(BASE_PARAMS["N"])
    tau_bar = int(BASE_PARAMS["tau_bar"])
    sched_default = k_schedule(N, t_max=t_max)

    log_lines.append(
        "PROPOSITION 2 FORWARD SIM: forced A_H=k(t) all states; "
        "t=0 P^L direct verify + t>0 NPL linked from CCP_{t-1}"
    )
    log_lines.append("=" * 105)
    log_lines.append(
        f"Fixed parameters: psi={PSI_FIXED}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}"
    )
    log_lines.append(
        f"State space (DEV_MODE={DEV_MODE}): "
        f"N={N}, x_bar={BASE_PARAMS['x_bar']}, "
        f"T_bar={BASE_PARAMS['T_bar']}, tau_bar={tau_bar}, "
        f"n_states={ss.n_states}"
    )
    log_lines.append(
        f"k/A_H schedule formula: k(t)=min(t, N-1) with N={N}; "
        f"default path = {sched_default}"
    )
    log_lines.append(
        "Forced adoption counts: at every t, ALL states use (A_L,A_H)=(0,k(t)) "
        "for flow payoffs (A_eff=α·k(t)) and transitions (mf CCP at "
        "(x,0,k(t),τ̄,T); next (A_L,A_H,τ) pinned). τ fixed at τ̄. "
        "Only (x, T) vary endogenously within the period VF."
    )
    log_lines.append(
        f"Tracked firm (same firm / rising A_H): "
        f"s_t=({TRACKED_X}, {TRACKED_A_L}, k(t), {tau_bar}, {TRACKED_T}). "
        "Rates = CCP at that state (C/G agent & HITL), not overall means."
    )
    log_lines.append(
        "Method (PRIMARY): t=0 = Prop1-style P^L via verify_low_adoption_eq + "
        "refine_low_eq_on_reachable (same as Prop2 baseline; NOT random NPL "
        f"high eq). t>0 = dampened forced-A_H NPL (max_iter={n_br}, tol={tol}, "
        "damping={damping}) warm-started from CCP_{t-1} (+W if available). "
        "DELTA_VALUES match Prop2 "
        f"{list(DELTA_VALUES)} (use --delta to subset; 4.0/5.0 optional/slower)."
    )
    log_lines.append(
        "Honesty: sequential NPL along an exogenous k(t)/A_H path — not a full "
        "closed-loop RE path (agents do not anticipate the entire future k "
        "schedule inside the continuation value). Classic Prop2 applies a "
        "one-shot CCP shock on A_L>=k; this forward sim turns k into a calendar "
        "path that forces competitor agent count A_H."
    )
    log_lines.append(
        f"NPL settings (t>0): max_iter={n_br}, tol={tol}, damping={damping}"
    )
    log_lines.append(
        f"Horizon: t_max={'N-1 (t=0..N-1)' if t_max is None else t_max}"
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
    log_lines.append("TABLE: Proposition 2 Forward-Sim Summary")
    log_lines.append("-" * 105)
    header = (
        f"{'delta':<7}| {'P^L?':<6}| {'adopt0':<8}| {'adopt_end':<10}| "
        f"{'jump':<8}| baseline agent (C/G)"
    )
    log_lines.append(header)
    log_lines.append("-" * 105)
    for row in summary_rows:
        if row.get("pl_exists"):
            log_lines.append(
                f"{row['delta']:<7.1f}| "
                f"{str(row['pl_exists']):<6}| "
                f"{row['adopt0']:<8.3f}| "
                f"{row['adopt_end']:<10.3f}| "
                f"{row['jump']:<8.3f}| "
                f"{row['pl_agent_C']:.3f}/{row['pl_agent_G']:.3f}"
            )
        else:
            log_lines.append(
                f"{row['delta']:<7.1f}| "
                f"{str(row['pl_exists']):<6}| "
                f"{'N/A':<8}| {'N/A':<10}| {'N/A':<8}| N/A"
            )

    return summary_rows


# ---------------------------------------------------------------------------
# Parse + plot (no NPL; reads OUTPUT_TXT)
# ---------------------------------------------------------------------------

def parse_forward_txt(text: str) -> Dict[str, Any]:
    """Parse prop2_forward_sim_verification.txt into blocks with path rows."""
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
                "Tracked adopt" in line
                or line.strip().startswith("Parameters")
            ):
                in_path = False
            if not in_path:
                continue
            parts = line.split()
            try:
                # Primary: t k_t A_H agent_C HITL_C agent_G HITL_G ...
                if len(parts) >= 7:
                    t = int(parts[0])
                    k_t = int(parts[1])
                    a_h = int(parts[2])
                    vals = list(map(float, parts[3:7]))
                    path_rows.append({
                        "t": t,
                        "k_t": k_t,
                        "A_H": a_h,
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
) -> List[Path]:
    """
    Plot C/G agent & HITL tracked-firm CCP rates vs t for each delta.

    Saves under prop2_forward_sim/figures/prop2_forward_*.
    """
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
        print("  [warn] Prop2 forward: no path blocks to plot")
        return []

    rate_lab = (
        r"tracked firm CCP; forced $A_H{=}k(t)$; "
        r"$t$>0 NPL linked from CCP$_{t-1}$"
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
        k_ann = ", ".join(f"t{r['t']}→k{r['k_t']}" for r in rows[:4])
        if len(rows) > 4:
            k_ann += ", …"
        ax.set_title(rf"$\delta$={blk['delta']}" + f"\n({k_ann})", fontsize=10)
        ax.set_xlabel(r"$t$")
        if i % ncols == 0:
            ax.set_ylabel("Tracked-firm CCP")
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        r"Prop 2 forward: C/G agent & HITL vs $t$"
        "\n"
        rf"({rate_lab}; $t$=0 $P^L$ direct verify; "
        r"$k(t)=\min(t,N-1)$)",
        y=1.04,
    )
    fig.tight_layout()
    stem = "prop2_forward_rates_vs_t"
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
        ax2.set_xticklabels([str(r["k_t"]) for r in rows])
        ax2.set_xlabel(r"$k_t = A_H$ (imposed)")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel("Tracked-firm CCP")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            rf"Prop 2 forward vs $t$ "
            rf"($\delta$={blk['delta']}; {rate_lab})"
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        d_tag = str(blk["delta"]).replace(".", "p")
        stem_d = f"prop2_forward_rates_vs_t_delta{d_tag}"
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
            "Prop2 forward simulation: at every t force A_H=k(t), A_L=0, τ=τ̄ "
            "on ALL states (payoffs + transitions). "
            "t=0 = P^L direct verify; "
            "t>0 = NPL linked from CCP_{t-1} along k(t)=min(t, N-1)."
        )
    )
    parser.add_argument(
        "--delta", type=float, nargs="*", default=None,
        help="Optional subset of delta values (default: Prop2 full grid).",
    )
    parser.add_argument(
        "--t-max", type=int, default=None,
        help="Max t inclusive (default: N-1, i.e. t=0..N-1).",
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
        "PROPOSITION 2: Forward Sim "
        "(forced A_H=k(t); t=0 P^L direct; t>0 linked NPL)"
    )
    print("=" * 70)
    print(
        f"Parameters: psi={PSI_FIXED}, "
        f"delta_x_base={DELTA_X_BASE}, delta_x_comp={DELTA_X_COMP}"
    )
    print(f"Delta sweep: {delta_values}")
    print(
        f"DEV_MODE={DEV_MODE}  "
        f"(N={BASE_PARAMS['N']}, x_bar={BASE_PARAMS['x_bar']}, "
        f"tau_bar={BASE_PARAMS['tau_bar']})"
    )
    print(f"Schedule: k(t)=min(t, N-1)  →  A_H forced, A_L=0, τ=τ̄ fixed")
    print(
        f"Tracked firm: ({TRACKED_X}, {TRACKED_A_L}, k(t), "
        f"{BASE_PARAMS['tau_bar']}, {TRACKED_T})"
    )
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
