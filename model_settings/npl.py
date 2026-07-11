"""
NPL algorithm using sparse transition matrices for speed.
"""

import numpy as np
from typing import Tuple, List, Optional
from model import (
    ModelParams, StateSpace, compute_payoffs, build_sparse_transitions,
    A_NONE, A_HITL, A_AGENT, EULER_MASCHERONI
)


def choice_probs_from_values(v: np.ndarray) -> np.ndarray:
    v_max = v.max(axis=1, keepdims=True)
    ev = np.exp(v - v_max)
    return ev / ev.sum(axis=1, keepdims=True)


def ex_ante_value(v: np.ndarray) -> np.ndarray:
    v_max = v.max(axis=1)
    return v_max + np.log(np.sum(np.exp(v - v_max[:, None]), axis=1)) + EULER_MASCHERONI


def npl_iteration(
    ss, params, P_C, P_G, N_C, N_G, u_C, u_G,
    max_vf_iter=200, vf_tol=1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """One NPL iteration using sparse matrices."""
    # Build transitions once (the expensive part, but only once per NPL iter)
    F_0, F_L, F_H = build_sparse_transitions(ss, params, P_C, P_G, N_C, N_G)

    results = []
    for u in [u_C, u_G]:
        W = np.zeros(ss.n_states)
        for it in range(max_vf_iter):
            v = np.column_stack([
                u[:, 0] + params.beta * F_0.dot(W),
                u[:, 1] + params.beta * F_L.dot(W),
                u[:, 2] + params.beta * F_H.dot(W),
            ])
            W_new = ex_ante_value(v)
            if np.max(np.abs(W_new - W)) < vf_tol:
                W = W_new
                break
            W = W_new

        v = np.column_stack([
            u[:, 0] + params.beta * F_0.dot(W),
            u[:, 1] + params.beta * F_L.dot(W),
            u[:, 2] + params.beta * F_H.dot(W),
        ])
        results.append(choice_probs_from_values(v))

    return results[0], results[1]


def run_npl(
    ss, params, N_C, N_G,
    P_C_init=None, P_G_init=None,
    max_iter=500, tol=1e-5, damping=0.5, verbose=True
) -> dict:
    """
    Run NPL with dampening: P_{k+1} = damping * BR(P_k) + (1-damping) * P_k.
    This stabilizes convergence near multiple equilibria.
    """
    n = ss.n_states
    P_C = np.ones((n, 3)) / 3.0 if P_C_init is None else P_C_init.copy()
    P_G = np.ones((n, 3)) / 3.0 if P_G_init is None else P_G_init.copy()
    u_C, u_G = compute_payoffs(ss, params)
    history = []
    converged = False

    for k in range(max_iter):
        P_C_br, P_G_br = npl_iteration(
            ss, params, P_C, P_G, N_C, N_G, u_C, u_G
        )
        # Dampened update
        P_C_new = damping * P_C_br + (1.0 - damping) * P_C
        P_G_new = damping * P_G_br + (1.0 - damping) * P_G
        diff = max(
            np.max(np.abs(P_C_new - P_C)),
            np.max(np.abs(P_G_new - P_G))
        )
        history.append(diff)

        if verbose and (k % 10 == 0 or diff < tol):
            ac = (P_C_new[:, 1] + P_C_new[:, 2]).mean()
            ag = (P_G_new[:, 1] + P_G_new[:, 2]).mean()
            hc = P_C_new[:, 1].mean()
            hg = P_G_new[:, 1].mean()
            print(f"  NPL {k:3d}: dP={diff:.2e}  "
                  f"adopt(C/G)={ac:.3f}/{ag:.3f}  "
                  f"hitl(C/G)={hc:.3f}/{hg:.3f}")

        P_C, P_G = P_C_new, P_G_new
        if diff < tol:
            converged = True
            if verbose:
                print(f"  Converged at iter {k}")
            break

    return {"P_C": P_C, "P_G": P_G, "converged": converged,
            "n_iter": k + 1, "history": history}


def generate_initial_profiles(n_states, n_inits=50, seed=42):
    rng = np.random.RandomState(seed)
    inits = []

    def make(p0, pL, pH):
        P = np.full((n_states, 3), [p0, pL, pH])
        return P

    inits.append((make(1/3, 1/3, 1/3), make(1/3, 1/3, 1/3), "uniform"))
    inits.append((make(0.9, 0.05, 0.05), make(0.9, 0.05, 0.05), "low_adopt"))
    inits.append((make(0.1, 0.8, 0.1), make(0.1, 0.8, 0.1), "high_HITL"))
    inits.append((make(0.05, 0.1, 0.85), make(0.05, 0.1, 0.85), "high_agent"))
    inits.append((make(0.1, 0.7, 0.2), make(0.1, 0.2, 0.7), "asym_CL_GH"))
    inits.append((make(0.1, 0.7, 0.2), make(0.8, 0.1, 0.1), "asym_CL_G0"))

    for i in range(n_inits - len(inits)):
        inits.append((
            rng.dirichlet([1, 1, 1], size=n_states),
            rng.dirichlet([1, 1, 1], size=n_states),
            f"rand_{i}"
        ))

    return inits[:n_inits]


def find_all_equilibria(
    ss, params, N_C, N_G,
    n_inits=50, tol=1e-6, distinct_tol=0.01,
    verbose=True, seed=42
) -> List[dict]:
    inits = generate_initial_profiles(ss.n_states, n_inits, seed)
    equilibria = []

    for i, (P_C_init, P_G_init, label) in enumerate(inits):
        if verbose:
            print(f"\n--- Init {i+1}/{len(inits)}: {label} ---")

        result = run_npl(
            ss, params, N_C, N_G,
            P_C_init=P_C_init, P_G_init=P_G_init,
            tol=tol, verbose=verbose
        )
        if not result["converged"]:
            continue

        is_new = True
        for eq in equilibria:
            d = max(np.max(np.abs(result["P_C"] - eq["P_C"])),
                    np.max(np.abs(result["P_G"] - eq["P_G"])))
            if d < distinct_tol:
                is_new = False
                break

        if is_new:
            stats = {
                "adopt_C": (result["P_C"][:, 1] + result["P_C"][:, 2]).mean(),
                "adopt_G": (result["P_G"][:, 1] + result["P_G"][:, 2]).mean(),
                "hitl_C": result["P_C"][:, 1].mean(),
                "hitl_G": result["P_G"][:, 1].mean(),
                "agent_C": result["P_C"][:, 2].mean(),
                "agent_G": result["P_G"][:, 2].mean(),
            }
            result["label"] = label
            result["stats"] = stats
            equilibria.append(result)
            if verbose:
                print(f"  >>> NEW eq #{len(equilibria)}: "
                      f"C={stats['adopt_C']:.2f}(H={stats['hitl_C']:.2f},A={stats['agent_C']:.2f}) "
                      f"G={stats['adopt_G']:.2f}(H={stats['hitl_G']:.2f},A={stats['agent_G']:.2f})")

    return equilibria


def classify_equilibrium(stats: dict) -> str:
    total_adopt = (stats["adopt_C"] + stats["adopt_G"]) / 2
    total_agent = (stats["agent_C"] + stats["agent_G"]) / 2
    total_hitl = (stats["hitl_C"] + stats["hitl_G"]) / 2

    if total_adopt < 0.3:
        return "P^L (low adoption)"
    elif total_agent > 0.4:
        return "P^H (aggressive)"
    elif total_hitl > 0.4 and total_agent < 0.2:
        return "P^M (cautious)"
    elif abs(stats["agent_C"] - stats["agent_G"]) > 0.2:
        return "Asymmetric"
    else:
        return "Other"


if __name__ == "__main__":
    import time
    params = ModelParams(N=4, x_bar=5, T_bar=3, tau_bar=3)
    ss = StateSpace(params)
    print(f"States: {ss.n_states}")

    t0 = time.time()
    result = run_npl(ss, params, 2, 2, max_iter=50, verbose=True)
    t1 = time.time()
    print(f"\nConverged: {result['converged']} in {t1-t0:.1f}s")
