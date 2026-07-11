"""
Core model: state space, payoffs, transitions for the LLM adoption dynamic game.

Performance: precomputes transition structure once per NPL iteration,
then VF iteration uses vectorized numpy dot products.
"""

import numpy as np
from scipy import sparse
from dataclasses import dataclass, field
from typing import Tuple

A_NONE = 0
A_HITL = 1
A_AGENT = 2
EULER_MASCHERONI = 0.5772156649


@dataclass
class ModelParams:
    N: int = 6
    beta: float = 0.95
    x_bar: int = 9
    T_bar: int = 4
    tau_bar: int = 4
    x_underbar: int = 2
    phi: float = 0.3

    delta: float = 2.0
    alpha: float = 2.0

    theta1_slope: float = 0.5
    theta1_intercept: float = 0.0
    K_L: float = 4.0
    K_H: float = 3.0

    psi: float = 1.0
    r: float = 0.2
    p_fail_base: float = 0.3
    p_fail_decay: float = 0.05

    omega_C_E: float = 3.0
    omega_G_E: float = 1.0
    mu_C: float = 2.0
    mu_G: float = 0.5
    lam: float = 1.5

    delta_x_base: float = 0.7
    delta_x_comp: float = 0.05

    def p_fail(self, T: int) -> float:
        return max(0.01, self.p_fail_base - self.p_fail_decay * T)

    def C(self, x) -> float:
        return self.theta1_intercept + self.theta1_slope * x

    def xi(self, T: int) -> float:
        return self.p_fail(T) * 5.0


@dataclass
class StateSpace:
    params: ModelParams
    states: np.ndarray = field(init=False)
    state_index: dict = field(init=False)
    n_states: int = field(init=False)
    n_actions: int = 3

    def __post_init__(self):
        p = self.params
        states_list = []
        index = {}
        idx = 0
        for x in range(p.x_bar + 1):
            for A_L in range(p.N):
                for A_H in range(p.N):
                    if A_L + A_H > p.N - 1:
                        continue
                    for tau in range(p.tau_bar + 1):
                        for T in range(p.T_bar + 1):
                            s = (x, A_L, A_H, tau, T)
                            states_list.append(s)
                            index[s] = idx
                            idx += 1
        self.states = np.array(states_list, dtype=np.int32)
        self.state_index = index
        self.n_states = len(states_list)


def compute_payoffs(ss: StateSpace, params: ModelParams) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized payoff computation. Returns u_C, u_G: (n_states, 3)."""
    s = ss.states
    x = s[:, 0].astype(float)
    A_L = s[:, 1].astype(float)
    A_H = s[:, 2].astype(float)
    tau = s[:, 3].astype(float)
    T = s[:, 4]

    A_eff = A_L + params.alpha * A_H
    g_tau = tau / params.tau_bar if params.tau_bar > 0 else np.zeros_like(tau)
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


def build_sparse_transitions(
    ss: StateSpace, params: ModelParams,
    P_C: np.ndarray, P_G: np.ndarray,
    N_C: int, N_G: int
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build sparse transition matrices F_0, F_L, F_H.
    Called once per NPL iteration, then reused for all VF iterations.
    """
    n = ss.n_states
    rows = [[] for _ in range(3)]
    cols = [[] for _ in range(3)]
    vals = [[] for _ in range(3)]

    for idx in range(n):
        x, A_L, A_H, tau, T = tuple(ss.states[idx])
        A_eff = A_L + params.alpha * A_H

        # Competitor choice probs (mean-field)
        p_C_L = P_C[idx, A_HITL]
        p_C_H = P_C[idx, A_AGENT]
        p_G_L = P_G[idx, A_HITL]
        p_G_H = P_G[idx, A_AGENT]

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

        # T transition
        T_next_list = []
        if T < params.T_bar:
            T_next_list = [(T, 1.0 - params.phi), (T + 1, params.phi)]
        else:
            T_next_list = [(params.T_bar, 1.0)]

        for a_i in range(3):
            # x transition
            if a_i == A_NONE:
                p_up = min(0.95, params.delta_x_base + params.delta_x_comp * A_eff)
                if x < params.x_bar:
                    x_list = [(x, 1.0 - p_up), (x + 1, p_up)]
                else:
                    x_list = [(params.x_bar, 1.0)]
            else:
                x_list = [(0, 1.0)]

            # tau transition
            A_H_trust = E_AH + (1.0 if a_i == A_AGENT else 0.0)
            E_F = A_H_trust * params.p_fail(T)
            recovery = params.r * (params.tau_bar - tau)
            damage = params.psi * E_F
            tau_cont = max(0.0, min(float(params.tau_bar), tau + recovery - damage))
            tau_lo = int(np.floor(tau_cont))
            tau_hi = min(tau_lo + 1, params.tau_bar)
            if tau_lo == tau_hi:
                tau_list = [(tau_lo, 1.0)]
            else:
                w_tau = tau_cont - tau_lo
                tau_list = [(tau_lo, 1.0 - w_tau), (tau_hi, w_tau)]

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
            shape=(n, n)
        )
        # Normalize rows
        row_sums = np.array(mat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        inv_sums = sparse.diags(1.0 / row_sums)
        mat = inv_sums @ mat
        F.append(mat)

    return F[0], F[1], F[2]


if __name__ == "__main__":
    params = ModelParams(N=4, x_bar=5, T_bar=3, tau_bar=3)
    ss = StateSpace(params)
    print(f"Number of states: {ss.n_states}")

    u_C, u_G = compute_payoffs(ss, params)
    print(f"Payoffs OK. Shape: {u_C.shape}")

    P = np.ones((ss.n_states, 3)) / 3.0
    print("Building sparse transitions...")
    F_0, F_L, F_H = build_sparse_transitions(ss, params, P, P, 2, 2)
    print(f"F_0: {F_0.shape}, nnz={F_0.nnz}")
    print(f"F_L: {F_L.shape}, nnz={F_L.nnz}")
    print(f"F_H: {F_H.shape}, nnz={F_H.nnz}")

    # Test VF iteration
    W = np.zeros(ss.n_states)
    for it in range(50):
        v = np.column_stack([
            u_C[:, 0] + params.beta * F_0.dot(W),
            u_C[:, 1] + params.beta * F_L.dot(W),
            u_C[:, 2] + params.beta * F_H.dot(W),
        ])
        v_max = v.max(axis=1)
        W_new = v_max + np.log(np.sum(np.exp(v - v_max[:, None]), axis=1)) + EULER_MASCHERONI
        diff = np.max(np.abs(W_new - W))
        W = W_new
        if it % 10 == 0:
            print(f"  VF iter {it}: diff={diff:.2e}")
        if diff < 1e-8:
            print(f"  VF converged at iter {it}")
            break

    print("Model + transitions OK.")
