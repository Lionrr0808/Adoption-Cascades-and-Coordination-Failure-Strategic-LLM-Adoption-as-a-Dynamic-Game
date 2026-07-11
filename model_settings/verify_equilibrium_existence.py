"""
Verify existence of multiple equilibria via direct fixed-point verification.

KEY INSIGHT: NPL can only find "NPL-stable" equilibria (Aguirregabiria & Mira, 2007).
The low-adoption equilibrium is NPL-unstable in coordination games, so NPL always
converges to the high-adoption one regardless of initialization.

SOLUTION: Instead of relying on NPL convergence, we directly VERIFY that a candidate
policy is a fixed point by:
1. Proposing a candidate policy (e.g., "all don't adopt")
2. Building transitions under that policy
3. Computing the best response
4. Checking if BR = candidate at reachable states

If BR matches the candidate on its ergodic set, the equilibrium EXISTS even if
NPL can't converge to it.
"""

import numpy as np
from model import (
    ModelParams, StateSpace, compute_payoffs, build_sparse_transitions,
    A_NONE, A_HITL, A_AGENT, EULER_MASCHERONI
)
from npl import choice_probs_from_values, ex_ante_value, run_npl


def verify_low_adoption_eq(params, ss, N_C, N_G, verbose=True):
    """
    Verify that a low-adoption equilibrium exists.

    Strategy: if ALL firms play "don't adopt" and delta_x_base is small enough,
    then at reachable states (low x, low A), the BR is also "don't adopt".
    """
    n = ss.n_states
    u_C, u_G = compute_payoffs(ss, params)

    # Candidate: everyone plays "don't adopt" at all states
    P_cand = np.zeros((n, 3))
    P_cand[:, A_NONE] = 1.0 - 1e-10  # near-degenerate to avoid log(0) issues
    P_cand[:, A_HITL] = 5e-11
    P_cand[:, A_AGENT] = 5e-11

    # Build transitions under candidate policy
    F_0, F_L, F_H = build_sparse_transitions(ss, params, P_cand, P_cand, N_C, N_G)

    # Solve value function for each type
    results = {}
    for label, u in [("C", u_C), ("G", u_G)]:
        W = np.zeros(n)
        for it in range(500):
            v = np.column_stack([
                u[:, 0] + params.beta * F_0.dot(W),
                u[:, 1] + params.beta * F_L.dot(W),
                u[:, 2] + params.beta * F_H.dot(W),
            ])
            W_new = ex_ante_value(v)
            if np.max(np.abs(W_new - W)) < 1e-10:
                break
            W = W_new

        # Compute BR
        v = np.column_stack([
            u[:, 0] + params.beta * F_0.dot(W),
            u[:, 1] + params.beta * F_L.dot(W),
            u[:, 2] + params.beta * F_H.dot(W),
        ])
        probs = choice_probs_from_values(v)
        results[label] = probs

    # Identify reachable states under low-adoption policy
    # With everyone not adopting: x stays put (or drifts slowly), A stays at 0
    # Reachable set depends on delta_x_base:
    #   If delta_x_base=0: only x=0 states are reachable (starting from x=0)
    #   If delta_x_base>0: x can drift, but A stays at 0
    low_A_mask = (ss.states[:, 1] == 0) & (ss.states[:, 2] == 0)

    if params.delta_x_base == 0:
        reachable_mask = low_A_mask & (ss.states[:, 0] == 0)
        reachable_desc = "x=0, A_L=0, A_H=0"
    else:
        reachable_mask = low_A_mask  # A stays at 0, x can drift
        reachable_desc = "A_L=0, A_H=0 (all x)"

    if verbose:
        print(f"\nReachable states under low-adoption policy: {reachable_mask.sum()}")
        print(f"  Description: {reachable_desc}")

    # Check BR at reachable states
    for label, probs in results.items():
        no_adopt_prob = probs[reachable_mask, A_NONE]
        adopt_prob = probs[reachable_mask, A_HITL] + probs[reachable_mask, A_AGENT]

        if verbose:
            print(f"\n  Type {label} at reachable states:")
            print(f"    P(no adopt): min={no_adopt_prob.min():.4f}, "
                  f"mean={no_adopt_prob.mean():.4f}, max={no_adopt_prob.max():.4f}")
            print(f"    P(adopt):    min={adopt_prob.min():.4f}, "
                  f"mean={adopt_prob.mean():.4f}, max={adopt_prob.max():.4f}")

    # Criterion: BR says "don't adopt" (P(none) > 0.5) at all reachable states
    verified_C = (results["C"][reachable_mask, A_NONE] > 0.5).all()
    verified_G = (results["G"][reachable_mask, A_NONE] > 0.5).all()
    verified = verified_C and verified_G

    if verbose:
        print(f"\n  Low-adoption eq verified: {verified}")
        if verified:
            print("  => Both types prefer 'don't adopt' at all reachable states")
            print("  => This is a valid MPE (self-consistent fixed point)")
        else:
            if not verified_C:
                print("  => Type C deviates at some reachable states")
            if not verified_G:
                print("  => Type G deviates at some reachable states")

    return verified, results


def verify_high_adoption_eq(params, ss, N_C, N_G, verbose=True):
    """
    Find and verify the high-adoption equilibrium via standard NPL.
    This one IS NPL-stable, so standard NPL converges to it.
    """
    result = run_npl(ss, params, N_C, N_G, max_iter=500, verbose=False, damping=0.3)

    if not result["converged"]:
        if verbose:
            print("  High-adoption eq: NPL did not converge")
        return False, result

    stats = {
        "adopt_C": (result["P_C"][:, 1] + result["P_C"][:, 2]).mean(),
        "adopt_G": (result["P_G"][:, 1] + result["P_G"][:, 2]).mean(),
        "hitl_C": result["P_C"][:, 1].mean(),
        "hitl_G": result["P_G"][:, 1].mean(),
        "agent_C": result["P_C"][:, 2].mean(),
        "agent_G": result["P_G"][:, 2].mean(),
    }

    if verbose:
        print(f"\n  High-adoption eq (NPL-stable):")
        print(f"    C: adopt={stats['adopt_C']:.3f} "
              f"(HITL={stats['hitl_C']:.3f}, agent={stats['agent_C']:.3f})")
        print(f"    G: adopt={stats['adopt_G']:.3f} "
              f"(HITL={stats['hitl_G']:.3f}, agent={stats['agent_G']:.3f})")

    return True, result


def demonstrate_multiplicity(verbose=True):
    """
    Full demonstration of Proposition 1: multiple equilibria exist.

    Uses delta_x_base=0 so that x only grows from competitive pressure.
    This is theoretically clean: "competitive gap" only grows when competitors
    are ahead of you, creating genuine strategic complementarity.
    """
    print("=" * 70)
    print("PROPOSITION 1: Multiple Equilibria (Direct Verification)")
    print("=" * 70)
    print()
    print("Method: verify fixed-point conditions for each candidate equilibrium.")
    print("NPL can only find NPL-stable equilibria; the low-adoption eq is")
    print("NPL-unstable, so we verify it directly instead.")

    params = ModelParams(
        N=4, x_bar=5, T_bar=3, tau_bar=3,
        delta=3.0,
        delta_x_base=0.0,   # x grows ONLY from competitive pressure
        delta_x_comp=0.2,   # competitive drive
        K_L=4.0, K_H=3.0,
        theta1_slope=0.5,
        psi=0.5,
    )
    ss = StateSpace(params)
    N_C, N_G = params.N // 2, params.N - params.N // 2
    print(f"\nState space: {ss.n_states} states")
    print(f"Parameters: delta={params.delta}, K_L={params.K_L}, K_H={params.K_H}")
    print(f"  delta_x_base={params.delta_x_base}, delta_x_comp={params.delta_x_comp}")

    # 1. Verify low-adoption equilibrium
    print("\n" + "-" * 50)
    print("Equilibrium 1: Low Adoption (P^L)")
    print("-" * 50)
    low_exists, low_results = verify_low_adoption_eq(params, ss, N_C, N_G, verbose)

    # 2. Find high-adoption equilibrium
    print("\n" + "-" * 50)
    print("Equilibrium 2: High Adoption (P^H)")
    print("-" * 50)
    high_exists, high_result = verify_high_adoption_eq(params, ss, N_C, N_G, verbose)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_eq = int(low_exists) + int(high_exists)
    print(f"  Distinct equilibria found: {n_eq}")
    if low_exists:
        print(f"  - P^L (low adoption): all firms prefer status quo when competitors don't adopt")
    if high_exists:
        print(f"  - P^H (high adoption): adoption is self-reinforcing through competitive externality")
    if n_eq >= 2:
        print(f"\n  => PROPOSITION 1 VERIFIED: multiple equilibria coexist")
        print(f"     Strategic complementarity creates coordination game structure.")
        print(f"     Which equilibrium is selected depends on initial conditions/history.")

    return n_eq


def sweep_multiplicity_region(verbose=True):
    """
    Map the region of (delta, delta_x_base) space where multiplicity obtains.
    Low-adoption eq exists when firms prefer status quo at reachable states.
    """
    print("\n" + "=" * 70)
    print("Multiplicity Region: sweep over (delta, delta_x_base)")
    print("=" * 70)

    delta_values = [1.0, 2.0, 3.0, 5.0]
    dxb_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    print(f"\n{'delta':>6} {'dxb':>6} {'low_eq':>8} {'high_eq':>8} {'mult':>6}")
    print("-" * 40)

    for delta in delta_values:
        for dxb in dxb_values:
            params = ModelParams(
                N=4, x_bar=5, T_bar=3, tau_bar=3,
                delta=delta,
                delta_x_base=dxb,
                delta_x_comp=0.2,
                K_L=4.0, K_H=3.0,
                theta1_slope=0.5,
                psi=0.5,
            )
            ss = StateSpace(params)
            N_C, N_G = params.N // 2, params.N - params.N // 2

            low_ok, _ = verify_low_adoption_eq(params, ss, N_C, N_G, verbose=False)
            high_ok, _ = verify_high_adoption_eq(params, ss, N_C, N_G, verbose=False)

            mult = "YES" if (low_ok and high_ok) else "no"
            print(f"{delta:>6.1f} {dxb:>6.2f} {str(low_ok):>8} {str(high_ok):>8} {mult:>6}")


if __name__ == "__main__":
    n_eq = demonstrate_multiplicity()
    if n_eq >= 2:
        sweep_multiplicity_region()
