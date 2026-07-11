"""
Verify each proposition in the paper through numerical simulation.

Run this script to produce all verification results and figures.
Usage: python verify_propositions.py [--prop N] [--all]
"""

import numpy as np
import argparse
import json
import os
import sys

from model import ModelParams, StateSpace, compute_payoffs, A_NONE, A_HITL, A_AGENT
from npl import (
    run_npl, find_all_equilibria, classify_equilibrium,
    generate_initial_profiles
)

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")
os.makedirs(os.path.join(OUT_DIR, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

# Use smaller state space for faster iteration during development.
# Increase for final results.
DEV_MODE = True
if DEV_MODE:
    BASE_PARAMS = dict(N=4, x_bar=5, T_bar=3, tau_bar=3)
    N_INITS = 20
else:
    BASE_PARAMS = dict(N=6, x_bar=9, T_bar=4, tau_bar=4)
    N_INITS = 50


def prop1_multiple_equilibria():
    """
    Proposition 1: Strategic Complementarity and Multiple Equilibria.

    Verify that for delta large enough, multiple distinct equilibria exist.
    Sweep delta and report number of equilibria found.
    """
    print("=" * 70)
    print("PROPOSITION 1: Multiple Equilibria")
    print("=" * 70)

    delta_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    results_table = []

    for delta in delta_values:
        print(f"\n--- delta = {delta} ---")
        params = ModelParams(**BASE_PARAMS, delta=delta, psi=0.5)
        ss = StateSpace(params)
        N_C, N_G = params.N // 2, params.N - params.N // 2

        eqs = find_all_equilibria(
            ss, params, N_C, N_G,
            n_inits=N_INITS, verbose=False
        )

        row = {
            "delta": delta,
            "n_equilibria": len(eqs),
            "types": [],
        }
        for eq in eqs:
            eq_type = classify_equilibrium(eq["stats"])
            row["types"].append(eq_type)
            print(f"  Equilibrium: {eq_type}")
            print(f"    C: adopt={eq['stats']['adopt_C']:.3f} "
                  f"(HITL={eq['stats']['hitl_C']:.3f}, "
                  f"agent={eq['stats']['agent_C']:.3f})")
            print(f"    G: adopt={eq['stats']['adopt_G']:.3f} "
                  f"(HITL={eq['stats']['hitl_G']:.3f}, "
                  f"agent={eq['stats']['agent_G']:.3f})")

        results_table.append(row)

    print("\n" + "=" * 70)
    print("SUMMARY: Number of distinct equilibria by delta")
    print("-" * 50)
    print(f"{'delta':>8}  {'# equil':>8}  {'types'}")
    print("-" * 50)
    for row in results_table:
        types_str = ", ".join(row["types"]) if row["types"] else "none found"
        print(f"{row['delta']:>8.1f}  {row['n_equilibria']:>8d}  {types_str}")

    return results_table


def prop2_cascade():
    """
    Proposition 2: Adoption Cascade Dynamics.

    Starting from the low-adoption equilibrium, exogenously force k firms
    to upgrade and check if remaining firms' best responses cascade.
    """
    print("\n" + "=" * 70)
    print("PROPOSITION 2: Adoption Cascade Dynamics")
    print("=" * 70)

    params = ModelParams(**BASE_PARAMS, delta=3.0, psi=0.5)
    ss = StateSpace(params)
    N_C, N_G = params.N // 2, params.N - params.N // 2

    # First find the low-adoption equilibrium
    P_low_init = np.zeros((ss.n_states, 3))
    P_low_init[:, A_NONE] = 0.9
    P_low_init[:, A_HITL] = 0.05
    P_low_init[:, A_AGENT] = 0.05

    result_low = run_npl(
        ss, params, N_C, N_G,
        P_C_init=P_low_init, P_G_init=P_low_init.copy(),
        verbose=False
    )

    if not result_low["converged"]:
        print("  WARNING: Low-adoption equilibrium did not converge")
        return

    baseline_adopt = (
        (result_low["P_C"][:, A_HITL] + result_low["P_C"][:, A_AGENT]).mean() +
        (result_low["P_G"][:, A_HITL] + result_low["P_G"][:, A_AGENT]).mean()
    ) / 2
    print(f"  Baseline low-adoption rate: {baseline_adopt:.3f}")

    # Simulate exogenous shocks: set A_L or A_H to k for initial state
    print("\n  Cascade test: forcing k competitors to upgrade (HITL)...")
    print(f"  {'k forced':>10}  {'adopt rate after':>18}  {'cascade?':>10}")
    print("  " + "-" * 45)

    for k in range(params.N):
        # Modify initial probs: bias toward states with A_L >= k
        P_C_shock = result_low["P_C"].copy()
        P_G_shock = result_low["P_G"].copy()

        # Simple approach: for states where A_L >= k, increase upgrade prob
        for idx in range(ss.n_states):
            x, A_L, A_H, tau, T = ss.unpack(idx)
            if A_L >= k:
                # Shift probability toward upgrading
                P_C_shock[idx] = [0.1, 0.7, 0.2]
                P_G_shock[idx] = [0.1, 0.3, 0.6]

        result_shock = run_npl(
            ss, params, N_C, N_G,
            P_C_init=P_C_shock, P_G_init=P_G_shock,
            verbose=False
        )

        if result_shock["converged"]:
            adopt_after = (
                (result_shock["P_C"][:, A_HITL] + result_shock["P_C"][:, A_AGENT]).mean() +
                (result_shock["P_G"][:, A_HITL] + result_shock["P_G"][:, A_AGENT]).mean()
            ) / 2
            cascaded = adopt_after > baseline_adopt + 0.2
            print(f"  {k:>10d}  {adopt_after:>18.3f}  "
                  f"{'YES' if cascaded else 'no':>10}")
        else:
            print(f"  {k:>10d}  {'(not converged)':>18}  {'?':>10}")


def prop3_cautious_coordination():
    """
    Proposition 3: Cautious Coordination Equilibrium.

    Sweep psi (trust damage) and show that for large psi, a P^M equilibrium
    emerges where firms adopt HITL but not agentic.
    """
    print("\n" + "=" * 70)
    print("PROPOSITION 3: Cautious Coordination Equilibrium")
    print("=" * 70)

    psi_values = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    results = []

    for psi in psi_values:
        print(f"\n--- psi = {psi} ---")
        params = ModelParams(**BASE_PARAMS, delta=3.0, psi=psi)
        ss = StateSpace(params)
        N_C, N_G = params.N // 2, params.N - params.N // 2

        eqs = find_all_equilibria(
            ss, params, N_C, N_G,
            n_inits=N_INITS, verbose=False
        )

        row = {"psi": psi, "equilibria": []}
        for eq in eqs:
            eq_type = classify_equilibrium(eq["stats"])
            row["equilibria"].append({
                "type": eq_type,
                "stats": eq["stats"]
            })
            print(f"  {eq_type}: HITL(C)={eq['stats']['hitl_C']:.3f} "
                  f"agent(C)={eq['stats']['agent_C']:.3f} "
                  f"HITL(G)={eq['stats']['hitl_G']:.3f} "
                  f"agent(G)={eq['stats']['agent_G']:.3f}")
        results.append(row)

    print("\n" + "=" * 70)
    print("SUMMARY: Equilibrium types by psi (trust damage intensity)")
    print("-" * 60)
    print(f"{'psi':>6}  {'# equil':>8}  {'types'}")
    print("-" * 60)
    for row in results:
        types_str = ", ".join([e["type"] for e in row["equilibria"]])
        print(f"{row['psi']:>6.1f}  {len(row['equilibria']):>8d}  {types_str}")

    return results


def prop4_cascade_reversal():
    """
    Proposition 4: Trust-Induced Cascade Reversal.

    Start from P^H (high agentic adoption), reduce trust exogenously,
    show that firms switch from H to L.
    """
    print("\n" + "=" * 70)
    print("PROPOSITION 4: Trust-Induced Cascade Reversal")
    print("=" * 70)

    params = ModelParams(**BASE_PARAMS, delta=3.0, psi=2.0)
    ss = StateSpace(params)
    N_C, N_G = params.N // 2, params.N - params.N // 2

    # Find high-agentic equilibrium
    P_H_init = np.zeros((ss.n_states, 3))
    P_H_init[:, A_NONE] = 0.05
    P_H_init[:, A_HITL] = 0.1
    P_H_init[:, A_AGENT] = 0.85

    result_high = run_npl(
        ss, params, N_C, N_G,
        P_C_init=P_H_init.copy(), P_G_init=P_H_init.copy(),
        verbose=False
    )

    if not result_high["converged"]:
        print("  WARNING: High-adoption equilibrium did not converge. "
              "Trying with different initialization...")
        result_high = run_npl(
            ss, params, N_C, N_G,
            verbose=False  # uniform init
        )

    print(f"  High-adoption equilibrium:")
    print(f"    C: agent={result_high['P_C'][:, A_AGENT].mean():.3f} "
          f"HITL={result_high['P_C'][:, A_HITL].mean():.3f}")
    print(f"    G: agent={result_high['P_G'][:, A_AGENT].mean():.3f} "
          f"HITL={result_high['P_G'][:, A_HITL].mean():.3f}")

    # Now test: at different trust levels, what's the equilibrium?
    print("\n  Trust level sweep (holding other states fixed):")
    print(f"  {'tau':>5}  {'agent(C)':>10}  {'HITL(C)':>10}  "
          f"{'agent(G)':>10}  {'HITL(G)':>10}")
    print("  " + "-" * 50)

    for tau_level in range(params.tau_bar + 1):
        # Look at choice probs at states with this trust level
        mask = ss.states[:, 3] == tau_level
        if mask.sum() == 0:
            continue
        agent_C = result_high["P_C"][mask, A_AGENT].mean()
        hitl_C = result_high["P_C"][mask, A_HITL].mean()
        agent_G = result_high["P_G"][mask, A_AGENT].mean()
        hitl_G = result_high["P_G"][mask, A_HITL].mean()
        print(f"  {tau_level:>5d}  {agent_C:>10.3f}  {hitl_C:>10.3f}  "
              f"{agent_G:>10.3f}  {hitl_G:>10.3f}")


def prop5_risk_type_sorting():
    """
    Proposition 5: Risk-Type Sorting in Asymmetric Equilibria.

    Show that C-types and G-types sort into different strategies.
    Sweep omega_C_E (conservative error cost) to show sorting intensity.
    """
    print("\n" + "=" * 70)
    print("PROPOSITION 5: Risk-Type Sorting")
    print("=" * 70)

    omega_C_values = [1.0, 2.0, 3.0, 5.0, 8.0]  # omega_G_E fixed at 1.0

    print(f"  {'omega_C_E':>10}  {'agent(C)':>10}  {'HITL(C)':>10}  "
          f"{'agent(G)':>10}  {'HITL(G)':>10}  {'sorting':>10}")
    print("  " + "-" * 65)

    for omega_C_E in omega_C_values:
        params = ModelParams(
            **BASE_PARAMS, delta=3.0, psi=1.5,
            omega_C_E=omega_C_E, omega_G_E=1.0
        )
        ss = StateSpace(params)
        N_C, N_G = params.N // 2, params.N - params.N // 2

        # Use asymmetric initialization
        P_C_init = np.zeros((ss.n_states, 3))
        P_C_init[:, A_NONE] = 0.1
        P_C_init[:, A_HITL] = 0.7
        P_C_init[:, A_AGENT] = 0.2
        P_G_init = np.zeros((ss.n_states, 3))
        P_G_init[:, A_NONE] = 0.1
        P_G_init[:, A_HITL] = 0.2
        P_G_init[:, A_AGENT] = 0.7

        result = run_npl(
            ss, params, N_C, N_G,
            P_C_init=P_C_init, P_G_init=P_G_init,
            verbose=False
        )

        if result["converged"]:
            ac = result["P_C"][:, A_AGENT].mean()
            hc = result["P_C"][:, A_HITL].mean()
            ag = result["P_G"][:, A_AGENT].mean()
            hg = result["P_G"][:, A_HITL].mean()
            sorting = ag - ac  # positive = G more agentic than C
            print(f"  {omega_C_E:>10.1f}  {ac:>10.3f}  {hc:>10.3f}  "
                  f"{ag:>10.3f}  {hg:>10.3f}  {sorting:>10.3f}")
        else:
            print(f"  {omega_C_E:>10.1f}  {'(not converged)':>45}")


def prop6_welfare():
    """
    Proposition 6: Welfare Ranking.

    For different parameter configurations, compute welfare under each
    equilibrium and verify the three regimes.
    """
    print("\n" + "=" * 70)
    print("PROPOSITION 6: Welfare Ranking Across Equilibria")
    print("=" * 70)

    # Test three regimes
    configs = [
        {"label": "(i) Low psi (P^H dominates)",
         "delta": 3.0, "psi": 0.3, "K_L": 4.0, "K_H": 3.0},
        {"label": "(ii) High psi (P^M dominates, over-automation)",
         "delta": 3.0, "psi": 5.0, "K_L": 4.0, "K_H": 3.0},
        {"label": "(iii) High K (P^L dominates, arms race)",
         "delta": 3.0, "psi": 1.0, "K_L": 10.0, "K_H": 9.0},
    ]

    for config in configs:
        label = config.pop("label")
        print(f"\n--- {label} ---")
        params = ModelParams(**BASE_PARAMS, **config)
        ss = StateSpace(params)
        N_C, N_G = params.N // 2, params.N - params.N // 2

        eqs = find_all_equilibria(
            ss, params, N_C, N_G,
            n_inits=N_INITS, verbose=False
        )

        if not eqs:
            print("  No equilibria found.")
            continue

        # Compute approximate welfare for each equilibrium
        # Welfare ~ average ex-ante value across states
        u_C, u_G = compute_payoffs(ss, params)

        for eq in eqs:
            eq_type = classify_equilibrium(eq["stats"])
            # Approximate welfare: avg deterministic payoff under equilibrium probs
            w_C = np.sum(eq["P_C"] * u_C, axis=1).mean()
            w_G = np.sum(eq["P_G"] * u_G, axis=1).mean()
            w_total = (N_C * w_C + N_G * w_G) / params.N

            print(f"  {eq_type:40s}  W(C)={w_C:>7.3f}  W(G)={w_G:>7.3f}  "
                  f"W_total={w_total:>7.3f}")


def prop7_asymmetric_homogeneous():
    """
    Proposition 7: Asymmetric Equilibria among Homogeneous Firms.

    Set all firms to the same type and show asymmetric equilibria
    still exist (some play H, others play L).
    """
    print("\n" + "=" * 70)
    print("PROPOSITION 7: Asymmetric Equilibria among Homogeneous Firms")
    print("=" * 70)

    # All firms are aggressive (same type)
    params = ModelParams(**BASE_PARAMS, delta=3.0, psi=2.0,
                         omega_C_E=2.0, omega_G_E=2.0,  # same
                         mu_C=1.0, mu_G=1.0)  # same
    ss = StateSpace(params)

    # All firms are one type (say all G, so N_C=0)
    # But our code requires N_C >= 1 for transition computation.
    # Workaround: set N_C = N, N_G = 0 and make C=G params identical.
    N_C, N_G = params.N, 0

    print("  All firms identical (omega_C_E = omega_G_E = 2.0, mu_C = mu_G = 1.0)")
    print("  Searching for asymmetric equilibria...\n")

    eqs = find_all_equilibria(
        ss, params, N_C, N_G,
        n_inits=N_INITS, verbose=False
    )

    if not eqs:
        print("  No equilibria found. Try adjusting parameters.")
        return

    # Check for asymmetry within type
    for i, eq in enumerate(eqs):
        eq_type = classify_equilibrium(eq["stats"])
        agent_C = eq["P_C"][:, A_AGENT].mean()
        hitl_C = eq["P_C"][:, A_HITL].mean()
        print(f"  Equilibrium {i+1}: {eq_type}")
        print(f"    adopt={eq['stats']['adopt_C']:.3f}  "
              f"HITL={hitl_C:.3f}  agent={agent_C:.3f}")

    print("\n  Note: true asymmetric equilibria among identical firms require")
    print("  breaking the symmetric-within-type assumption. The above shows")
    print("  that even with identical parameters, different initializations")
    print("  can converge to different equilibria, demonstrating multiplicity.")


def phase_diagram():
    """
    Generate phase diagram in (delta, psi) space showing which
    equilibrium types exist.
    """
    print("\n" + "=" * 70)
    print("PHASE DIAGRAM: Equilibrium types in (delta, psi) space")
    print("=" * 70)

    delta_grid = [1.0, 2.0, 3.0, 4.0, 5.0]
    psi_grid = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

    print(f"\n{'':>8}", end="")
    for psi in psi_grid:
        print(f"{'psi='+str(psi):>12}", end="")
    print()
    print("-" * (8 + 12 * len(psi_grid)))

    for delta in delta_grid:
        print(f"d={delta:<5.1f}", end="")
        for psi in psi_grid:
            params = ModelParams(**BASE_PARAMS, delta=delta, psi=psi)
            ss = StateSpace(params)
            N_C, N_G = params.N // 2, params.N - params.N // 2

            eqs = find_all_equilibria(
                ss, params, N_C, N_G,
                n_inits=10, verbose=False  # fewer inits for speed
            )

            types = set()
            for eq in eqs:
                t = classify_equilibrium(eq["stats"])
                if "low" in t.lower():
                    types.add("L")
                elif "cautious" in t.lower():
                    types.add("M")
                elif "aggressive" in t.lower() or "agentic" in t.lower():
                    types.add("H")
                elif "asym" in t.lower():
                    types.add("A")
                else:
                    types.add("?")

            label = ",".join(sorted(types)) if types else "-"
            print(f"{label:>12}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Verify propositions")
    parser.add_argument("--prop", type=int, default=None,
                        help="Run specific proposition (1-7)")
    parser.add_argument("--all", action="store_true",
                        help="Run all propositions")
    parser.add_argument("--phase", action="store_true",
                        help="Generate phase diagram")
    args = parser.parse_args()

    if args.prop == 1 or args.all:
        prop1_multiple_equilibria()
    if args.prop == 2 or args.all:
        prop2_cascade()
    if args.prop == 3 or args.all:
        prop3_cautious_coordination()
    if args.prop == 4 or args.all:
        prop4_cascade_reversal()
    if args.prop == 5 or args.all:
        prop5_risk_type_sorting()
    if args.prop == 6 or args.all:
        prop6_welfare()
    if args.prop == 7 or args.all:
        prop7_asymmetric_homogeneous()
    if args.phase:
        phase_diagram()

    if not args.prop and not args.all and not args.phase:
        print("Usage: python verify_propositions.py --prop N  (1-7)")
        print("       python verify_propositions.py --all")
        print("       python verify_propositions.py --phase")
        print("\nRunning quick sanity check (Prop 1 only)...")
        prop1_multiple_equilibria()


if __name__ == "__main__":
    main()
