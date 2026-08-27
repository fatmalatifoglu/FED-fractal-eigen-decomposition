#!/usr/bin/env python3
"""Quick check: Step 6 uses v = F^(1/2) y (not F^(-1/2) y)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from fed_cov_main import fed_cov_main


def main() -> None:
    print("=== FED main verification (Python) ===\n")
    rng = np.random.default_rng(42)

    # Test 1: algebra
    L = 6
    A = rng.standard_normal((L, L))
    C = (A @ A.T + A @ A.T) / 2.0
    F_diag = np.array([1.0, 2.5, 0.7, 3.2, 1.8, 0.4])
    Finv = np.diag(1.0 / F_diag)
    Fh = np.diag(np.sqrt(F_diag))

    lam_gt, _ = eigh(C, Finv)
    lam_gt = np.sort(np.real(lam_gt))[::-1]

    M = Fh @ C @ Fh
    M = (M + M.T) / 2.0
    lams, Y = np.linalg.eigh(M)
    order = np.argsort(lams)[::-1]
    lams = lams[order]
    Y = Y[:, order]
    V = Fh @ Y

    res = 0.0
    for i in range(L):
        res = max(res, np.linalg.norm(C @ V[:, i] - lams[i] * (Finv @ V[:, i])))
    orth = np.linalg.norm(V.T @ Finv @ V - np.eye(L), "fro")

    print("Test 1 (random C,F):")
    print(f"  eigenvalue diff:     {np.linalg.norm(lam_gt - lams):.2e}")
    print(f"  max Cv residual:     {res:.2e}  (target ~0)")
    print(f"  Finv orth deviation: {orth:.2e}  (target ~1e-14)")

    V_old = np.diag(1.0 / np.sqrt(F_diag)) @ Y
    res_old = 0.0
    for i in range(L):
        res_old = max(res_old, np.linalg.norm(C @ V_old[:, i] - lams[i] * (Finv @ V_old[:, i])))
    print(f"  (old F^(-1/2)Y residual for comparison: {res_old:.4f})\n")

    # Test 2: fed_cov_main output fields
    x = rng.standard_normal(256)
    out = fed_cov_main(
        x,
        L=48,
        num_modes=8,
        f_transform="power",
        f_param=2,
        verbose=False,
    )

    print("Test 2 (fed_cov_main):")
    print(f"  Finv_orthogonality_deviation: {out['Finv_orthogonality_deviation']:.2e}")
    print(f"  cv_eq_residual:               {out['cv_eq_residual']:.2e}")
    print(f"  formula: {out['formula']}")

    ok = res < 1e-8 and orth < 1e-6 and out["Finv_orthogonality_deviation"] < 1e-6
    if ok:
        print("\nPASS — main release implementation is consistent.")
    else:
        print("\nWARN — check tolerances or implementation.")


if __name__ == "__main__":
    main()
