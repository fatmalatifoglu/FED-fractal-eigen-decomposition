"""
Fractal Eigen Decomposition (covariance form, main release).

Generalized eigenproblem:  C v = lambda * F^(-1) v
Equivalent symmetric form: M y = lambda y,  M = F^(1/2) C F^(1/2)
Back-projection (Step 6):    v = F^(1/2) y
F^(-1)-normalization:        v_i <- v_i / sqrt(v_i' F^(-1) v_i)

When F = I this reduces to standard SSA (Proposition 4).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

FStrategy = Literal["row", "seg", "none"]
FTransform = Literal["identity", "linear", "power", "exp", "threshold", "sigmoid"]


def fed_cov_main(
    x: np.ndarray,
    *,
    L: int | None = None,
    num_modes: int = 12,
    f_strategy: FStrategy = "row",
    f_transform: FTransform = "power",
    f_param: float = 2.0,
    kmax: int = 6,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run FED covariance decomposition on a 1-D signal."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size

    if L is None:
        L = n // 3
    L = min(int(L), n // 2)
    K = n - L + 1

    # Hankel embedding
    X_traj = np.zeros((L, K))
    for i in range(K):
        X_traj[:, i] = x[i : i + L]

    # Covariance matrix
    C = (X_traj @ X_traj.T) / K
    C = (C + C.T) / 2.0

    # Raw fractal metric
    HFD_raw = np.zeros(L)
    min_len = max(8, 2 * kmax)
    if f_strategy == "row":
        for i in range(L):
            row_data = X_traj[i, :]
            if K >= min_len:
                HFD_raw[i] = higuchi_fd(row_data, kmax)
            else:
                HFD_raw[i] = np.std(row_data)
    elif f_strategy == "seg":
        for i in range(L):
            seg = x[i : i + K]
            if seg.size >= min_len:
                HFD_raw[i] = higuchi_fd(seg, kmax)
            else:
                HFD_raw[i] = np.std(seg)
    elif f_strategy == "none":
        HFD_raw = np.ones(L)
    else:
        raise ValueError(f"Unknown f_strategy: {f_strategy!r}")

    F_diag = transform_F(HFD_raw, f_transform, f_param)
    F_diag = np.maximum(F_diag, 1e-6)
    F_inv_diag = 1.0 / F_diag
    F_inv = np.diag(F_inv_diag)

    if verbose:
        print(f"  [FED-COV main] N={n}, L={L}, K={K}")
        print(
            f"  [FED-COV main] F: min={F_diag.min():.3f}, max={F_diag.max():.3f}, "
            f"ratio={F_diag.max() / F_diag.min():.2f}"
        )

    F_half = np.diag(np.sqrt(F_diag))
    M = F_half @ C @ F_half
    M = (M + M.T) / 2.0

    lams, Y = np.linalg.eigh(M)
    order = np.argsort(lams)[::-1]
    lams = lams[order]
    Y = Y[:, order]

    k_use = min(num_modes, L - 1)
    lams = lams[:k_use]
    Y = Y[:, :k_use]

    V = F_half @ Y
    for i in range(k_use):
        nrm = np.sqrt(max(float(V[:, i].T @ F_inv @ V[:, i]), np.finfo(float).eps))
        V[:, i] /= nrm

    U = np.zeros((L, k_use))
    for i in range(k_use):
        nrm = np.linalg.norm(V[:, i]) + np.finfo(float).eps
        U[:, i] = V[:, i] / nrm

    components = np.zeros((n, k_use))
    sigmas = np.zeros(k_use)
    energies = np.zeros(k_use)

    for i in range(k_use):
        u_i = U[:, i]
        var_i = float(u_i.T @ C @ u_i)
        sigma_i = np.sqrt(max(K * var_i, 0.0))
        sigmas[i] = sigma_i
        if sigma_i < np.finfo(float).eps:
            continue

        V_traj_i = (X_traj.T @ u_i) / sigma_i
        X_i = sigma_i * np.outer(u_i, V_traj_i)
        components[:, i] = antidiag_avg(X_i, n)
        energies[i] = np.sum(components[:, i] ** 2)

    x_hat = np.sum(components, axis=1)
    residue = x - x_hat

    epsilon_L2 = np.linalg.norm(residue) / (np.linalg.norm(x) + np.finfo(float).eps)

    R_traj = np.zeros((L, K))
    for i in range(K):
        R_traj[:, i] = residue[i : i + L]

    res_F_norm = np.sqrt(np.trace(R_traj.T @ F_half @ F_half @ R_traj))
    sig_F_norm = np.sqrt(np.trace(X_traj.T @ F_half @ F_half @ X_traj))
    epsilon_F_global = res_F_norm / (sig_F_norm + np.finfo(float).eps)

    HFD_thr = np.quantile(HFD_raw, 0.75)
    high_idx = HFD_raw > HFD_thr

    if np.any(high_idx) and np.any(~high_idx):
        res_high = R_traj[high_idx, :]
        sig_high = X_traj[high_idx, :]
        epsilon_F_local = np.linalg.norm(res_high, "fro") / (
            np.linalg.norm(sig_high, "fro") + np.finfo(float).eps
        )
        res_low = R_traj[~high_idx, :]
        sig_low = X_traj[~high_idx, :]
        epsilon_F_low = np.linalg.norm(res_low, "fro") / (
            np.linalg.norm(sig_low, "fro") + np.finfo(float).eps
        )
    else:
        epsilon_F_local = epsilon_L2
        epsilon_F_low = epsilon_L2

    mod_HFD_active = np.zeros(k_use)
    for i in range(k_use):
        weights = np.abs(U[:, i])
        if np.sum(weights) > np.finfo(float).eps:
            mod_HFD_active[i] = np.sum(weights * HFD_raw) / np.sum(weights)

    if k_use >= 3:
        if np.std(energies) > 0 and np.std(mod_HFD_active) > 0:
            F_selectivity = float(np.corrcoef(energies, mod_HFD_active)[0, 1])
        else:
            F_selectivity = float("nan")
    else:
        F_selectivity = float("nan")

    Finv_orth_dev = np.linalg.norm(V.T @ F_inv @ V - np.eye(k_use), "fro")
    cv_residual = np.linalg.norm(C @ V[:, 0] - lams[0] * (F_inv @ V[:, 0]))

    out = {
        "components": components,
        "modes_2d_F": V,
        "modes_2d_E": U,
        "lambda": lams,
        "sigmas": sigmas,
        "energies": energies,
        "HFD_raw": HFD_raw,
        "F_diag": F_diag,
        "F_inv_diag": F_inv_diag,
        "X_traj": X_traj,
        "C": C,
        "x_orig": x,
        "x_hat": x_hat,
        "residue": residue,
        "epsilon_L2": float(epsilon_L2),
        "epsilon_F_global": float(epsilon_F_global),
        "epsilon_F_local": float(epsilon_F_local),
        "epsilon_F_low": float(epsilon_F_low),
        "F_selectivity": F_selectivity,
        "mod_HFD_active": mod_HFD_active,
        "Finv_orthogonality_deviation": float(Finv_orth_dev),
        "cv_eq_residual": float(cv_residual),
        "L": L,
        "K": K,
        "N": n,
        "k": k_use,
        "params": {
            "L": L,
            "NumModes": num_modes,
            "FStrategy": f_strategy,
            "FTransform": f_transform,
            "FParam": f_param,
            "Kmax": kmax,
            "Verbose": verbose,
        },
        "formula": "C v = lambda * F^(-1) v  [v = F^(1/2) y]",
    }

    if verbose:
        print(
            f"  [FED-COV main] epsL2={epsilon_L2:.4f} | epsF_g={epsilon_F_global:.4f} | "
            f"epsF_y={epsilon_F_local:.4f} | epsF_d={epsilon_F_low:.4f}"
        )
        print(
            f"  [FED-COV main] F_selectivity={F_selectivity:.3f} | lambda_1={lams[0]:.3f}"
        )
        print(
            f"  [FED-COV main] check: Finv_orth_dev={Finv_orth_dev:.2e} | "
            f"cv_residual={cv_residual:.2e}"
        )

    return out


def transform_F(HFD: np.ndarray, transform: str, param: float) -> np.ndarray:
    hfd_n = HFD - np.min(HFD)
    if np.max(hfd_n) > np.finfo(float).eps:
        hfd_n = hfd_n / np.max(hfd_n)

    transform = transform.lower()
    if transform == "identity":
        f_out = np.ones_like(HFD, dtype=float)
    elif transform == "linear":
        f_out = 0.1 + 0.9 * hfd_n
    elif transform == "power":
        f_out = 0.1 + 0.9 * (hfd_n**param)
    elif transform == "exp":
        f_out = np.exp(param * hfd_n)
        f_out = f_out / np.max(f_out)
        f_out = np.maximum(f_out, 0.01)
    elif transform == "threshold":
        thr = np.quantile(HFD, param)
        f_out = np.full_like(HFD, 0.1, dtype=float)
        f_out[HFD > thr] = 1.0
    elif transform == "sigmoid":
        z = (HFD - np.mean(HFD)) / (np.std(HFD) + np.finfo(float).eps)
        f_out = 1.0 / (1.0 + np.exp(-param * z))
        f_out = np.maximum(f_out, 0.01)
    else:
        raise ValueError(f"Unknown F transform: {transform!r}")

    return np.maximum(f_out, 1e-6)


def antidiag_avg(X: np.ndarray, N: int) -> np.ndarray:
    L, K = X.shape
    y = np.zeros(N)
    c = np.zeros(N)
    for i in range(L):
        for j in range(K):
            n = i + j
            if 0 <= n < N:
                y[n] += X[i, j]
                c[n] += 1.0
    return y / np.maximum(c, 1.0)


def higuchi_fd(x: np.ndarray, kmax: int) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    N = x.size
    kmax = min(kmax, max(1, N // 3))
    if N < 8 or kmax < 2:
        return float(np.std(x))

    L = np.zeros(kmax)
    for k in range(1, kmax + 1):
        Lk = 0.0
        for m in range(1, k + 1):
            idx = np.arange(m - 1, N, k)
            Nm = idx.size - 1
            if Nm < 1:
                continue
            denom = Nm * (k**2) / (N - 1)
            if denom < np.finfo(float).eps:
                continue
            Lk += np.sum(np.abs(np.diff(x[idx]))) / denom
        L[k - 1] = Lk / k

    valid = L > np.finfo(float).eps
    if np.count_nonzero(valid) < 2:
        return float(np.std(x))

    ks = np.arange(1, kmax + 1)[valid]
    p = np.polyfit(np.log(ks), np.log(L[valid] + np.finfo(float).eps), 1)
    fd = float(np.clip(-p[0], 0.0, 2.0))
    if not np.isfinite(fd):
        return float(np.std(x))
    return fd
