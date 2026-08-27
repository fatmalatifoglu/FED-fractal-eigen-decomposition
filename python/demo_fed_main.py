#!/usr/bin/env python3
"""Minimal FED demonstration (main release, Python)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fed_cov_main import fed_cov_main


def build_synthetic_signal(fs: int = 128, duration: int = 4, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t_len = fs * duration
    t = np.arange(t_len) / fs

    c1 = 1.0 * np.sin(2 * np.pi * 3 * t)
    c2 = 0.7 * np.sin(2 * np.pi * 12 * t) * (0.6 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    c3 = np.zeros(t_len)
    for bt in (0.5, 1.5, 2.5, 3.5):
        c3 += 0.5 * np.exp(-((t - bt) ** 2) / (2 * 0.1**2)) * np.sin(2 * np.pi * 30 * t)

    x = c1 + c2 + c3 + 0.1 * rng.standard_normal(t_len)
    return t, x


def print_row(name: str, out: dict) -> None:
    print(
        f"{name:<22}  {out['epsilon_L2']:5.3f}   "
        f"{out['epsilon_F_local']:9.3f}   {out['F_selectivity']:5.3f}   "
        f"{out['Finv_orthogonality_deviation']:8.1e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FED main demo (Python)")
    parser.add_argument("--fs", type=int, default=128)
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--L", type=int, default=48)
    parser.add_argument("--num-modes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-fig", action="store_true", default=True)
    parser.add_argument("--no-save-fig", dest="save_fig", action="store_false")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()

    t, x = build_synthetic_signal(fs=args.fs, duration=args.duration, seed=args.seed)
    n = x.size

    print("\n=== FED main demo (Python) ===")
    print(f"Fs={args.fs} Hz, N={n}, L={args.L}, k={args.num_modes}\n")

    common = {
        "L": args.L,
        "num_modes": args.num_modes,
        "f_strategy": "row",
        "kmax": 6,
        "verbose": True,
    }

    print("--- SSA baseline (F = I) ---")
    out_ssa = fed_cov_main(x, **common, f_transform="identity", f_param=1)

    print("\n--- FED default (F = HFD^2, p = 2) ---")
    out_fed = fed_cov_main(x, **common, f_transform="power", f_param=2)

    print("\n=== Summary ===")
    print(f"{'Method':<22}  epsL2   epsF_high   F-Sel   Finv_orth")
    print("-" * 58)
    print_row("SSA (F=I)", out_ssa)
    print_row("FED (HFD^2,p=2)", out_fed)

    if args.save_fig:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not installed; skipping figure export.")
            print("Install with: pip install matplotlib")
        else:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")

            for ax, out, title in (
                (axes[0], out_ssa, f"SSA (F=I)  epsL2={out_ssa['epsilon_L2']:.3f}"),
                (axes[1], out_fed, f"FED (HFD^2)  epsL2={out_fed['epsilon_L2']:.3f}"),
            ):
                ax.plot(t, x, color=(0.2, 0.2, 0.2), label="Signal")
                ax.plot(t, out["x_hat"], linewidth=1.1, label="Reconstruction")
                ax.grid(True)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_title(title)
                ax.legend(loc="best")

            fpath = args.out_dir / "demo_fed_main_reconstruction.png"
            fig.tight_layout()
            fig.savefig(fpath, dpi=150)
            plt.close(fig)
            print(f"\nFigure saved: {fpath}")

    print("\nDone. Run verify_fed_main.py for the Step-6 algebra check.")


if __name__ == "__main__":
    main()
