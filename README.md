# FED — Fractal Eigen Decomposition (main release)

Minimal public release of the **covariance-form FED** used in the BSPC manuscript, with equivalent **MATLAB** and **Python** implementations.

## Repository layout

```
FED_github/
├── matlab/          MATLAB implementation  →  see [matlab/README.md](matlab/README.md)
├── python/          Python implementation  →  see [python/README.md](python/README.md)
├── LICENSE
└── README.md        (this file — shared overview)
```

## Core idea

Generalized eigenproblem on the Hankel trajectory covariance:

```
C v = λ F^(-1) v
```

Symmetric reduction and back-projection (Algorithm 1, Step 6):

```
M = F^(1/2) C F^(1/2),   M y = λ y,   v = F^(1/2) y
v_i ← v_i / sqrt(v_i' F^(-1) v_i)
```

When `F = I`, FED reduces to standard SSA.

## Implementations

| Language | Folder | Quick start |
|----------|--------|-------------|
| MATLAB | [`matlab/`](matlab/) | `cd matlab` → `verify_fed_main` → `demo_fed_main` |
| Python | [`python/`](python/) | `cd python` → `pip install -r requirements.txt` → `python verify_fed_main.py` |

Both folders provide the same three entry points:

- **`fed_cov_main`** — main decomposition
- **`demo_fed_main`** — synthetic SSA vs FED demo
- **`verify_fed_main`** — Step-6 algebra check

> **Note:** Exact demo values will differ slightly between MATLAB and Python due to differing RNG implementations; qualitative behavior (SSA low-error/low-selectivity vs. FED's selectivity–reconstruction trade-off) is preserved.

## Default parameters (paper)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `L` | `floor(N/3)` if empty | Window length |
| `NumModes` / `num_modes` | 12 | Number of components |
| `FTransform` / `f_transform` | `'power'` | `'identity'` → SSA |
| `FParam` / `f_param` | 2 | HFD² |
| `Kmax` / `kmax` | 6 | Higuchi k_max |
| `FStrategy` / `f_strategy` | `'row'` | Row-wise HFD on trajectory matrix |

## Output fields (selected)

- `components`, `x_hat`, `residue` — time-domain modes and reconstruction
- `epsilon_L2`, `epsilon_F_local`, `epsilon_F_low` — error metrics
- `F_selectivity` — mode–HFD alignment
- `Finv_orthogonality_deviation`, `cv_eq_residual` — internal consistency checks

## Relation to full project

This folder is a **minimal public subset**. The full research codebase (EEG pipeline, ablation tables, paper figures) lives in the parent `FED_Basic` repository. Internal development names (`fed_cov_v4`, etc.) map to **`fed_cov_main`** in both language folders.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the corresponding BSPC article (Fractal Eigen Decomposition for EEG-based schizophrenia classification).
