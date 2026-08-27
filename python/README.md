# FED — Python

Python implementation of covariance-form FED. See the [main README](../README.md) for the algorithm overview and shared parameters.

## Files

| File | Description |
|------|-------------|
| `fed_cov_main.py` | Main decomposition function (Higuchi FD, F-transform, SSA-style reconstruction) |
| `demo_fed_main.py` | Synthetic signal demo: SSA vs FED (HFD², p=2) |
| `verify_fed_main.py` | Algebra + numerical sanity check for Step 6 |
| `requirements.txt` | Dependencies |

## Requirements

- Python 3.9+
- `numpy` (required)
- `scipy` (required for `verify_fed_main.py` generalized eigen check)
- `matplotlib` (optional, for demo figure export)

## Quick start

```bash
cd python
pip install -r requirements.txt
python verify_fed_main.py
python demo_fed_main.py
python demo_fed_main.py --no-save-fig
```

## Example

```python
import numpy as np
from fed_cov_main import fed_cov_main

x = np.random.randn(512)
out = fed_cov_main(
    x, L=48, num_modes=12, f_transform="power", f_param=2, verbose=True
)
print(f"epsL2={out['epsilon_L2']:.3f}, F-selectivity={out['F_selectivity']:.3f}")
```

## Output dict (selected keys)

Same keys as documented in the [main README](../README.md#output-fields-selected).

> **Note:** Exact demo values will differ slightly from the MATLAB version due to differing RNG implementations (MATLAB vs NumPy); qualitative behavior (SSA low-error/low-selectivity vs. FED's selectivity–reconstruction trade-off) is preserved.
