# FED — MATLAB

MATLAB implementation of covariance-form FED. See the [main README](../README.md) for the algorithm overview and shared parameters.

## Files

| File | Description |
|------|-------------|
| `fed_cov_main.m` | Main decomposition function (Higuchi FD, F-transform, SSA-style reconstruction) |
| `demo_fed_main.m` | Synthetic signal demo: SSA vs FED (HFD², p=2) |
| `verify_fed_main.m` | Algebra + numerical sanity check for Step 6 |

## Requirements

- MATLAB R2019b+ (tested on R2022a)
- No toolboxes required for the core demo
- `exportgraphics` used when available; otherwise `print` for figure export

## Quick start

```matlab
cd matlab
verify_fed_main          % sanity check (~1 s)
demo_fed_main            % demo + figure in output/
demo_fed_main('SaveFig', false)
```

## Example

```matlab
x = randn(512, 1);
out = fed_cov_main(x, 'L', 48, 'NumModes', 12, ...
    'FTransform', 'power', 'FParam', 2, 'Verbose', true);
fprintf('epsL2=%.3f, F-selectivity=%.3f\n', out.epsilon_L2, out.F_selectivity);
```

## Output struct (selected fields)

Same field names as documented in the [main README](../README.md#output-fields-selected).
