# Ferromagnetism lab — data processing

## Layout

- `proc.py` — marimo notebook (run with `marimo run proc.py` or `marimo edit proc.py`).
  - **Part A** — virgin magnetization curve `B(H)` and `μ_r(H)` from
    dual-cursor scope readings, cubic-spline interpolated.
  - **Part B** — fit of `NI/B` vs air-gap `L'` to extract `μ_0` from the
    inverse slope and `μ_iron` from the intercept.
- `instruments.py` — uncertainty models for the bench instruments used
  (HP 34401A DMM, Agilent DSO7012A scope).

## Fit method

Part B uses ODR on `(L', V_x/V_y)` with σ from the voltage readings only.
The apparatus factor `K = N²A / (R_x R_y C)` is common to every point in a
run, so its uncertainty is folded in *after* the fit as a multiplicative
factor on `μ_0 = 1/(K · slope)` — otherwise ODR would dilute a shared bias
by `√N`.

## Run

```sh
uv run marimo run proc.py
```
