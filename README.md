# Ferromagnetism lab — data processing

## Layout

- `proc.py` — marimo notebook (run with `marimo run proc.py` or `marimo edit proc.py`).
  - **Part A** — virgin magnetization curve $B(H)$ and $\mu_r(H)$ from
    dual-cursor scope readings, cubic-spline interpolated.
  - **Part B** — fit of $NI/B$ vs air-gap $L'$ to extract $\mu_0$ from the
    inverse slope and $\mu_\text{iron}$ from the intercept.
- `instruments.py` — uncertainty models for the bench instruments used
  (HP 34401A DMM, Agilent DSO7012A scope).

## Fit method

Part B uses ODR on $(L', V_x/V_y)$ with $\sigma$ from the voltage readings only.
The apparatus factor

$$K = \frac{N^2 A}{R_x R_y C}$$

is common to every point in a run, so its uncertainty is folded in *after*
the fit as a multiplicative factor on

$$\mu_0 = \frac{1}{K \cdot \text{slope}}$$

— otherwise ODR would dilute a shared bias by $\sqrt{N}$.

## Run

```sh
pip3 install -r requirements.txt
marimo run proc.py
```
