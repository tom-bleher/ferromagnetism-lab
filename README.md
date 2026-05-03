# Ferromagnetism lab — data processing

## Layout

The two experiment parts live in their own subdirectories:

- `ferromagnetism/` — Part 1, hysteresis loop and $\mu_0$ extraction.
  - `proc.py` — marimo notebook (`uv run marimo edit ferromagnetism/proc.py`).
    - **Part A** — initial magnetization curve $B(H)$ and $\mu_r(H)$ from
      dual-cursor scope readings, cubic-spline interpolated.
    - **Part B** — fit of $NI/B$ vs copper-gap $L'$ to extract $\mu_0$ from the
      inverse slope and $\mu_\text{iron}$ from the intercept.
  - `data/data.xlsx`, `data/data.md` — apparatus constants, virgin-curve
    cursor readings, and per-run gap-sweep cursor readings.
  - `moodle/` — supplementary MATLAB scripts from the course Moodle and their
    Python translations.

- `curie/` — Part 2, Curie-temperature analysis.
  - `proc.py` — marimo notebook (`uv run marimo edit curie/proc.py`);
    calibrates the LabVIEW branches, subtracts a high-temperature linear
    background, extracts remanence, fixed-field, and loop-area $M(T)$
    proxies, and normalizes them to 0--1.
  - `data/{first,second,third}/` — LabVIEW capture runs.
  - Apparatus constants are read from `../ferromagnetism/data/data.xlsx`.

- `instruments.py` — uncertainty models for the bench instruments
  (HP 34401A DMM, Agilent DSO7012A scope), shared by both notebooks.

Generated figures are written directly to `../report/media/`, so rerunning
either notebook updates the files used by the LaTeX report.

## Fit method (Part B)

ODR on $(L', V_x/V_y)$ with $\sigma$ from the voltage readings only. The
apparatus factor

$$K = \frac{N^2 A}{R_x R_y C}$$

is common to every point in a run, so its uncertainty is folded in *after*
the fit as a multiplicative factor on

$$\mu_0 = \frac{1}{K \cdot \text{slope}}$$

— otherwise ODR would dilute a shared bias by $\sqrt{N}$.

## Run

```sh
uv sync
uv run marimo edit ferromagnetism/proc.py  # Part 1
uv run marimo edit curie/proc.py           # Part 2
```
