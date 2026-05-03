import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", app_title="Ferromagnetism")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ferromagnetism

    ### **Part A — Magnetization curve of the iron core**

    Primary AC current is swept; the scope dual-cursor readings
    $\Delta V_x,\Delta V_y$ span the **full peak-to-peak** of each axis
    (the positive to negative saturation distance on the hysteresis loop),
    so $V_\text{peak} = \Delta V/2$ and

    $$H = \frac{N}{2\,L\,R_x}\,\Delta V_x, \qquad B = \frac{R_y\,C}{2\,N\,A}\,\Delta V_y.$$

    Cubic-spline fits give smooth $B(H)$ and $\mu_r(H)=\frac{B}{\mu_0 H}$.

    ### **Part B — Vacuum permeability $\mu_0$**

    Copper plates of total thickness $L'$ are inserted with $B$ held constant.
    Ampère's law gives

    $$\frac{B}{\mu}\,L + \frac{B}{\mu_0}\,L' = N I \;\;\Longrightarrow\;\; \frac{N I}{B} = \frac{1}{\mu_0}\,L' + \frac{L}{\mu_\text{iron}},$$

    so a linear fit of $N I/B$ vs $L'$ yields $\mu_0$ from the inverse slope
    and $\mu_\text{iron}$ from the intercept.
    """)
    return


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from uncertainties import ufloat, unumpy as unp
    from scipy.interpolate import CubicSpline

    from taulab import (
        PhysicalSize,
        combine,
        fit_functions,
        nsigma,
        odr_fit,
        read_table,
    )

    from instruments import (
        caliper,
        column_lsd,
        digital_multimeter_resistance,
        oscilloscope_dual_cursor,
        reading_lsd,
        ruler,
    )

    plt.rcParams.update({
        'figure.figsize':    (8.0, 4.8),
        'figure.dpi':        110,
        'savefig.dpi':       600,
        'font.size':         11,
        'font.family':       'DejaVu Sans',
        'mathtext.fontset':  'cm',
        'axes.titlesize':    12,
        'axes.titlepad':     8,
        'axes.labelsize':    11,
        'axes.labelpad':     5,
        'axes.grid':         True,
        'axes.axisbelow':    True,
        'grid.alpha':        0.25,
        'grid.linestyle':    '--',
        'grid.linewidth':    0.5,
        'legend.fontsize':   9,
        'legend.framealpha': 0.92,
        'lines.linewidth':   1.4,
        'xtick.direction':   'in',
        'ytick.direction':   'in',
        'xtick.top':         True,
        'ytick.right':       True,
    })

    _here = Path(__file__).resolve().parent
    DATA_XLSX = _here / 'data' / 'data.xlsx'
    FIG_DIR    = _here.parent.parent / 'report' / 'media'
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # CODATA-2022 μ₀ (post-2019 SI); uncertainty ~10⁻¹⁶ is negligible
    # vs. our ~10⁻⁸ measurement σ, but is carried for nσ reporting.
    MU0_THEO   = 1.25663706127e-6
    MU0_THEO_U = ufloat(MU0_THEO, 2.0e-16)

    def fmt(x, unit=''):
        """LaTeX $v \\pm \\sigma\\,\\mathrm{unit}$ for ufloat / PhysicalSize."""
        if hasattr(x, 'value'):
            x = ufloat(x.value, x.uncertainty)
        core = format(x, '.2uL')
        umod = r'\,\mathrm{' + unit + '}' if unit else ''
        return f'${core}{umod}$'

    return (
        CubicSpline,
        DATA_XLSX,
        FIG_DIR,
        MU0_THEO,
        MU0_THEO_U,
        PhysicalSize,
        caliper,
        column_lsd,
        combine,
        digital_multimeter_resistance,
        fit_functions,
        fmt,
        np,
        nsigma,
        odr_fit,
        oscilloscope_dual_cursor,
        pd,
        plt,
        read_table,
        reading_lsd,
        ruler,
        ufloat,
        unp,
    )


@app.cell(hide_code=True)
def _(mo):
    _budget_md = r"""
    | Quantity | Value | Uncertainty (rel) | Comment/Origin |
    |---|---|---|---|
    | $N$ (turns) | $250$ | — | count given |
    | $L$ (Ampère loop) | $0.48\,\mathrm{m}$ | $\frac{2\sqrt{2}\,\mathrm{mm}}{\sqrt{12}} \approx 0.82\,\mathrm{mm}$ $(0.17\%)$ | two ruler measurements for sides $a,b$ (res $1\,\mathrm{mm}$) and indirect $L = 2(a+b)$: $\sigma_L = 2\sqrt{\sigma_a^2 + \sigma_b^2}$ |
    | $L'$ | per measurement | $\frac{0.05\,\mathrm{mm}}{\sqrt{12}} \approx 14.4\,\mu\mathrm{m}$ | caliper (res $0.05\,\mathrm{mm}$) |
    | $R_x$ (current-sense) | $2.999\,\mathrm{\Omega}$ | $4.3\,\mathrm{m\Omega}$ $(0.14\%)$ | HP 34401A manual, p. 216: 1-year resistance accuracy on 100 Ω range, $0.010\%$ reading + $0.004\%$ range, plus $\mathrm{LSD}/\sqrt{12}$ |
    | $R_y$ (integrator) | $11.10\,\mathrm{k\Omega}$ | $2.1\,\mathrm{\Omega}$ $(0.019\%)$ | HP 34401A manual, p. 216: 1-year resistance accuracy on 100 kΩ range, $0.010\%$ reading + $0.001\%$ range |
    | $C$ (integrator) | $20.1\,\mu\mathrm{F}$ | $2.0\,\mathrm{nF}$ $(0.01\%)$ | value and uncertainty given |
    | $A$ (core cross-section) | $16.0\,\mathrm{cm^{2}}$ | $A\,\sqrt{2}\,\dfrac{1\,\mathrm{mm}/\sqrt{12}}{40\,\mathrm{mm}} \approx 0.16\,\mathrm{cm^{2}}$ $(1.02\%)$ | two ruler measurements for sides $a,b\approx 4\,\mathrm{cm}$ (res $1\,\mathrm{mm}$) and indirect $A=a\cdot b$: $\sigma_A/A = \sqrt{(\sigma_a/a)^2 + (\sigma_b/b)^2}$ |
    | $\Delta V_x,\,\Delta V_y$ (scope, dual-cursor) | per measurement | $\sqrt{(0.024\,\lvert V\rvert)^2 + (5\,\mathrm{mV})^2 + (\mathrm{LSD}/\sqrt{12})^2}$ | Agilent 7000A data sheet, p. 18: dual-cursor accuracy $=2.0\%$ vertical gain + $0.4\%$ full scale; full scale approximated by the measured cursor span |
    """
    mo.vstack([mo.md("## Uncertainties"), mo.center(mo.md(_budget_md))])
    return


@app.cell
def _(
    DATA_XLSX,
    MU0_THEO,
    caliper,
    column_lsd,
    combine,
    digital_multimeter_resistance,
    fmt,
    mo,
    np,
    oscilloscope_dual_cursor,
    read_table,
    reading_lsd,
    ruler,
    ufloat,
):
    _p = read_table(DATA_XLSX, sheet_name='apparatus').iloc[0]
    N      = int(_p['N'])
    L_nom  = float(_p['L (m)'])
    Rx_nom = float(_p['Rx (Ω)'])
    Ry_nom = float(_p['Ry (Ω)'])
    C_nom  = float(_p['C (F)'])
    A_nom  = float(_p['A (m²)'])

    # L = 2(a + b) from two 1-mm-ruler readings ⇒ σ_L = 2√(σ_a²+σ_b²).
    u_L  = 2.0 * combine(ruler(), ruler())
    u_Lp = caliper()

    # Ry should be measured to get LSD
    u_Rx = digital_multimeter_resistance(Rx_nom)
    u_Ry = digital_multimeter_resistance(Ry_nom, include_lsd=False)
    u_C  = 0.0001 * C_nom

    # A = a·b with a = b = 4 cm on a 1-mm ruler ⇒ σ_A/A ≈ 1.0 %.
    _u_side_rel = ruler() / 0.04
    u_A         = A_nom * combine(_u_side_rel, _u_side_rel)

    def u_V(V, lsd=None):
        """Scope ΔV σ — oscilloscope dual-cursor spec ⊕ LSD/√12."""
        V = np.asarray(V, dtype=float)
        if lsd is None:
            lsd = column_lsd(V) if V.ndim > 0 else reading_lsd(V.item())
        return oscilloscope_dual_cursor(V, lsd=lsd)

    L  = ufloat(L_nom,  u_L,  'L')
    Rx = ufloat(Rx_nom, u_Rx, 'Rx')
    Ry = ufloat(Ry_nom, u_Ry, 'Ry')
    C  = ufloat(C_nom,  u_C,  'C')
    A  = ufloat(A_nom,  u_A,  'A')

    # ΔV is peak-to-peak; V_peak = ΔV/2, and the 1/2 is absorbed into
    # the calibration constants so they apply directly to ΔV.
    H_per_Vx = N / (2 * L * Rx)
    B_per_Vy = Ry * C / (2 * N * A)

    f_drive = 50.0
    omega   = 2 * np.pi * f_drive
    T_drive = 1.0 / f_drive
    tau_RC  = Ry * C
    wRyC    = omega * tau_RC
    _wRyC_nom = wRyC.n
    _integrator_gain_ratio = _wRyC_nom / np.sqrt(1 + _wRyC_nom**2)
    _integrator_gain_error_pct = (1 - _integrator_gain_ratio) * 100
    _integrator_phase_deg = np.degrees(np.arctan(1 / _wRyC_nom))

    mo.md(f"""
    **Apparatus constants**

    | | value |
    |---|---|
    | $N$ | ${N}$ |
    | $L$ | {fmt(L, 'm')} |
    | $R_x$ | {fmt(Rx, r'\\Omega')} |
    | $R_y$ | {fmt(Ry, r'\\Omega')} |
    | $C$ | {fmt(C, 'F')} |
    | $A$ | {fmt(A, 'm^{{2}}')} |
    | $\\mu_0^{{\\mathrm{{CODATA}}}}$ | ${MU0_THEO:.4e}\\,\\mathrm{{T\\,m\\,A^{{-1}}}}$ |

    **Integrator check**

    The $R_y$–$C$ circuit is a valid integrator at the 50 Hz drive
    frequency. Its exact transfer function is
    $G_{{RC}}(\\omega)=1/(1+i\\omega R_yC)$, which reduces to the ideal
    integrator $1/(i\\omega R_yC)$ when $\\omega R_yC\\gg1$:

    $$\\tau = R_y C = ({tau_RC*1e3:.2uL})\\,\\mathrm{{ms}} \\;\\gg\\; T = 1/f = {T_drive*1e3:.1f}\\,\\mathrm{{ms}}, \\qquad \\omega\\tau = {wRyC:.2uL}.$$

    Quantitatively,
    $\\left|G_{{RC}}/(1/i\\omega R_yC)\\right|={_integrator_gain_ratio:.5f}$,
    so the non-ideal amplitude correction is only {_integrator_gain_error_pct:.3f}%
    and the phase departure from an ideal integrator is {_integrator_phase_deg:.2f}°.
    """)
    return A, B_per_Vy, C, H_per_Vx, L, N, Rx, Ry, u_Lp, u_V


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part A — Initial magnetization curve $B(H)$ and $\mu_r(H)$

    **Scope of Part A.** The data captured here is a 13-row table of
    $(\Delta V_x, \Delta V_y)$ peak-to-peak readings, one per primary
    drive level. Each row characterizes a single Lissajous loop by its
    *peak-to-peak extent* (top-bottom and left-right excursion on the
    XY display); the loop trajectory itself was not exported. From
    these readings we can extract $B_{\text{peak}}$, $H_{\text{peak}}$,
    $\mu_r(H)$, and the saturation $B$-value, but **not** the
    remanence $B_r=B(H=0)$, the coercivity $H_c=H$ at $B=0$, or the
    hysteresis-loop area (energy loss per cycle). Those would require
    sampling the full loop (XY-mode waveform export, as the Curie
    experiment does). This is a known scope-of-data limitation.
    """)
    return


@app.cell
def _(
    B_per_Vy,
    DATA_XLSX,
    H_per_Vx,
    MU0_THEO,
    read_table,
    u_V,
    unp,
):
    pt1 = read_table(DATA_XLSX, sheet_name='virgin-curve', usecols=[0, 1])
    pt1.columns = ['dVx', 'dVy']

    Vx_u = unp.uarray(pt1['dVx'].to_numpy(), u_V(pt1['dVx'].to_numpy()))
    Vy_u = unp.uarray(pt1['dVy'].to_numpy(), u_V(pt1['dVy'].to_numpy()))

    H_u      = H_per_Vx * Vx_u
    B_u      = B_per_Vy * Vy_u
    mu_rel_u = B_u / H_u / MU0_THEO

    pt1['H']        = unp.nominal_values(H_u)
    pt1['sH']       = unp.std_devs(H_u)
    pt1['B']        = unp.nominal_values(B_u)
    pt1['sB']       = unp.std_devs(B_u)
    pt1['mu_rel']   = unp.nominal_values(mu_rel_u)
    pt1['s_mu_rel'] = unp.std_devs(mu_rel_u)
    return (pt1,)


@app.cell
def _(CubicSpline, FIG_DIR, np, plt, pt1):
    # B(H) anchored at the origin (initial magnetization starts at (0, 0));
    # μ_r(H) not anchored — μ_r(0) is a finite initial permeability we
    # do not measure directly.
    H_nodes = np.concatenate([[0.0], pt1['H'].to_numpy()])
    B_nodes = np.concatenate([[0.0], pt1['B'].to_numpy()])
    H_sorted, _ui = np.unique(H_nodes, return_index=True)
    cs_B    = CubicSpline(H_sorted, B_nodes[_ui], bc_type='natural')
    H_grid  = np.linspace(0.0, pt1['H'].max() * 1.01, 500)

    H_mu    = pt1['H'].to_numpy()
    mu_vals = pt1['mu_rel'].to_numpy()
    Hmu_sorted, _uj = np.unique(H_mu, return_index=True)
    cs_mu    = CubicSpline(Hmu_sorted, mu_vals[_uj], bc_type='natural')
    Hmu_grid = np.linspace(H_mu.min(), H_mu.max(), 500)
    peak_idx = pt1['mu_rel'].idxmax()

    fig, axB = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    axMu = axB.twinx()

    C_B, C_MU = 'C0', 'C3'

    axB.plot(H_grid, cs_B(H_grid), '-', color=C_B, alpha=0.85, label=r'$B$  spline')
    axB.errorbar(pt1['H'], pt1['B'],
                 xerr=pt1['sH'], yerr=pt1['sB'],
                 fmt='o', color=C_B, mfc='white', markersize=5,
                 ecolor=C_B, elinewidth=0.9, capsize=2.5, label=r'$B$  data')
    axB.set_xlabel(r'$H$  (A / m)')
    axB.set_ylabel(r'$B$  (T)', color=C_B)
    axB.tick_params(axis='y', colors=C_B)
    axB.spines['left'].set_color(C_B)
    axB.minorticks_on()
    axB.grid(True, which='major', alpha=0.25)
    axB.grid(True, which='minor', alpha=0.10)

    axMu.plot(Hmu_grid, cs_mu(Hmu_grid), '-', color=C_MU, alpha=0.85,
              label=r'$\mu_r$  spline')
    axMu.errorbar(pt1['H'], pt1['mu_rel'],
                  xerr=pt1['sH'], yerr=pt1['s_mu_rel'],
                  fmt='s', color=C_MU, mfc='white', markersize=5,
                  ecolor=C_MU, elinewidth=0.9, capsize=2.5, label=r'$\mu_r$  data')
    axMu.axvline(pt1.loc[peak_idx, 'H'], color=C_MU,
                 linestyle=':', linewidth=1.0, alpha=0.75, label=r'$\mu_r$ peak')
    axMu.set_ylabel(r'$\mu_r = \mu / \mu_0$', color=C_MU)
    axMu.tick_params(axis='y', colors=C_MU)
    axMu.spines['right'].set_color(C_MU)
    axMu.grid(False)

    axB.set_title('Initial magnetization curve and relative permeability — iron toroid')

    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axMu.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, loc='center right', framealpha=0.92)

    fig.savefig(FIG_DIR / 'fig_BH_mu.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig_BH_mu.png', bbox_inches='tight', dpi=600)
    fig
    return


@app.cell
def _(fmt, mo, pt1, ufloat):
    _i      = pt1['mu_rel'].idxmax()
    mu_peak = ufloat(pt1.loc[_i, 'mu_rel'], pt1.loc[_i, 's_mu_rel'])
    H_peak  = ufloat(pt1.loc[_i, 'H'],      pt1.loc[_i, 'sH'])
    B_peak  = ufloat(pt1.loc[_i, 'B'],      pt1.loc[_i, 'sB'])
    B_max   = ufloat(pt1['B'].iloc[-1],     pt1['sB'].iloc[-1])
    H_max   = ufloat(pt1['H'].iloc[-1],     pt1['sH'].iloc[-1])

    mo.callout(
        mo.md(f"""
    **Part A Results**

    * $\\mu_{{\\mathrm{{max}}}}/\\mu_0 =$ {fmt(mu_peak)} at $H \\approx$ {fmt(H_peak, r'A\\,m^{{-1}}')}; $B(H_{{\\mathrm{{peak}}}}) =$ {fmt(B_peak, 'T')}.
    * $B(H_{{\\mathrm{{max}}}}) =$ {fmt(B_max, 'T')} at $H =$ {fmt(H_max, r'A\\,m^{{-1}}')} (saturation regime).
    """),
        kind="success",
    )
    return B_max, mu_peak


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part B — Permeability of vacuum

    Linear model:

    $$\frac{N I}{B} = \frac{1}{\mu_0}\,L' + \frac{L}{\mu_\text{iron}}.$$

    For each sheet $V_y$ (and therefore $B$) is held constant while $L'$ is
    varied.
    """)
    return


@app.cell
def _(
    A,
    B_per_Vy,
    C,
    DATA_XLSX,
    H_per_Vx,
    L,
    MU0_THEO,
    MU0_THEO_U,
    N,
    PhysicalSize,
    Rx,
    Ry,
    fit_functions,
    fmt,
    mo,
    np,
    nsigma,
    odr_fit,
    pd,
    read_table,
    u_Lp,
    u_V,
    unp,
):
    def fit_gap(sheet):
        r"""ODR :math:`V_x/V_y` vs :math:`L'`; fold :math:`K=N^2A/(R_xR_yC)`
        in after the fit so apparatus σ doesn't dilute by :math:`\sqrt{N}`.
        :math:`\mu_0 = 1/(K \cdot \text{slope})`.
        """
        raw = read_table(DATA_XLSX, sheet_name=sheet)
        # Keep only rows whose plate count parses as a number; a blank row
        # then a text summary block follows the data.
        _plates = pd.to_numeric(raw.iloc[:, 0], errors='coerce')
        df = raw.loc[_plates.notna()].iloc[:, [0, 1, 3, 4]].copy()
        df.columns = ['plates', 'Lp_mm', 'dVx', 'dVy']
        df = df.reset_index(drop=True)

        Lp     = df['Lp_mm'].to_numpy() * 1e-3
        Vx, Vy = df['dVx'].to_numpy(), df['dVy'].to_numpy()
        sVx, sVy = u_V(Vx), u_V(Vy)

        r_vals = Vx / Vy
        sr = r_vals * np.sqrt((sVx / Vx) ** 2 + (sVy / Vy) ** 2)

        x, sx = Lp, np.full_like(Lp, u_Lp)
        result = odr_fit(fit_functions.linear, None, x, sx, r_vals, sr)

        # Each apparatus ufloat keeps its tag, so K's factors stay
        # correctly correlated through the product.
        K = (N ** 2) * A / (Rx * Ry * C)
        slope     = PhysicalSize.from_ufloat(K * result.param(1).to_ufloat())
        intercept = PhysicalSize.from_ufloat(K * result.param(0).to_ufloat())

        # I_peak = ΔV_x/(2 Rx); B = B_per_Vy · ΔV_y.  The 1/2 cancels in
        # NI/B (μ₀ invariant), but plotted y-values must match the fit line.
        B_u  = B_per_Vy * unp.uarray(Vy, sVy)
        NI_B = N * unp.uarray(Vx, sVx) / (2 * Rx) / B_u

        return dict(
            result=result,
            slope=slope, intercept=intercept,
            mu0_exp=1.0 / slope, mu_iron=L / intercept,
            y_full=unp.nominal_values(NI_B),
            sy_full=unp.std_devs(NI_B),
            x=x, sx=sx,
            B_mean=np.mean(unp.nominal_values(B_u)),
            H_max=H_per_Vx.nominal_value * Vx.max(),
        )

    RUNS = {
        'vacuum-permeability-a': dict(label='Three copper plates to none', marker='o', color='C0'),
        'vacuum-permeability-b': dict(label='Five copper plates to none',  marker='s', color='C2'),
    }
    fits = {sheet: fit_gap(sheet) for sheet in RUNS}

    _rows = [
        (r'$\langle B\rangle$',                  lambda f: rf'${f["B_mean"]:.2f}\,\mathrm{{T}}$'),
        (r'$H_{\mathrm{max}}$',                  lambda f: rf'${f["H_max"]:.0f}\,\mathrm{{A\,m^{{-1}}}}$'),
        (r'slope $= 1/\mu_0$',                   lambda f: fmt(f['slope'],     r'A\,T^{-1}\,m^{-1}')),
        (r'intercept $= L/\mu_{\mathrm{iron}}$', lambda f: fmt(f['intercept'], r'A\,T^{-1}')),
        (r'$\mu_0^{\mathrm{exp}}$',              lambda f: fmt(f['mu0_exp'],   r'T\,m\,A^{-1}')),
        (r'$n_\sigma$ vs CODATA',                lambda f: rf'${nsigma(f["mu0_exp"], MU0_THEO_U):.1f}$'),
        (r'$\mu_{\mathrm{iron}}/\mu_0$',         lambda f: fmt(f['mu_iron'] / MU0_THEO)),
        (r'$\chi^2/\nu$',                        lambda f: rf'${f["result"].redchi:.2f}\;({f["result"].chi2:.2f}/{f["result"].dof})$'),
        (r'$p$-value',                           lambda f: rf'${f["result"].p_value:.3f}$'),
    ]
    _run_cols = ' | '.join(RUNS[s]['label'] for s in fits)
    _header = f'| Quantity | {_run_cols} |'
    _sep    = '|---' * (1 + len(fits)) + '|'
    _body   = [f'| {label} | ' + ' | '.join(fn(f) for f in fits.values()) + ' |'
               for label, fn in _rows]
    mo.center(mo.md('\n'.join([_header, _sep, *_body])))
    return RUNS, fits


@app.cell
def _(FIG_DIR, RUNS, fits, np, plt):
    fig2, (ax_fit, ax_res) = plt.subplots(
        2, 1, figsize=(8.4, 6.2), sharex=True, constrained_layout=True,
        gridspec_kw={'height_ratios': [2.4, 1.0]},
    )

    def _draw(sheet, f):
        style         = RUNS[sheet]
        r             = f['result']
        Lp_mm, sx_mm  = f['x'] * 1e3, f['sx'] * 1e3
        slope_NIB     = f['slope'].value
        intercept_NIB = f['intercept'].value
        yhat_NIB      = slope_NIB * f['x'] + intercept_NIB

        ax_fit.errorbar(
            Lp_mm, f['y_full'], xerr=sx_mm, yerr=f['sy_full'],
            fmt=style['marker'], color=style['color'], mfc='white', markersize=5,
            ecolor=style['color'], elinewidth=0.9, capsize=2.5,
            label=rf"{style['label']}  ($\langle B\rangle\approx{f['B_mean']:.2f}$ T,"
                  rf"  $\chi^2/\nu={r.redchi:.2f}$,  $p={r.p_value:.2f}$)",
        )
        xs_m = np.linspace(0, f['x'].max() * 1.05, 60)
        ax_fit.plot(xs_m * 1e3, slope_NIB * xs_m + intercept_NIB,
                    '-', color=style['color'], alpha=0.85)

        ax_res.errorbar(
            Lp_mm, f['y_full'] - yhat_NIB, yerr=f['sy_full'],
            fmt=style['marker'], color=style['color'], mfc='white', markersize=5,
            ecolor=style['color'], elinewidth=0.9, capsize=2.5,
        )

    for _sheet, _f in fits.items():
        _draw(_sheet, _f)

    ax_fit.set_ylabel(r'$N I / B$   (A·m / T)')
    ax_fit.set_title(r"Vacuum permeability: $N I / B$ vs width of copper plates")
    ax_fit.legend(loc='upper left')
    ax_fit.minorticks_on()
    ax_fit.grid(True, which='minor', alpha=0.10)

    ax_res.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_res.set_xlabel(r"$L'$   (mm)")
    ax_res.set_ylabel('residual  (A·m / T)')
    ax_res.minorticks_on()
    ax_res.grid(True, which='minor', alpha=0.10)

    fig2.savefig(FIG_DIR / 'fig_copper_gap.pdf', bbox_inches='tight')
    fig2.savefig(FIG_DIR / 'fig_copper_gap.png', bbox_inches='tight', dpi=600)
    fig2
    return


@app.cell(hide_code=True)
def _(MU0_THEO, RUNS, fits, mo, np):
    # μ_0 sensitivity sweep. The experimental result μ_0^exp = 1/(K·slope),
    # with K = N²A/(R_x R_y C). The log-derivatives are exact:
    #
    #   d ln μ_0 / d ln N   = -2     d ln μ_0 / d ln R_x = +1
    #   d ln μ_0 / d ln A   = -1     d ln μ_0 / d ln R_y = +1
    #   d ln μ_0 / d ln C   = +1     d ln μ_0 / d ln L'  = +1
    #
    # (L' enters via the fit's x-axis: slope ∝ 1/L' in dimensional terms,
    # so any global rescaling of L' propagates with sign +1 into μ_0.)
    #
    # If μ_0^exp / μ_0^CODATA = ratio < 1, the fractional shift the model
    # has to explain in *each* quantity (acting alone) to bring the
    # measurement back to CODATA is:
    #
    #     ε_i = (1/ratio - 1) / s_i,
    #
    # where s_i is the log-derivative above. Smaller |ε| means a smaller
    # systematic in that quantity is sufficient to wipe out the bias —
    # i.e. that quantity is the most likely culprit.
    sens = {
        "N":  -2.0,
        "A":  -1.0,
        "R_x": +1.0,
        "R_y": +1.0,
        "C":   +1.0,
        "L'":  +1.0,
    }

    rows = ["| run | quantity | $d\\ln\\mu_0/d\\ln X$ | shift in $X$ to recover CODATA |",
            "|---|---|---:|---:|"]
    for sheet, f in fits.items():
        ratio = f["mu0_exp"].value / MU0_THEO
        delta = (1.0 / ratio) - 1.0  # fractional change in μ_0 needed
        label = RUNS[sheet]["label"]
        for q, s in sens.items():
            eps = delta / s
            rows.append(f"| {label} | ${q}$ | {s:+.0f} | {eps*100:+.1f}% |")

    table_md = "\n".join(rows)
    mo.md(rf"""
    ### $\mu_0$ sensitivity sweep — which apparatus constant could explain the bias?

    Each row shows the fractional change in one apparatus constant that
    would, *acting alone*, restore $\mu_0^\mathrm{{exp}}$ to its CODATA
    value. Quantities listed with the smallest required shift are the
    most likely systematic culprits given the calibration uncertainty
    budget on the previous table.

    {table_md}

    Reading the table against the apparatus uncertainty budget:

    - $N=250$ is a count; no uncertainty. A 21 % miscount is implausible.
    - $A$: budgeted at 1.0 %. To explain a 30–35 % bias on its own,
      the cross-section would need to be 23 cm² instead of 16 cm² —
      far outside the budget.
    - $R_x$, $R_y$, $C$: budgeted below 0.2 %. Off-by-30 % is
      not credible for any of them in isolation. (The film capacitor
      $C$ is the loosest of the three by typical-tolerance arguments,
      but its measured value is closer to its nominal than the bias
      requires.)
    - $L'$: budgeted at $\sim$15 µm absolute, which is 3–14 % relative
      depending on the plate count. Two effects compound: (i) the
      caliper LSD itself is appreciable on a 0.10 mm plate, and (ii)
      copper plates may stick or include burrs that bias the *effective*
      gap larger than nominal. **A coherent +30 % bias on $L'$ would
      explain the entire $\mu_0$ shortfall** — and is the only entry
      whose required shift is comparable to its plausible systematic
      range.

    Conclusion: the dominant systematic is most plausibly the effective
    copper-plate gap $L'$ (geometric thickness vs effective magnetic
    gap including mounting/contact effects), with the core cross-section
    $A$ as a secondary candidate. $R_x$, $R_y$, $C$, and $N$ are
    individually too tightly constrained to be primary suspects.
    """)
    return


@app.cell
def _(
    B_max,
    MU0_THEO_U,
    RUNS,
    fits,
    fmt,
    mo,
    mu_peak,
    nsigma,
):
    _per_run = '\n'.join(
        f"* *{RUNS[sheet]['label']}* — "
        f"{fmt(f['mu0_exp'], r'T\,m\,A^{-1}')} "
        f"(${nsigma(f['mu0_exp'], MU0_THEO_U):.1f}\\,\\sigma$ below CODATA)."
        for sheet, f in fits.items()
    )

    mo.callout(
        mo.md(f"""
    ### Bottomline

    **Part A** — $\\mu_{{\\mathrm{{max}}}}/\\mu_0 =$ {fmt(mu_peak)}; saturation $B =$ {fmt(B_max, 'T')}
    (consistent with soft-iron saturation $\\sim 1.2$–$2.1\\,\\mathrm{{T}}$).

    **Part B** — $\\mu_0^{{\\mathrm{{exp}}}}$ from the runs:

    {_per_run}
    """),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
