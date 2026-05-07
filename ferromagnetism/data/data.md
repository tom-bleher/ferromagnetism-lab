# Ferromagnetism Experiment

Raw measurements from the $B$–$H$ curve and vacuum-permeability experiment.

## Apparatus parameters

| Symbol | Value | Meaning |
|---|---|---|
| $N$   | $250$                                  | primary / secondary turns |
| $L$   | $0.48~\text{m}$                        | Ampère-loop perimeter used in the processing |
| $R_x$ | $2.999~\Omega$                         | current-sense resistor ($\Delta V_x = I R_x$) |
| $R_y$ | $11.1~k\Omega$                     | integrator input resistor |
| $C$   | $2.01\times 10^{-5}~\text{F}$          | integrator capacitor |
| $A$   | $0.0016~\text{m}^2$                    | ring cross-section area |
| $\mu_0$ | $1.25663706127\times 10^{-6}~\text{T·m/A}$ | vacuum permeability (CODATA) |

## Derived quantities — peak-envelope magnetization curve (legacy sheet `virgin-curve`)

`proc.py` treats the voltage columns as authoritative and recalculates the
derived fields. The precomputed spreadsheet `H`, `B`, and `I` columns are
legacy full-span values: they are exactly twice the peak-amplitude values used
in the report and should not be used as processed data.

The scope readings are peak-to-peak extents of the hysteresis loop, so the
processing uses half of each recorded voltage. This gives the loop envelope,
not the full loop trajectory:

$$H = \frac{\Delta V_x \, N}{2 L \, R_x} \quad[\text{A/m}]$$

$$B = \frac{\Delta V_y \, R_y \, C}{2 N \, A} \quad[\text{T}]$$

$$\mu = \frac{B}{H} \quad[\text{T·m/A}], \qquad \mu_{\mathrm{r}} = \frac{\mu}{\mu_0}$$

$H$ uses $N$ as the *primary* turn count; $B$ uses $N$ as the *secondary* turn
count. The apparatus stores a single $N$ so the formulas assume
$N_\text{primary}=N_\text{secondary}$.

## $\mu_0$ extraction — `vacuum-permeability`

The final copper-gap widths used by `proc.py` are the measured `L' (mm)` rows
in `data.xlsx`: run `a` uses $0.35, 0.20, 0.10, 0.00~\mathrm{mm}$ and run
`b` uses $0.50, 0.45, 0.35, 0.20, 0.10, 0.00~\mathrm{mm}$. These workbook
values are authoritative for the analysis; a rough sketch-note value of
$1.3~\mathrm{mm}$ was not used.

Per row,

$$I = \frac{\Delta V_x}{2R_x}, \qquad B = \frac{\Delta V_y \, R_y \, C}{2 N \, A}.$$

Within each run $B$ is held constant by the experimenter. Ampère's law on the
closed magnetic path with an air gap of length $L'$ gives

$$N I = H_\text{iron} \, L_\text{iron} + \frac{B \, L'}{\mu_0},$$

so plotting $NI/B$ against $L'$ is linear with slope $1/\mu_0$.

Valid while the iron operates in the saturation regime
($H_\text{iron} L_\text{iron} \approx \text{const}$ across the sweep).

`proc.py` fits $NI/B$ directly against $L'$ using the per-row uncertainties
propagated from $\Delta V_x$, $\Delta V_y$, $R_x$, $R_y$, $C$, and $A$.
The slope of this fit is reported as $1/\mu_0$.

## Current-limit convention

The external AC ammeter was used only for safety monitoring against the guide's
current limit, not for data processing. The processing uses oscilloscope
peak-to-peak voltages, so the current amplitude inferred from the scope is

$$I_\text{peak}=\frac{\Delta V_x}{2R_x}.$$

Any spreadsheet `I (A)` columns are legacy full-span derived values and should
not be used as authoritative processed data; in particular, they do not define
the safety-ammeter convention.
