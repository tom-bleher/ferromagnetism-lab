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

## Derived quantities — `virgin-curve`

`proc.py` treats the voltage columns as authoritative and recalculates the
derived fields; any precomputed spreadsheet columns are only a legacy check.

The scope readings are peak-to-peak values on the hysteresis loop, so the
processing uses half of each recorded voltage:

$$H = \frac{\Delta V_x \, N}{2 L \, R_x} \quad[\text{A/m}]$$

$$B = \frac{\Delta V_y \, R_y \, C}{2 N \, A} \quad[\text{T}]$$

$$\mu = \frac{B}{H} \quad[\text{T·m/A}], \qquad \mu_\text{rel} = \frac{\mu}{\mu_0}$$

$H$ uses $N$ as the *primary* turn count; $B$ uses $N$ as the *secondary* turn
count. The apparatus stores a single $N$ so the formulas assume
$N_\text{primary}=N_\text{secondary}$.

## $\mu_0$ extraction — `vacuum-permeability`

Per row,

$$I = \frac{\Delta V_x}{2R_x}, \qquad B = \frac{\Delta V_y \, R_y \, C}{2 N \, A}.$$

Within each run $B$ is held constant by the experimenter. Ampère's law on the
closed magnetic path with an air gap of length $L'$ gives

$$N I = H_\text{iron} \, L_\text{iron} + \frac{B \, L'}{\mu_0},$$

so plotting $\Delta V_x$ against $L'$ is linear with slope
$2R_x B / (N \mu_0)$. Solving for $\mu_0$,

$$\mu_0^\text{meas} = \frac{2R_x \, \langle B \rangle}{N \cdot \mathrm{slope}(\Delta V_x,\, L')}.$$

Valid while the iron operates in the saturation regime
($H_\text{iron} L_\text{iron} \approx \text{const}$ across the sweep).

`proc.py` instead fits the dimensionless ratio $r = \Delta V_x/\Delta V_y$
against $L'$ and folds the apparatus factor $K = N^2 A / (R_x R_y C)$ in
afterwards:

$$\mu_0^\text{meas} = \frac{1}{K \cdot \mathrm{slope}(r,\, L')}.$$

The two formulations agree when $\Delta V_y$ is held truly constant across
the sweep, which the data confirm. The ratio formulation has two practical
advantages: (a) the peak-to-peak factor of 2 cancels in the ratio, so the
extracted $\mu_0$ is invariant under the cursor convention; (b) folding $K$
in *after* the fit prevents the fully-correlated apparatus uncertainty from
being diluted as $\sqrt{N}$ independent samples by ODR.
