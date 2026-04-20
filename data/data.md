# Ferromagnetism Experiment

Raw measurements from the $B$–$H$ curve and vacuum-permeability experiment.

## Apparatus parameters

| Symbol | Value | Meaning |
|---|---|---|
| $N$   | $250$                                  | primary / secondary turns |
| $L$   | $0.48~\text{m}$                        | mean Ampère-loop length of the iron ring |
| $R_x$ | $2.999~\Omega$                         | current-sense resistor ($\Delta V_x = I R_x$) |
| $R_y$ | $11.1~k\Omega$                     | integrator input resistor |
| $C$   | $2.01\times 10^{-5}~\text{F}$          | integrator capacitor |
| $A$   | $0.0016~\text{m}^2$                    | ring cross-section area |
| $\mu_0$ | $1.25663706127\times 10^{-6}~\text{T·m/A}$ | vacuum permeability (CODATA) |

## Derived quantities — `virgin-curve`

$$H = \frac{\Delta V_x \, N}{L \, R_x} \quad[\text{A/m}]$$

$$B = \frac{\Delta V_y \, R_y \, C}{N \, A} \quad[\text{T}]$$

$$\mu = \frac{B}{H} \quad[\text{T·m/A}], \qquad \mu_\text{rel} = \frac{\mu}{\mu_0}$$

$H$ uses $N$ as the *primary* turn count; $B$ uses $N$ as the *secondary* turn
count. The apparatus stores a single $N$ so the formulas assume
$N_\text{primary}=N_\text{secondary}$.

## $\mu_0$ extraction — `vacuum-permeability`

Per row,

$$I = \frac{\Delta V_x}{R_x}, \qquad B = \frac{\Delta V_y \, R_y \, C}{N \, A}.$$

Within each run $B$ is held constant by the experimenter. Ampère's law on the
closed magnetic path with an air gap of length $L'$ gives

$$N I = H_\text{iron} \, L_\text{iron} + \frac{B \, L'}{\mu_0},$$

so plotting $\Delta V_x$ against $L'$ is linear with slope
$R_x B / (N \mu_0)$. Solving for $\mu_0$,

$$\mu_0^\text{meas} = \frac{R_x \, \langle B \rangle}{N \cdot \mathrm{slope}(\Delta V_x,\, L')}.$$

Valid while the iron operates below saturation
($H_\text{iron} L_\text{iron} \approx \text{const}$ across the sweep).
