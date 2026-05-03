"""Python translation of scope_hysteresis_area.m."""
import numpy as np
import matplotlib.pyplot as plt


def scope_hysteresis_area(file_name, delta):
    a = np.loadtxt(file_name)
    a[:, 0] -= a[:, 0].min()
    a[:, 1] -= a[:, 1].min()
    x, y = a[:, 0], a[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    plt.plot(x, y)
    plt.grid(True)

    dx = delta
    N = int(np.ceil((x_max - x_min) / dx))
    new_x, upper, lower = [], [], []
    for i in range(1, N + 1):
        left = x_min + (i - 1) * dx
        right = x_min + i * dx
        mask = (x >= left) & (x < right)
        if not mask.any():
            raise ValueError("ERROR: The steps are too small")
        y_bin = y[mask]
        hi, lo = y_bin.max(), y_bin.min()
        edge = i <= 3 or i > N - 3
        if edge or abs(hi - lo) > 0.1:
            new_x.append((left + right) / 2)
            upper.append(hi)
            lower.append(lo)

    new_x = np.array(new_x)
    upper = np.array(upper)
    lower = np.array(lower)

    plt.plot(new_x, upper, 'r')
    plt.plot(new_x, lower, 'g')

    max_sec_derv = max(np.diff(lower, n=2).max(), np.diff(upper, n=2).max())
    area_error = max_sec_derv * (x_max - x_min) ** 3 / N ** 2
    area = np.trapezoid(upper, new_x) - np.trapezoid(lower, new_x)

    plt.text(x_min, (y_max + y_min) / 2,
             f'Area = {area} \\pm {area_error}', fontsize=15)
    plt.show()
    return area, area_error


if __name__ == '__main__':
    import sys
    scope_hysteresis_area(sys.argv[1], float(sys.argv[2]))
