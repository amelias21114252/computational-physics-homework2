import math
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# Question 1: Exercise 6.11 - Overrelaxation
# ============================================================

def f_relax(x, c=2.0):
    return 1.0 - math.exp(-c * x)


def fp_relax(x, c=2.0):
    return c * math.exp(-c * x)


def solve_relax(c=2.0, x=1.0, omega=0.0, tol=1e-6):
    for n in range(1, 10001):
        xn = (1.0 + omega) * f_relax(x, c) - omega * x
        q = (1.0 + omega) * fp_relax(x, c) - omega

        if abs(q) > 1e-15 and abs(1.0 - 1.0 / q) > 1e-15:
            err = abs((x - xn) / (1.0 - 1.0 / q))
        else:
            err = abs(xn - x)

        x = xn
        if err < tol:
            return x, n, err

    raise RuntimeError("No convergence")


reference_root = solve_relax(omega=0.685, tol=1e-13)[0]
reference_slope = fp_relax(reference_root)
optimal_omega = reference_slope / (1.0 - reference_slope)

print("QUESTION 1")
print("reference root =", reference_root)
print("locally optimal omega =", optimal_omega)
for w in [0.0, 0.5, 0.68, 0.685]:
    x, n, e = solve_relax(omega=w)
    print(w, x, n, e)


# ============================================================
# Question 2: Exercise 6.13 - Wien displacement
# ============================================================

h = 6.62607015e-34
c = 2.99792458e8
kB = 1.380649e-23


def fw(x):
    return 5.0 * math.exp(-x) + x - 5.0


def bisect(a, b, tol=1e-6):
    fa = fw(a)
    while b - a > tol:
        m = (a + b) / 2.0
        fm = fw(m)
        if fa * fm <= 0.0:
            b = m
        else:
            a = m
            fa = fm
    return (a + b) / 2.0


xw = bisect(1.0, 10.0)
bw = h * c / (kB * xw)
Ts = bw / (502e-9)

print("\nQUESTION 2")
print("x =", xw)
print("b =", bw, "m K")
print("T_sun =", Ts, "K")


# ============================================================
# Question 3A: General multidimensional numerical gradient descent
# ============================================================

def numerical_gradient(func, theta, hstep=1e-5):
    """Centered numerical gradient of a scalar function."""
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)

    for j in range(len(theta)):
        hj = hstep * max(1.0, abs(theta[j]))
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += hj
        minus[j] -= hj
        grad[j] = (func(plus) - func(minus)) / (2.0 * hj)

    return grad


def gradient_descent(func, start, initial_step=0.1, max_iter=20000,
                     value_tol=1e-12, grad_tol=1e-14):
    """Normalized-gradient descent with a backtracking line search."""
    theta = np.asarray(start, dtype=float).copy()
    values = [func(theta)]
    path = [theta.copy()]

    for _ in range(max_iter):
        grad = numerical_gradient(func, theta)
        norm = np.linalg.norm(grad)

        if norm < grad_tol:
            break

        direction = -grad / norm
        step = initial_step

        while step > 1e-12:
            trial = theta + step * direction
            trial_value = func(trial)

            if np.isfinite(trial_value) and trial_value < values[-1]:
                break
            step *= 0.5

        if step <= 1e-12:
            break

        theta = trial
        path.append(theta.copy())
        values.append(trial_value)

        if abs(values[-2] - values[-1]) < value_tol:
            break

    return theta, np.asarray(values), np.asarray(path)


# ------------------------------------------------------------
# Test on f(x,y) = (x-2)^2 + (y-2)^2
# ------------------------------------------------------------

def test_function(theta):
    x, y = theta
    return (x - 2.0) ** 2 + (y - 2.0) ** 2


test_start = np.array([-1.0, 5.0])
test_solution, test_values, test_path = gradient_descent(
    test_function,
    test_start,
    initial_step=0.2,
)

print("\nQUESTION 3: GRADIENT-DESCENT TEST")
print("start =", test_start)
print("numerical minimum =", test_solution)
print("exact minimum = [2, 2]")
print("f(minimum) =", test_function(test_solution))

x_grid = np.linspace(-2.0, 6.0, 300)
y_grid = np.linspace(-2.0, 6.0, 300)
X, Y = np.meshgrid(x_grid, y_grid)
Z = (X - 2.0) ** 2 + (Y - 2.0) ** 2

plt.figure(figsize=(7, 6))
contours = plt.contour(X, Y, Z, levels=20)
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(
    test_path[:, 0],
    test_path[:, 1],
    "o-",
    markersize=4,
    label="Gradient-descent path",
)
plt.plot(
    test_path[0, 0],
    test_path[0, 1],
    "s",
    markersize=8,
    label="Initial point",
)
plt.plot(2.0, 2.0, "*", markersize=14, label="Exact minimum $(2,2)$")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(r"Numerical Gradient Descent on $f(x,y)=(x-2)^2+(y-2)^2$")
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "gradient_descent_test.png", dpi=200, bbox_inches="tight")
plt.close()


# ============================================================
# Question 3B: Schechter-function fit
# ============================================================
# The supplied assignment data file must be in the same directory.

# Columns: log10(galaxy mass), measured mass function, uncertainty.
# The mass ratio is 10**(logM - log_M_star).
data = np.loadtxt(BASE_DIR / "smf_cosmos.dat")
logM, nobs, sigma = data.T


def schechter(theta, x=None):
    if x is None:
        x = logM

    log_phi_star, log_M_star, alpha = theta
    ratio = 10.0 ** (x - log_M_star)

    return (
        np.log(10.0)
        * 10.0 ** log_phi_star
        * ratio ** (alpha + 1.0)
        * np.exp(-ratio)
    )


def chi2(theta):
    model = schechter(theta)
    return np.sum(((nobs - model) / sigma) ** 2)


starts = [
    [-2.3, 11.0, -1.2],
    [-2.0, 10.8, -1.0],
    [-3.0, 11.2, -1.5],
]

solutions = []
histories = []
paths = []

print("\nQUESTION 3: SCHECHTER FIT")

for start in starts:
    solution, history, path = gradient_descent(
        chi2,
        start,
        initial_step=0.1,
    )
    solutions.append(solution)
    histories.append(history)
    paths.append(path)
    print("start", start, "->", solution, "chi2 =", history[-1])

best_index = int(np.argmin([chi2(theta) for theta in solutions]))
best = solutions[best_index]

log_phi_star, log_M_star, alpha = best
phi_star = 10.0 ** log_phi_star
M_star = 10.0 ** log_M_star

print("best =", best)
print("phi* =", phi_star)
print("M* =", M_star)
print("chi2 =", chi2(best))


# Plot chi^2 versus gradient-descent step for all three starts.
plt.figure(figsize=(7, 5))
for start, history in zip(starts, histories):
    plt.semilogy(
        np.arange(len(history)),
        history,
        label="(" + ", ".join(f"{value:.1f}" for value in start) + ")",
    )

plt.xlabel("Gradient-descent step $i$")
plt.ylabel(r"$\chi^2$")
plt.legend(title=r"Start $(\log_{10}\phi^*, \log_{10}M^*, \alpha)$", fontsize=9)
plt.tight_layout()
plt.savefig(BASE_DIR / "chi2_vs_step.png", dpi=200, bbox_inches="tight")
plt.close()


# Plot the best-fit Schechter function and COSMOS measurements.
grid = np.linspace(logM.min(), logM.max(), 400)
grid_model = schechter(best, x=grid)

plt.figure(figsize=(7, 5))
plt.errorbar(
    10.0 ** logM,
    nobs,
    yerr=sigma,
    fmt="o",
    capsize=3,
    label="COSMOS data",
)
plt.loglog(
    10.0 ** grid,
    grid_model,
    label="Best-fit Schechter function",
)
plt.xlabel(r"$M_{\rm gal}$")
plt.ylabel(r"$n(M_{\rm gal})$")
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "schechter_fit.png", dpi=200, bbox_inches="tight")
plt.close()


# Export the per-start results; the LaTeX writeup reads this generated table.
rows = []
for start, solution, history in zip(starts, solutions, histories):
    rows.append([
        *start, *solution, 10.0 ** solution[0], 10.0 ** solution[1],
        history[-1], len(history) - 1,
    ])

with (BASE_DIR / "schechter_results.csv").open("w", newline="") as handle:
    csv_writer = csv.writer(handle)
    csv_writer.writerow([
        "start_log10_phi", "start_log10_M", "start_alpha",
        "log10_phi", "log10_M", "alpha", "phi_star", "M_star",
        "chi_squared", "accepted_steps",
    ])
    csv_writer.writerows(rows)

table_lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\small",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Starting point & $\phi^*$ & $M^*$ & $\alpha$ & $\chi^2$\\",
    r"\midrule",
]
for start, solution, history in zip(starts, solutions, histories):
    label = "(" + ",".join(f"{value:.1f}" for value in start) + ")"
    phi = 10.0 ** solution[0]
    mass = 10.0 ** solution[1]
    table_lines.append(
        f"${label}$ & ${phi * 1e3:.6f}\\times10^{{-3}}$ & "
        f"${mass / 1e10:.6f}\\times10^{{10}}$ & "
        f"${solution[2]:.6f}$ & ${history[-1]:.6f}$" + r"\\"
    )
table_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Schechter-fit parameters and objective values from each initial guess.}",
    r"\label{tab:starts}",
    r"\end{table}",
])
(BASE_DIR / "schechter_results.tex").write_text("\n".join(table_lines) + "\n")
print("Saved all figures, schechter_results.csv, and schechter_results.tex in", BASE_DIR)
