# Computational Physics --- Homework 2

This repository contains my solutions for Computational Physics Homework
2.

## Assignment

The homework contains three problems:

1.  **Exercise 6.11 --- Overrelaxation**
    -   Derive the error estimate for the overrelaxation method.
    -   Solve (x = 1-e\^{-2x}) using ordinary relaxation.
    -   Repeat the calculation using overrelaxation.
    -   Compare the number of iterations for different values of the
        overrelaxation parameter (`\omega`{=tex}).
2.  **Exercise 6.13 --- Wien's Displacement Constant**
    -   Derive the nonlinear equation (5e\^{-x}+x-5=0).
    -   Solve the equation numerically using binary search to an
        accuracy of (10\^{-6}).
    -   Calculate Wien's displacement constant.
    -   Use the peak wavelength of the Sun's spectrum to estimate its
        surface temperature.
3.  **Multidimensional Gradient Descent / Schechter Function Fit**
    -   Implement multidimensional gradient descent using numerical
        derivatives.
    -   Minimize the chi-squared statistic for a Schechter-function
        model of the COSMOS galaxy stellar mass function.
    -   Test the robustness of the fit using multiple initial positions
        in parameter space.
    -   Plot chi-squared as a function of iteration.
    -   Compare the best-fit Schechter function with the supplied data.


## Requirements

The Python program uses:

-   Python 3
-   NumPy
-   Matplotlib

Install the required packages with:

``` bash
python3 -m pip install numpy matplotlib
```

## Running the Code

Place `homework2.py` and `smf_cosmos.dat` in the same directory and run:

``` bash
python3 homework2.py
```

The program prints the numerical results for all three questions and
generates:

``` text
chi2_vs_step.png
schechter_fit.png
```

## Main Numerical Results

### Exercise 6.11

For

\[ x=1-e\^{-2x}, \]

the nonzero solution is approximately

\[ x `\approx 0.7968126`{=tex}. \]

Ordinary relaxation requires more iterations than overrelaxation. Values
of (`\omega`{=tex}) near (0.68) give particularly rapid convergence for
this problem.

### Exercise 6.13

Binary search gives the nontrivial positive solution

\[ x `\approx 4.965114`{=tex}. \]

The corresponding Wien displacement constant is

\[ b `\approx 2.89777`{=tex}`\times10`{=tex}\^{-3} {`\rm m\,K`{=tex}}.
\]

For a solar peak wavelength of (502,{`\rm nm`{=tex}}), the estimated
surface temperature is approximately

\[
T\_`\odot `{=tex}`\approx 5.77`{=tex}`\times10`{=tex}\^3 {`\rm K`{=tex}}.
\]

### Schechter-Function Fit

The third problem minimizes

\[ `\chi`{=tex}\^2 = `\sum`{=tex}\_i `\left`{=tex}(
`\frac{n_i-n_{\rm model}(M_i)}{\sigma_i}`{=tex} `\right`{=tex})\^2 \]

using numerical finite-difference derivatives and multidimensional
gradient descent.

The optimization variables are

\[
`\left`{=tex}(`\log`{=tex}\_{10}`\phi`{=tex}\^\*,,`\log`{=tex}\_{10}M\^\*,,`\alpha`{=tex}`\right`{=tex}).
\]

Several distinct initial parameter choices are included in the code to
test whether the numerical solution converges to the same minimum.


## Author

Amelia Stevens\
Computational Physics, Fall 2026
