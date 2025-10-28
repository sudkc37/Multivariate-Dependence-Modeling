
 <div align="center">
  <h2>Copula-Based Multivariate Dependence Modeling and Tail Risk Assessment in Financial Portfolios: Mathematical Foundations and Applications</h2>

  <p><i>Sudip Khadka</i></p>
  <p><i>Linear Statistical Modeling & Financial Risk Management</i></p>
  <p><i>University of Maryland, College Park</i></p>
</div>


<hr>



<h3>1. Motivation and Theoretical Foundation</h3>
Traditional Markowitz’s Modern Portfolio Theory relies on covariance-based risk measures and assume multivariate normal returns. However, asset returns often exhibit heavy tails, skewness, and asymmetric dependencies, violating these assumptions. Classical linear correlation measures fail to distinguish co-movements in gains versus losses, limiting their usefulness for extreme-event risk management.


Copula theory provides a rigorous mathematical framework to overcome these limitations leveraging Sklar’s theorem. By Sklar’s theorem any multivariate distribution with continuous marginals $F_1, \ldots, F_n$ can be decomposed into marginal distributions and a copula $C$ capturing dependence:

$$
F(x_1, \ldots, x_n) = C(F_1(x_1), \ldots, F_n(x_n)), \quad C: [0,1]^n \rightarrow [0,1].
$$

This decomposition isolates the dependence structure from marginal behavior which allows independent modeling of each component.  
The transformation

$$
u_i = F_i(x_i)
$$

maps returns to uniform $[0,1]$ variables, where copulas operate naturally.  Tail dependence measures $\lambda_L$ and $\lambda_U$ which further quantify the probability of joint extreme losses or gains that provides a mathematically precise metric for systemic risk.


<h3>1. Data and Marginal Analysis</h3>
I analyze six years (2019–2025) of daily returns for a diversified portfolio comprising major equity indices such as NBI, IXIC, GSPC, RUT, DJI, and Bitcoin (BTC-USD).  All returns are transformed to log returns are computed as:

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
$$


<p align="center">
<img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/log-price.png" alt="Screenshot 2024-12-09 at 2 17 15 PM" width="950" height="500">
 <br>
  <em>Figure 1: Six Years Log Price”</em>
</p>


To capture these distributional characteristics more accurately, I fit several parametric distributions like Normal, Student-t, and Cauchy. The Student-t and Cauchy models provide a superior fit, particularly in the tails. To ensure robustness, I also employ empirical cumulative distribution functions (CDFs), which avoid restrictive parametric assumptions and offer improved tail modeling performance. The analysis reveals significant deviations from normality:

- **Excess kurtosis:** All series are leptokurtic exhibiting heavier tails than Gaussian distributions.  
- **Skewness:** Most series show negative skewness indicating a higher likelihood of extreme losses.  
- **Non-normality:** Visual inspection confirms clear departures from the bell-curve shape.


<p align="center">
<img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/descriptions.png" alt="Screenshot 2024-12-09 at 2 17 15 PM" width="950" height="500">
 <br>
  <em>Figure 2: Marginal Distributions</em>
</p>




<h3>2. Copula Models and Estimation</h3>
<h4>2.1. Gaussian Copula</h4>
The Gaussian copula is defined as:

$$
C_{\Sigma}(u_1, \ldots, u_n) = \Phi_{\Sigma}\big(\Phi^{-1}(u_1), \ldots, \Phi^{-1}(u_n)\big)
$$

where, 

- $\Phi_{\Sigma}$ — multivariate normal cumulative distribution function (CDF) with correlation matrix $\Sigma$  
- $\Sigma$ — correlation matrix  
- $\Phi^{-1}$ — inverse standard normal CDF  
- $u_i \in [0,1]$ — uniform marginals



<p align="center">
  <img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/plots/D-vine2D.png" alt="Bivariate D-Vine Copula" width="45%" height="30%">
  <img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/plots/Gausioan-3D.png" alt="Trivariate Gaussian Copula" width="43%" height="30%">
</p>

<p align="center">
  <em>Figure 3: (Left) Bivariate Gaussian Copula Fit — (Right) Trivariate Gaussian Copula Fit</em>
</p>




The strong positive correlations among traditional equity indices (NBI, IXIC, GSPC, RUT, DJI) which form tight ellipsoidal clusters in both 2D and 3D visualizations, with simulated data (orange) closely overlapping the original observations (blue). The model also correctly reproduces the weak dependence between BTC-USD and traditional assets, generating appropriately scattered uniform distributions for these pairs. While computationally tractable, it exhibits tail independence:

$$
\lambda_L = \lambda_U = 0
$$

This limitation leads to underestimation of joint extreme events. Although maximum likelihood estimation (MLE) reproduces observed correlations, it fails to capture clustering in extreme losses.

<h4>2.2. Clayton Copula</h4>
The Clayton copula emphasizes lower-tail dependence and is given by:

$$
C_{\theta}(u_1, \ldots, u_n) =
\left( \sum_{i=1}^{n} u_i^{-\theta} - n + 1 \right)^{-1/\theta}, \quad \theta > 0
$$

Fitted parameters confirm stronger lower-tail clustering, consistent with the empirical evidence of joint market crashes. Simulations based on this copula reproduce joint crash behavior observed in real data.

<h4>2.3. Vine Copulas</h4>

High-dimensional dependence structures are modeled using vine copulas which decompose multivariate dependencies into bivariate components. For a $d$-dimensional continuous random vector $(X_1, X_2, \ldots, X_d)$ with marginal densities $f_i(x_i)$ and corresponding cumulative distribution functions $F_i(x_i)$, the joint density function can be decomposed into a product of its marginal densities and a series of conditional bivariate copula densities as follows:

$$
f(x_1, \ldots, x_d)
= \prod_{i=1}^{d} f_i(x_i)
  \cdot
  \prod_{j=1}^{d-1} \prod_{i=1}^{d-j}
  c_{i,\,i+j \mid i+1,\,\ldots,\,i+j-1}
  \left(
    F(x_i \mid x_{i+1}, \ldots, x_{i+j-1}),
    F(x_{i+j} \mid x_{i+1}, \ldots, x_{i+j-1})
  \right),
$$

where $c_{i,j \mid D}$ denotes a bivariate copula density describing the conditional dependence between variables $X_i$ and $X_j$ given the conditioning set $D$.

<h4>2.3.1. D-vines</h4>

It is used for sequential structure capturing local dependencies. For a D-vine with ordering $(1, 2, \ldots, d)$, the joint density can be decomposed as:

$$
f(x_1, \ldots, x_d) =
\prod_{i=1}^{d} f_i(x_i)
\cdot
\prod_{j=1}^{d-1} \prod_{i=1}^{d-j} 
c_{i, i+j \mid i+1, \ldots, i+j-1}
$$

where $(c_{i,j \mid D})$ denotes a **bivariate copula density** describing the conditional dependence between $(X_i)$ and $(X_j)$ given the conditioning set $D$.



**Structure of the D-vine**

- **Tree 1:** Sequential pairs  
$((1,2), (2,3), (3,4), \ldots, (d-1,d))$

- **Tree 2:** Pairs conditioned on the variable in between  
$((1,3 \mid 2), (2,4 \mid 3), \ldots, (d-2,d \mid d-1))$

- **Tree \(k\):** Pairs with \(k-1\) conditioning variables  
$((1, k+1 \mid 2, \ldots, k), (2, k+2 \mid 3, \ldots, k+1), \ldots)$

<p align="center">
  <img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/plots/gausioan.png" alt="Bivariate Gaussian Copula" width="45%" height="30%">
  <img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/plots/D-vine3D.png" alt="Trivariate D-Vine Copula" width="43%" height="30%">
</p>

<p align="center">
  <em>Figure 3: (Left) Bivariate Gaussian Copula Fit — (Right) Trivariate Gaussian Copula Fit</em>
</p>



- **C-vine:** Star-shaped structure with a central hub (GSP), representing market-driven effects.

- **R-vine:** Fully flexible, data-driven structure capturing asymmetric and non-nested dependencies.

Simulations from these vine copulas closely reproduce joint distributions across multiple projections, effectively modeling complex market interactions.

