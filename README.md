
 <div align="center">
  <h2>Copula-Based Multivariate Dependence Modeling and Tail Risk Assessment in Financial Portfolios: Mathematical Foundations and Applications</h2>

  <p><i>Sudip Khadka</i></p>
  <p><i>Linear Statistical Modeling & Financial Risk Management</i></p>
  <p><i>University of Maryland, College Park</i></p>
</div>


<hr>



<h3>1. Motivation and Theoretical Foundation</h3>
Traditional Markowitz’s Modern Portfolio Theory relies on covariance-based risk measures and assume multivariate normal returns. However, asset returns often exhibit heavy tails, skewness, and asymmetric dependencies, violating these assumptions. Classical linear correlation measures fail to distinguish co-movements in gains versus losses, limiting their usefulness for extreme-event risk management.


Copula theory provides a rigorous mathematical framework to overcome these limitations leveraging Sklar’s theorem. By Sklar’s theorem any multivariate distribution with continuous marginals $F_1, \ldots, F_n$ can be decomposed into marginal distributions and a copula $C$ capturing the dependenceies:

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
I analyzed six years (2019–2025) of daily returns for a diversified portfolio comprising major equity indices such as NBI, IXIC, GSPC, RUT, DJI, and Bitcoin (BTC-USD).  All returns are transformed to log returns and are computed as:

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
$$


<p align="center">
<img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/log-price.png" alt="Screenshot 2024-12-09 at 2 17 15 PM" width="950" height="500">
 <br>
  <em>Figure 1: Six Years Log Price”</em>
</p>


To capture these distributional characteristics more accurately, I fitted several parametric distributions like Normal, Student-t, and Cauchy. The Student-t and Cauchy models provide a superior fit, particularly in the tails. To ensure robustness, I also employ empirical cumulative distribution functions (CDFs), which avoid restrictive parametric assumptions and offer improved tail modeling performance. The analysis reveals significant deviations from normality:

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



<h4>2.2. Vine Copulas</h4>

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
  <img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/plots/D-vine2D.png" alt="Bivariate D- Copula" width="45%" height="30%">
  <img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/plots/D-vine3D.png" alt="Trivariate D-Vine Copula" width="43%" height="30%">
</p>

<p align="center">Vine
  <em>Figure 4: (Left) Bivariate D-Vine Copula Fit — (Right) Trivariate D-Vine Copula Fit</em>
</p>

Model successfully captured the strong positive dependencies among traditional equity indices (NASDAQ, S&P 500, Dow Jones, Russell 2000), as evidenced by excellent overlap between simulated (blue) and original (orange) data in these pairwise plots. The model effectively reproduces both linear and non-linear dependence patterns across most variable pairs, with 2D plots showing good alignment in correlation structures and 3D plots demonstrating adequate capture of multivariate relationships. Bitcoin (BTC-USD) exhibits weaker and more scattered dependencies with traditional equity indices, which the D-vine captures with moderate success but shows some density concentration differences. The sequential pairs like ^IXIC vs ^GSPC and ^GSPC vs ^DJI demonstrate particularly strong model performance, reflecting the D-vine's strength in modeling adjacent variables in its path structure. However, the D-vine's sequential path structure may not optimally represent all complex dependencies, as it assumes relationships are primarily mediated through neighboring variables in the ordering, potentially missing direct dependencies between non-adjacent pairs and being sensitive to variable sequencing. For example, if Bitcoin (BTC-USD) and the Russell 2000 (^RUT) are not adjacent in the sequence, their direct relationship must be modeled through intermediate variables, which may explain why plots like ^RUT vs BTC-USD show noticeable density differences between simulated and original data despite both variables having strong individual relationships with other indices.

<h4>2.2.2. C-vines</h4>

For a Canonical Vine (C-vine) with root nodes $(1, 2, \ldots, d-1\)$, the joint density can be expressed as:

$$
f(x_1, \ldots, x_d) =
\prod_{i=1}^{d} f_i(x_i)
\cdot
\prod_{j=1}^{d-1} \prod_{i=j+1}^{d}
c_{j, i \mid 1, \ldots, j-1}
$$

where $(c_{j, i \mid 1, \ldots, j-1}\)$ represents a bivariate copula density describing the conditional dependence between variables $(X_j\) and \(X_i\)$ given the conditioning variables $(X_1, \ldots, X_{j-1}\)$.



**Structure of the C-vine**

- **Tree 1:** Star structure with a root node (e.g., node 1):  
  $((1,2), (1,3), \ldots, (1,d)\)$

- **Tree 2:** Star structure with a new root (e.g., node 2):  
  $((2,3 \mid 1), (2,4 \mid 1), \ldots, (2,d \mid 1)\)$

- **Tree \(k\):** Each subsequent tree has a new root and adds one more conditioning variable:  
  $((k, k+1 \mid 1, \ldots, k-1), (k, k+2 \mid 1, \ldots, k-1), \ldots\)$

Each tree in the C-vine has a central node connected to all others, forming a hierarchical starstructure. The Star-shaped structure with a central hub (GSP) represents market-driven effects.


<p align="center">
  <img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/plots/Cvine2D.png" alt="Bivariate C-Vine Copula" width="45%" height="30%">
  <img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/plots/Cvine3D.png" alt="Trivariate C-Vine Copula" width="43%" height="30%">
</p>

<p align="center">Vine
  <em>Figure 4: (Left) Bivariate C-Vine Copula Fit — (Right) Trivariate C-Vine Copula Fit</em>
</p>

The model's star structure, where each tree has a central node connecting to all others appears effective for modeling relationships where one variable acts as a hub. This phenomenon is visible in plots involving major indices that show consistent correlation patterns. However, the C-vine's structure imposes a rigid hierarchical dependency assumption where all relationships must flow through central nodes at each level. For example, in plots like ^IXIC vs ^RUT or BTC-USD vs ^DJI, the visible dispersion between simulated and original data suggests that forcing these relationships to be mediated through a central variable (rather than allowing direct connections) may inadequately capture the true dependence structure. This limitation becomes evident when variables like Bitcoin have fundamentally different market drivers than traditional equity indices and may not naturally relate through the same hub variables.

<h4>2.2.3. R-vines</h4>

Regular vines (R-vine) are the most general class of vine copulas, encompassing both C-vines and D-vines as special cases.The joint density for a $d$-dimensional continuous random vector is expressed as:

$$
f(x_1, \ldots, x_d) =
\prod_{i=1}^{d} f_i(x_i)
\cdot
\prod_{k=1}^{d-1}
\prod_{e \in E_k}
c_e
$$

where:
- $( E_k \)$ — the set of edges in tree $( T_k \)$,
- $( c_e \)$ — the copula density associated with edge $( e \)$, representing the dependence between variables connected by that edge (possibly conditional on others).

**Proximity Condition**

In tree $( T_j \)$, two nodes can be connected only if their corresponding edges in the previous tree $( T_{j-1} \)$ share a common node. This ensures the hierarchical structure and conditional dependence consistency of the R-vine construction.

**Our observation:**

  <p align="center">
  <img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/plots/Rvine2D.png" alt="Bivariate R-Vine Copula" width="45%" height="30%">
  <img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/plots/Rvine3D.png" alt="Trivariate R-Vine Copula" width="43%" height="30%">
</p>

<p align="center">Vine
  <em>Figure 4: (Left) Bivariate R-Vine Copula Fit — (Right) Trivariate R-Vine Copula Fit</em>
</p>

The model successfully captures both strong dependencies among equity indices (^IXIC vs ^GSPC, ^GSPC vs ^DJI) and weaker, more complex relationships involving Bitcoin, with particularly improved performance visible in plots like ^RUT vs BTC-USD vs ^DJI and ^NBI vs BTC-USD where the density distributions align more closely than in the D-vine results. The 3D plots reveal that the R-vine effectively models conditional dependencies and multivariate structures without being constrained to sequential paths, allowing direct modeling of relationships between any variable pairs regardless of their position in the structure. However, the R-vine's increased flexibility comes at the cost of higher computational complexity and potential overfitting risk—for example, in plots like ^RUT vs BTC-USD vs ^DJI, while the simulated data closely matches the original's dense clustering patterns, this tight fit may indicate the model is capturing sample-specific noise rather than generalizable market relationships, particularly problematic given Bitcoin's high volatility and relatively short history compared to traditional equity indices, which could lead to poor performance when market dynamics shift.

<h4>2.3. Clayton Copula</h4>
The Clayton copula emphasizes lower-tail dependence and is given by:

$$
C_{\theta}(u_1, \ldots, u_n) =
\left( \sum_{i=1}^{n} u_i^{-\theta} - n + 1 \right)^{-1/\theta}, \quad \theta > 0
$$

Fitted parameters confirm stronger lower-tail clustering, consistent with the empirical evidence of joint market crashes. Simulations based on this copula reproduce joint crash behavior observed in real data.

<h3>3. Value at Risk (VaR) Analysis</h3>

We estimate the 5-day Value at Risk (VaR) at a 99% confidence level for an equally-weighted portfolio of six financial assets with an initial value of $100,000. We formally define VaR as:


$$
\text{VaR}_{\alpha} = -\inf \{ x \in \mathbb{R} : F_R(x) \geq 1 - \alpha \}
$$

where $( F_R \)$ is the cumulative distribution function (CDF) of the portfolio returns $( R \)$, and  
$( \alpha = 0.99 \)$ represents the 99% confidence level. We construct an equally-weighted portfolio where each asset receives equal dollar allocation:

$$
w_i = \frac{n \cdot P_i}{V_0}
$$

where:

- $V_0$ — initial portfolio value  
- $n$ — number of assets  
- $P_i$ — current price of asset $( i \)$ 
- $w_i$ — number of shares (portfolio weight) allocated to asset $( i \)$

<h4>3.1. Simulation Approach</h4>
Our VaR estimation employs six different methods to model portfolio return distributions:

<h4>3.1.1.  Copula-Based Methods (Gaussian, Clayton, D-vine, C-vine, R-vine):</h4>

For each copula model we:

- generate $( N \)$ realizations from the fitted copula $( C_{\theta} \)$:

$$
\mathbf{U}_j = (U_{1,j}, \ldots, U_{n,j}) \sim C_{\theta}(\mathbf{u}), \quad j = 1, \ldots, N
$$

- Transform to asset returns using fitted marginal distributions:

$$
\tilde{r}_{i,j} = \Phi^{-1}(U_{i,j}; \mu_i, \sigma_i)
$$

where $( \Phi^{-1} \)$ denotes the inverse CDF (quantile function) of the chosen marginal    
   distribution for asset $( i \)$.

- Compute Portfolio Returns per Scenario:

$$
R_j = \sum_{i=1}^{n} \tilde{r}_{i,j} \cdot w_i
$$

- Aggregate into 5-Day Rolling Returns:

$$
R_j^{(5)} = \sum_{t=0}^{4} R_{j+t}
$$

- Estimate the 99% VaR:

$$
\text{VaR}_{0.99} = -Q_{0.01}(R^{(5)}) \times V_0
$$

   where $( Q_{0.01}(\cdot) \)$ denotes the 1st percentile (empirical quantile).

<h4>3.1.2.  Covariance Method (Parametric Approach):</h4>
We will use historical portfolio returns directly:

- Compute historical portfolio returns:

 $$
R_t = \sum_{i=1}^{n} r_{i,t} \cdot w_i
$$


- Compute 5-day rolling returns from historical data.
- Estimate VaR empirically:


$$
\text{VaR}_{0.99} = - \text{Percentile}(R^{(5)}, 0.01) \times V_0
$$


<p align="center">
<img src="https://github.com/sudkc37/Multivariate-Dependence-Modeling/blob/master/plots/VaR.png" alt="Screenshot 2024-12-09 at 2 17 15 PM" width="950" height="500">
 <br>
  <em>Figure 1: Portfolio VaR</em>
</p>
