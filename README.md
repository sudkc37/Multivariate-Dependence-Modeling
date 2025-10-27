
 <div align="center">
  <h2>Copula-Based Multivariate Dependence Modeling and Tail Risk Assessment in Financial Portfolios: Mathematical Foundations and Applications</h2>

  <p><i>Sudip Khadka</i></p>
</div>


<hr>



<h3>1. Motivation and Theoretical Foundation</h3>
Traditional Markowitz’s Modern Portfolio Theory relies on covariance-based risk measures and assume multivariate normal returns. However, asset returns often exhibit heavy tails, skewness, and asymmetric dependencies, violating these assumptions. Classical linear correlation measures fail to distinguish co-movements in gains versus losses, limiting their usefulness for extreme-event risk management.


Copula theory provides a rigorous mathematical framework to overcome these limitations leveraging Sklar’s theorem. By Sklar’s theorem any multivariate distribution with continuous marginals $F_1, \ldots, F_n$ can be decomposed into marginal distributions and a copula $C$ capturing dependence:

$$
F(x_1, \ldots, x_n) = C(F_1(x_1), \ldots, F_n(x_n)), \quad C: [0,1]^n \rightarrow [0,1].
$$

This decomposition isolates the dependence structure from marginal behavior, allowing independent modeling of each component.  
The transformation

$$
u_i = F_i(x_i)
$$

maps returns to uniform $[0,1]$ variables, where copulas operate naturally.  Tail dependence measures $\lambda_L$ and $\lambda_U$ which further quantify the probability of joint extreme losses or gains that provides a mathematically precise metric for systemic risk.


<h4>1.1 Motivation and Problem Context</h4>
