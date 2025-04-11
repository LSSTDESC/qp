# Parameterizations

The pages below provide an overview of the five main supported `qp` parameterizations:

```{toctree}
:titlesonly:

hist.md
interp.md
irregularinterp.md
mixmod.md
quant.md

```

There exist additional parameterizations, however they are in various stages of completeness and are not guaranteed to work with all of the functions described in the documentation. For more information about these parameterizations see the parameterization [API documentation](#qp.parameterizations.hist.hist.hist_gen).

## SciPy stats parameterizations

`qp` also incorporates all of the continuous distributions from `scipy.stats` as parameterizations, allowing you to create `Ensembles` parameterized by these distributions. The main limitations on these parameterizations is that they cannot be converted to, as they have no conversion functions. Additionally, their `create_ensemble()` method takes a data dictionary instead of specific arguments, as shown in [the Cookbook](../cookbook.md#creating-an-ensemble-from-a-qp.stats-distribution).

You can access these parameterizations from the `qp.stats` module, as shown below:

```{doctest}

>>> import qp
>>> ens_n = qp.stats.norm.create_ensemble(dict(loc=np.array([0,1]),scale=np.array([0.25,0.5])))
>>> ens_n
Ensemble(the_class=norm,shape=(2,1))

```

The available parameterizations are listed below, with links to their SciPy documentation for more information.

| Function           | Description                                                                  |
| ------------------ | ---------------------------------------------------------------------------- |
| alpha              | An alpha continuous random variable.                                         |
| anglit             | An anglit continuous random variable.                                        |
| arcsine            | An arcsine continuous random variable.                                       |
| argus              | Argus distribution                                                           |
| beta               | A beta continuous random variable.                                           |
| betaprime          | A beta prime continuous random variable.                                     |
| bradford           | A Bradford continuous random variable.                                       |
| burr               | A Burr (Type III) continuous random variable.                                |
| burr12             | A Burr (Type XII) continuous random variable.                                |
| cauchy             | A Cauchy continuous random variable.                                         |
| chi                | A chi continuous random variable.                                            |
| chi2               | A chi-squared continuous random variable.                                    |
| cosine             | A cosine continuous random variable.                                         |
| crystalball        | Crystalball distribution                                                     |
| dgamma             | A double gamma continuous random variable.                                   |
| dpareto_lognorm    | A double Pareto lognormal continuous random variable.                        |
| dweibull           | A double Weibull continuous random variable.                                 |
| erlang             | An Erlang continuous random variable.                                        |
| expon              | An exponential continuous random variable.                                   |
| exponnorm          | An exponentially modified Normal continuous random variable.                 |
| exponweib          | An exponentiated Weibull continuous random variable.                         |
| exponpow           | An exponential power continuous random variable.                             |
| f                  | An F continuous random variable.                                             |
| fatiguelife        | A fatigue-life (Birnbaum-Saunders) continuous random variable.               |
| fisk               | A Fisk continuous random variable.                                           |
| foldcauchy         | A folded Cauchy continuous random variable.                                  |
| foldnorm           | A folded normal continuous random variable.                                  |
| genlogistic        | A generalized logistic continuous random variable.                           |
| gennorm            | A generalized normal continuous random variable.                             |
| genpareto          | A generalized Pareto continuous random variable.                             |
| genexpon           | A generalized exponential continuous random variable.                        |
| genextreme         | A generalized extreme value continuous random variable.                      |
| gausshyper         | A Gauss hypergeometric continuous random variable.                           |
| gamma              | A gamma continuous random variable.                                          |
| gengamma           | A generalized gamma continuous random variable.                              |
| genhalflogistic    | A generalized half-logistic continuous random variable.                      |
| genhyperbolic      | A generalized hyperbolic continuous random variable.                         |
| geninvgauss        | A Generalized Inverse Gaussian continuous random variable.                   |
| gibrat             | A Gibrat continuous random variable.                                         |
| gompertz           | A Gompertz (or truncated Gumbel) continuous random variable.                 |
| gumbel_r           | A right-skewed Gumbel continuous random variable.                            |
| gumbel_l           | A left-skewed Gumbel continuous random variable.                             |
| halfcauchy         | A Half-Cauchy continuous random variable.                                    |
| halflogistic       | A half-logistic continuous random variable.                                  |
| halfnorm           | A half-normal continuous random variable.                                    |
| halfgennorm        | The upper half of a generalized normal continuous random variable.           |
| hypsecant          | A hyperbolic secant continuous random variable.                              |
| invgamma           | An inverted gamma continuous random variable.                                |
| invgauss           | An inverse Gaussian continuous random variable.                              |
| invweibull         | An inverted Weibull continuous random variable.                              |
| irwinhall          | An Irwin-Hall (Uniform Sum) continuous random variable.                      |
| jf_skew_t          | Jones and Faddy skew-t distribution.                                         |
| johnsonsb          | A Johnson SB continuous random variable.                                     |
| johnsonsu          | A Johnson SU continuous random variable.                                     |
| kappa4             | Kappa 4 parameter distribution.                                              |
| kappa3             | Kappa 3 parameter distribution.                                              |
| ksone              | Kolmogorov-Smirnov one-sided test statistic distribution.                    |
| kstwo              | Kolmogorov-Smirnov two-sided test statistic distribution.                    |
| kstwobign          | Limiting distribution of scaled Kolmogorov-Smirnov two-sided test statistic. |
| landau             | A Landau continuous random variable.                                         |
| laplace            | A Laplace continuous random variable.                                        |
| laplace_asymmetric | An asymmetric Laplace continuous random variable.                            |
| levy               | A Levy continuous random variable.                                           |
| levy_l             | A left-skewed Levy continuous random variable.                               |
| levy_stable        | A Levy-stable continuous random variable.                                    |
| logistic           | A logistic (or Sech-squared) continuous random variable.                     |
| loggamma           | A log gamma continuous random variable.                                      |
| loglaplace         | A log-Laplace continuous random variable.                                    |
| lognorm            | A lognormal continuous random variable.                                      |
| loguniform         | A loguniform or reciprocal continuous random variable.                       |
| lomax              | A Lomax (Pareto of the second kind) continuous random variable.              |
| maxwell            | A Maxwell continuous random variable.                                        |
| mielke             | A Mielke Beta-Kappa / Dagum continuous random variable.                      |
| moyal              | A Moyal continuous random variable.                                          |
| nakagami           | A Nakagami continuous random variable.                                       |
| ncx2               | A non-central chi-squared continuous random variable.                        |
| ncf                | A non-central F distribution continuous random variable.                     |
| nct                | A non-central Student's t continuous random variable.                        |
| norm               | A normal continuous random variable.                                         |
| norminvgauss       | A Normal Inverse Gaussian continuous random variable.                        |
| pareto             | A Pareto continuous random variable.                                         |
| pearson3           | A pearson type III continuous random variable.                               |
| powerlaw           | A power-function continuous random variable.                                 |
| powerlognorm       | A power log-normal continuous random variable.                               |
| powernorm          | A power normal continuous random variable.                                   |
| rdist              | An R-distributed (symmetric beta) continuous random variable.                |
| rayleigh           | A Rayleigh continuous random variable.                                       |
| rel_breitwigner    | A relativistic Breit-Wigner random variable.                                 |
| rice               | A Rice continuous random variable.                                           |
| recipinvgauss      | A reciprocal inverse Gaussian continuous random variable.                    |
| semicircular       | A semicircular continuous random variable.                                   |
| skewcauchy         | A skewed Cauchy random variable.                                             |
| skewnorm           | A skew-normal random variable.                                               |
| studentized_range  | A studentized range continuous random variable.                              |
| t                  | A Student's t continuous random variable.                                    |
| trapezoid          | A trapezoidal continuous random variable.                                    |
| triang             | A triangular continuous random variable.                                     |
| truncexpon         | A truncated exponential continuous random variable.                          |
| truncnorm          | A truncated normal continuous random variable.                               |
| truncpareto        | An upper truncated Pareto continuous random variable.                        |
| truncweibull_min   | A doubly truncated Weibull minimum continuous random variable.               |
| tukeylambda        | A Tukey-Lamdba continuous random variable.                                   |
| uniform            | A uniform continuous random variable.                                        |
| vonmises           | A Von Mises continuous random variable.                                      |
| vonmises_line      | A Von Mises continuous random variable.                                      |
| wald               | A Wald continuous random variable.                                           |
| weibull_min        | Weibull minimum continuous random variable.                                  |
| weibull_max        | Weibull maximum continuous random variable.                                  |
| wrapcauchy         | A wrapped Cauchy continuous random variable.                                 |
