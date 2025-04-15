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

`qp` also incorporates all of the continuous distributions from <inv:scipy:scipy.stats> as parameterizations, allowing you to create `Ensembles` parameterized by these distributions. The main limitations on these parameterizations is that they cannot be converted to, as they have no conversion functions. Additionally, their `create_ensemble()` method takes a data dictionary instead of specific arguments, as shown in [the Cookbook](../cookbook/ensemblemanipulation.md#creating-an-ensemble-from-a-qp-stats-distribution).

You can access these parameterizations from the `qp.stats` module, as shown below:

```{doctest}

>>> import qp
>>> ens_n = qp.stats.norm.create_ensemble(dict(loc=np.array([0,1]),scale=np.array([0.25,0.5])))
>>> ens_n
Ensemble(the_class=norm,shape=(2,1))

```

The available parameterizations are listed below, with links to their SciPy documentation for more information.

| Function                                                                                                                                      | Description                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [alpha](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.alpha.html#scipy.stats.alpha)                                        | An alpha continuous random variable.                                         |
| [anglit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anglit.html#scipy.stats.anglit)                                     | An anglit continuous random variable.                                        |
| [arcsine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.arcsine.html#scipy.stats.arcsine)                                  | An arcsine continuous random variable.                                       |
| [argus](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.argus.html#scipy.stats.argus)                                        | Argus distribution                                                           |
| [beta](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html#scipy.stats.beta)                                           | A beta continuous random variable.                                           |
| [betaprime](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betaprime.html#scipy.stats.betaprime)                            | A beta prime continuous random variable.                                     |
| [bradford](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bradford.html#scipy.stats.bradford)                               | A Bradford continuous random variable.                                       |
| [burr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.burr.html#scipy.stats.burr)                                           | A Burr (Type III) continuous random variable.                                |
| [burr12](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.burr12.html#scipy.stats.burr12)                                     | A Burr (Type XII) continuous random variable.                                |
| [cauchy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cauchy.html#scipy.stats.cauchy)                                     | A Cauchy continuous random variable.                                         |
| [chi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi.html#scipy.stats.chi)                                              | A chi continuous random variable.                                            |
| [chi2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html#scipy.stats.chi2)                                           | A chi-squared continuous random variable.                                    |
| [cosine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cosine.html#scipy.stats.cosine)                                     | A cosine continuous random variable.                                         |
| [crystalball](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.crystalball.html#scipy.stats.crystalball)                      | Crystalball distribution                                                     |
| [dgamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dgamma.html#scipy.stats.dgamma)                                     | A double gamma continuous random variable.                                   |
| [dpareto_lognorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dpareto_lognorm.html#scipy.stats.dpareto_lognorm)          | A double Pareto lognormal continuous random variable.                        |
| [dweibull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dweibull.html#scipy.stats.dweibull)                               | A double Weibull continuous random variable.                                 |
| [erlang](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.erlang.html#scipy.stats.erlang)                                     | An Erlang continuous random variable.                                        |
| [expon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html#scipy.stats.expon)                                        | An exponential continuous random variable.                                   |
| [exponnorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html#scipy.stats.exponnorm)                            | An exponentially modified Normal continuous random variable.                 |
| [exponweib](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponweib.html#scipy.stats.exponweib)                            | An exponentiated Weibull continuous random variable.                         |
| [exponpow](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponpow.html#scipy.stats.exponpow)                               | An exponential power continuous random variable.                             |
| [f](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html#scipy.stats.f)                                                    | An F continuous random variable.                                             |
| [fatiguelife](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fatiguelife.html#scipy.stats.fatiguelife)                      | A fatigue-life (Birnbaum-Saunders) continuous random variable.               |
| [fisk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisk.html#scipy.stats.fisk)                                           | A Fisk continuous random variable.                                           |
| [foldcauchy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.foldcauchy.html#scipy.stats.foldcauchy)                         | A folded Cauchy continuous random variable.                                  |
| [foldnorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.foldnorm.html#scipy.stats.foldnorm)                               | A folded normal continuous random variable.                                  |
| [genlogistic](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genlogistic.html#scipy.stats.genlogistic)                      | A generalized logistic continuous random variable.                           |
| [gennorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gennorm.html#scipy.stats.gennorm)                                  | A generalized normal continuous random variable.                             |
| [genpareto](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html#scipy.stats.genpareto)                            | A generalized Pareto continuous random variable.                             |
| [genexpon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genexpon.html#scipy.stats.genexpon)                               | A generalized exponential continuous random variable.                        |
| [genextreme](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html#scipy.stats.genextreme)                         | A generalized extreme value continuous random variable.                      |
| [gausshyper](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gausshyper.html#scipy.stats.gausshyper)                         | A Gauss hypergeometric continuous random variable.                           |
| [gamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html#scipy.stats.gamma)                                        | A gamma continuous random variable.                                          |
| [gengamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gengamma.html#scipy.stats.gengamma)                               | A generalized gamma continuous random variable.                              |
| [genhalflogistic](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genhalflogistic.html#scipy.stats.genhalflogistic)          | A generalized half-logistic continuous random variable.                      |
| [genhyperbolic](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genhyperbolic.html#scipy.stats.genhyperbolic)                | A generalized hyperbolic continuous random variable.                         |
| [geninvgauss](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geninvgauss.html#scipy.stats.geninvgauss)                      | A Generalized Inverse Gaussian continuous random variable.                   |
| [gibrat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gibrat.html#scipy.stats.gibrat)                                     | A Gibrat continuous random variable.                                         |
| [gompertz](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gompertz.html#scipy.stats.gompertz)                               | A Gompertz (or truncated Gumbel) continuous random variable.                 |
| [gumbel_r](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html#scipy.stats.gumbel_r)                               | A right-skewed Gumbel continuous random variable.                            |
| [gumbel_l](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_l.html#scipy.stats.gumbel_l)                               | A left-skewed Gumbel continuous random variable.                             |
| [halfcauchy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfcauchy.html#scipy.stats.halfcauchy)                         | A Half-Cauchy continuous random variable.                                    |
| [halflogistic](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halflogistic.html#scipy.stats.halflogistic)                   | A half-logistic continuous random variable.                                  |
| [halfnorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfnorm.html#scipy.stats.halfnorm)                               | A half-normal continuous random variable.                                    |
| [halfgennorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfgennorm.html#scipy.stats.halfgennorm)                      | The upper half of a generalized normal continuous random variable.           |
| [hypsecant](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypsecant.html#scipy.stats.hypsecant)                            | A hyperbolic secant continuous random variable.                              |
| [invgamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html#scipy.stats.invgamma)                               | An inverted gamma continuous random variable.                                |
| [invgauss](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html#scipy.stats.invgauss)                               | An inverse Gaussian continuous random variable.                              |
| [invweibull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invweibull.html#scipy.stats.invweibull)                         | An inverted Weibull continuous random variable.                              |
| [irwinhall](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.irwinhall.html#scipy.stats.irwinhall)                            | An Irwin-Hall (Uniform Sum) continuous random variable.                      |
| [jf_skew_t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jf_skew_t.html#scipy.stats.jf_skew_t)                            | Jones and Faddy skew-t distribution.                                         |
| [johnsonsb](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.johnsonsb.html#scipy.stats.johnsonsb)                            | A Johnson SB continuous random variable.                                     |
| [johnsonsu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.johnsonsu.html#scipy.stats.johnsonsu)                            | A Johnson SU continuous random variable.                                     |
| [kappa4](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kappa4.html#scipy.stats.kappa4)                                     | Kappa 4 parameter distribution.                                              |
| [kappa3](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kappa3.html#scipy.stats.kappa3)                                     | Kappa 3 parameter distribution.                                              |
| [ksone](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ksone.html#scipy.stats.ksone)                                        | Kolmogorov-Smirnov one-sided test statistic distribution.                    |
| [kstwo](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstwo.html#scipy.stats.kstwo)                                        | Kolmogorov-Smirnov two-sided test statistic distribution.                    |
| [kstwobign](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstwobign.html#scipy.stats.kstwobign)                            | Limiting distribution of scaled Kolmogorov-Smirnov two-sided test statistic. |
| [landau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.landau.html#scipy.stats.landau)                                     | A Landau continuous random variable.                                         |
| [laplace](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html#scipy.stats.laplace)                                  | A Laplace continuous random variable.                                        |
| [laplace_asymmetric](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace_asymmetric.html#scipy.stats.laplace_asymmetric) | An asymmetric Laplace continuous random variable.                            |
| [levy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy.html#scipy.stats.levy)                                           | A Levy continuous random variable.                                           |
| [levy_l](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_l.html#scipy.stats.levy_l)                                     | A left-skewed Levy continuous random variable.                               |
| [levy_stable](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html#scipy.stats.levy_stable)                      | A Levy-stable continuous random variable.                                    |
| [logistic](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logistic.html#scipy.stats.logistic)                               | A logistic (or Sech-squared) continuous random variable.                     |
| [loggamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loggamma.html#scipy.stats.loggamma)                               | A log gamma continuous random variable.                                      |
| [loglaplace](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loglaplace.html#scipy.stats.loglaplace)                         | A log-Laplace continuous random variable.                                    |
| [lognorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm)                                  | A lognormal continuous random variable.                                      |
| [loguniform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html#scipy.stats.loguniform)                         | A loguniform or reciprocal continuous random variable.                       |
| [lomax](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lomax.html#scipy.stats.lomax)                                        | A Lomax (Pareto of the second kind) continuous random variable.              |
| [maxwell](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.maxwell.html#scipy.stats.maxwell)                                  | A Maxwell continuous random variable.                                        |
| [mielke](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mielke.html#scipy.stats.mielke)                                     | A Mielke Beta-Kappa / Dagum continuous random variable.                      |
| [moyal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moyal.html#scipy.stats.moyal)                                        | A Moyal continuous random variable.                                          |
| [nakagami](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nakagami.html#scipy.stats.nakagami)                               | A Nakagami continuous random variable.                                       |
| [ncx2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html#scipy.stats.ncx2)                                           | A non-central chi-squared continuous random variable.                        |
| [ncf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncf.html#scipy.stats.ncf)                                              | A non-central F distribution continuous random variable.                     |
| [nct](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html#scipy.stats.nct)                                              | A non-central Student's t continuous random variable.                        |
| [norm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm)                                           | A normal continuous random variable.                                         |
| [norminvgauss](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norminvgauss.html#scipy.stats.norminvgauss)                   | A Normal Inverse Gaussian continuous random variable.                        |
| [pareto](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html#scipy.stats.pareto)                                     | A Pareto continuous random variable.                                         |
| [pearson3](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearson3.html#scipy.stats.pearson3)                               | A pearson type III continuous random variable.                               |
| [powerlaw](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powerlaw.html#scipy.stats.powerlaw)                               | A power-function continuous random variable.                                 |
| [powerlognorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powerlognorm.html#scipy.stats.powerlognorm)                   | A power log-normal continuous random variable.                               |
| [powernorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powernorm.html#scipy.stats.powernorm)                            | A power normal continuous random variable.                                   |
| [rdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rdist.html#scipy.stats.rdist)                                        | An R-distributed (symmetric beta) continuous random variable.                |
| [rayleigh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html#scipy.stats.rayleigh)                               | A Rayleigh continuous random variable.                                       |
| [rel_breitwigner](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rel_breitwigner.html#scipy.stats.rel_breitwigner)          | A relativistic Breit-Wigner random variable.                                 |
| [rice](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rice.html#scipy.stats.rice)                                           | A Rice continuous random variable.                                           |
| [recipinvgauss](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.recipinvgauss.html#scipy.stats.recipinvgauss)                | A reciprocal inverse Gaussian continuous random variable.                    |
| [semicircular](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.semicircular.html#scipy.stats.semicircular)                   | A semicircular continuous random variable.                                   |
| [skewcauchy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewcauchy.html#scipy.stats.skewcauchy)                         | A skewed Cauchy random variable.                                             |
| [skewnorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html#scipy.stats.skewnorm)                               | A skew-normal random variable.                                               |
| [studentized_range](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.studentized_range.html#scipy.stats.studentized_range)    | A studentized range continuous random variable.                              |
| [t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t)                                                    | A Student's t continuous random variable.                                    |
| [trapezoid](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trapezoid.html#scipy.stats.trapezoid)                            | A trapezoidal continuous random variable.                                    |
| [triang](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.triang.html#scipy.stats.triang)                                     | A triangular continuous random variable.                                     |
| [truncexpon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncexpon.html#scipy.stats.truncexpon)                         | A truncated exponential continuous random variable.                          |
| [truncnorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm)                            | A truncated normal continuous random variable.                               |
| [truncpareto](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncpareto.html#scipy.stats.truncpareto)                      | An upper truncated Pareto continuous random variable.                        |
| [truncweibull_min](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncweibull_min.html#scipy.stats.truncweibull_min)       | A doubly truncated Weibull minimum continuous random variable.               |
| [tukeylambda](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukeylambda.html#scipy.stats.tukeylambda)                      | A Tukey-Lamdba continuous random variable.                                   |
| [uniform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html#scipy.stats.uniform)                                  | A uniform continuous random variable.                                        |
| [vonmises](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises.html#scipy.stats.vonmises)                               | A Von Mises continuous random variable.                                      |
| [vonmises_line](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises_line.html#scipy.stats.vonmises_line)                | A Von Mises continuous random variable.                                      |
| [wald](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wald.html#scipy.stats.wald)                                           | A Wald continuous random variable.                                           |
| [weibull_min](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html#scipy.stats.weibull_min)                      | Weibull minimum continuous random variable.                                  |
| [weibull_max](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_max.html#scipy.stats.weibull_max)                      | Weibull maximum continuous random variable.                                  |
| [wrapcauchy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wrapcauchy.html#scipy.stats.wrapcauchy)                         | A wrapped Cauchy continuous random variable.                                 |
