# A primer on the `qp` package

The goal of the `qp` package is to provide a method of representing, performing operations on, and storing probability distributions. These probability distributions can be represented by a set of data points, or by parameters for continuous, analytic functions. The way a distribution is represented is called its **parameterization**, which refers both to the parameters that are used for this representation and the methods used to turn the input data or parameters into a probability distribution.

## Probability distributions

A probability distribution describes the likelihoods of possible outcomes. Within `qp`, a _distribution_ is used interchangeably to mean either a probability density function (PDF) or a cumulative distribution function (CDF), depending on the parameterization being used.

```{figure} ../assets/primer-PDF-norm.svg
:alt: Plot of the PDF of a normal distribution
```

The PDF gives the probability of a random variable falling within a range of values. For a given outcome {math}`x`, {math}`P(x)` is the relative likelihood that a random variable would give that outcome. Such a function has to always be positive, since an outcome can't have a negative probability, and the function must be normalized (its integral adds up to 1), because the overall probability that one of the outcomes happens is 1.

```{figure} ../assets/primer-CDF-norm.svg
:alt: Plot of the CDF of a normal distribution
```

The CDF is essentially the integral of the PDF. The CDF at a given x value is the probability that the outcome is less than or equal to {math}`x`. Thus the minimum of the CDF is 0 at the minimum x value, and from there it must be a non-decreasing, one-to-one function that reaches its maximum value of 1 at the maximum x value.'

### Other distribution functions

While the PDF and the CDF are the main methods used in `qp` to parameterize a distribution, there are other functions that can be derived for these distributions that are useful, and are available in `qp`. We provide a quick explanation of these functions below.

#### Percent Point Function (PPF)

(plot)
The percent point function (PPF) is the inverse of the CDF. It returns the outcome {math}`x` which has a probability less than or equal to the given probability. This means that the distribution of input values is limited to the range [0,1]. This function can be easily used to return the quantiles of a given distribution.

#### Survival Function (SF)

(plot)
The survival function (SF) is also known as the complementary cumulative distribution function. It is defined as {math}`SF(x) = 1 - CDF(x)`, so that it returns the probability that an outcome is greater than the given outcome {math}`x`.

#### Inverse Survival Function (ISF)

(plot)

The inverse survival function (ISF) serves the same purpose for the survival function as the PPF does for the CDF. It provides the outcome {math}`x` that has a probability greater than the given probability. The ISF is equivalent to {math}`CDF(1-p)`, where {math}`p` is the given probability.

## Parameterizations

A mentioned above, a **parameterization** in `qp` refers to how a distribution is represented. For example, in the above sample plots, each of these distributions are represented using the normal distribution, which is an analytic function that has the parameters "mean" and "standard deviation".

For distributions that are represented by data points, there are currently three main supported `qp` parameterizations: **histogram**, **interpolation**, and **quantiles**.

- what is a parameterization?

- List the main, supported types of distribution representations that can be created in the code, their pros and cons, link to page describing in detail
  1.  Histogram
  2.  Quantiles
  3.  Interpolated
  4.  Mixed Gaussian Models
