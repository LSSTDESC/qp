# Interpolation

Interpolated distributions are defined with:

- **$x$ values** (`xvals`): $n$ ordered values representing points on the distribution.
- **$y$ values** (`yvals`): $n$ values that correspond to the probability associated with each $x$ value.

![interpolation-example](../assets/interp-gamma-example.svg)

## Use Cases

The interpolation parameterization works well for most distributions, provided there is a high enough density of $x$ values.

- figure of what an example of this parameterization looks like
- how to make an ensemble
- how to convert an ensemble to this parameterization
  - need `xvals`
- details of how this parameterization works
- any known issues

## Data structure

- coordinates are 'xvals'
- data is 'yvals'
