# Parameterizations

There are currently four main supported parameterizations in `qp`.

Below are detailed descriptions of each of the main supported parameterizations, including their data structures, converting to those parameterizations, and so on.

## Histogram

The

- figure of what an example of this parameterization looks like
- how to make an ensemble of histograms

### Conversion

- how to convert an ensemble to this parameterization
  - need `bins`
- details of how this parameterization works
- any known issues

## Interpolated

- figure of what an example of this parameterization looks like
- how to make an ensemble
- how to convert an ensemble to this parameterization
  - need `xvals`
- details of how this parameterization works
- any known issues

## Quantile

- figure of what an example of this parameterization looks like
- how to make an ensemble
- how to convert an ensemble to this parameterization
  - need `quants`
- details of how this parameterization works
  - pdf constructors
- any known issues

## Mixed Model

- figure of what an example of this parameterization looks like
- how to make an ensemble
- how to convert an ensemble to this parameterization
  - will do it without any arguments but optional arguments are: `ncomps`, `nsamples`, `random_state`
- details of how this parameterization works
- any known issues

## Additional parameterizations

- some additional parameterizations exist, you can see some documentation for them in the API (link)
- link to list of scipy distributions that exist as well
  - mention conversion to these distributions is not possible
- you can also add your own parameterizations, see here for guidelines and how to get started (link)
