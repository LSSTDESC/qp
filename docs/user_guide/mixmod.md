# Gaussian Mixed Model

## Introduction

## Data structure

- data table: weights, stds, means

## Conversion

## Implementation notes

- currently the Gaussian mixed model parameterization `rvs()` method is not functional. This also means that converting Gaussian mixed model `Ensembles` to other types of `Ensembles` via conversion methods that use sampling will not work.

- figure of what an example of this parameterization looks like
- how to make an ensemble
- how to convert an ensemble to this parameterization
  - will do it without any arguments but optional arguments are: `ncomps`, `nsamples`, `random_state`
- details of how this parameterization works
- any known issues
