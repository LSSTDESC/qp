# Quantile

## Introduction

- figure of what an example of this parameterization looks like
- how to make an ensemble

- details of how this parameterization works
  - ensure_extent: on by default, adds in extra quants to make sure the range of quants is from (0,1)
  - pdf constructors
- any known issues
  - the interpolated PDF can go to negative values
  - if dealing with normal distributions when creating, make sure not to have infinite values
  -

## Data structure

- coordinates are 'quants'
- data is 'locs'
- metadata table includes pdf_constructor_name and ensure_extent

## Conversion

- how to convert an ensemble to this parameterization
  - need `quants`
