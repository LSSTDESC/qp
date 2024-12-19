# Data Schema

* basic data schema below

* metadata table is dictionary of arrays 
    * pdf_name, pdf_version, and x values 
* data table is dictionary of arrays 
    * typically data has one 2d array of y values, with a shape (num_pdfs (rows), len(xvals) (columns))

* data schema is slightly different for each parameterization
* For all parameterizations that are not continuous, just the 'x values' in the metadata table and 'y values' in the data table have different names 
* the format of the data itself within those places is roughly the same 


## Histogram parameterization

* x is 'bins'
* y is 'pdfs', number of columns is len(bins) - y instead of length of bins

## Interpolated parameterization

* x is 'xvals'
* y is 'yvals'

## Quantile parameterization

* x is 'quants'
* y is 'locs'
* metadata table includes pdf_constructor_name and check_input 

## Mixed model parameterization

data table
weights, stds, means 


## Spline 

data table
splx, sply, spln 

## Packed interp 

* x is 'xvals', also packing_type and log_floor
* y is 'ypacked', also 'ymax'


'columns' or arrays in data table
* ypacked: Array of values, (shape = (x,))
* ymax: Array of 1 value
