# Creating New Parameterizations

## Required 

* init function 
    * should include functions to store values needed, pass pdf shape to base class constructor
    * should call addmetadata and addobjdata methods
* functions to access each of the data and metadata fields 
* _pdf and _cdf hook functions 
* implement _updated_ctor_param function
* define functions to convert other ensembles to this representation
* register class with the factor and make function available 
* add test data generation (or simply a test data file) to the tests folder and add a test to the automatically generated tests

## Optional 
 
* _sf, _ppf, _isf, _rvs functions for faster evaluation
* plotting (plot_native)


## Code template 

* below is a code template for what a basic parameterization should look like
* use this to start your new parameterizations 

```{literalinclude} parameterization_template.py
```