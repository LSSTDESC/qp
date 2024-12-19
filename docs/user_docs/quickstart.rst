===============
Getting Started
===============


Installation
============

* pip install for non-parallel

* git clone and conda install via yaml file for parallel hdf5


What is qp?
===========


* The point of qp is to provide a method of representing and storing distributions that are not continuous, analytic functions

* Essentially a mini stats primer on distributions 
    * What is a distribution/pdf? 
    * What are the qualities of a distribution (i.e. normalized, always positive, etc)
    * What can you do with a distribution? (i.e. CDF, moments, etc)


* List the main, supported types of distribution representations that can be created in the code, their pros and cons
	1. Histogram
	2. Quantiles
	3. Interpolated
	4. Mixed Models


Basic Usage 
===========

* The main object of qp is the qp.Ensemble 
    * This is a data structure that can store many distributions with the same parameterizations
    * General ensemble structure: 
        * qp metadata: tells you parameterization, qp version, and the 'x values' of whatever the distribution parameterization is
        * qp objdata: the 'y values' of the distributions 
        * Optional: qp ancil: this contains whatever information you choose
    * The exact configuration differs by ensemble, see qp data schema for details

* Creating a qp ensemble 
    * from data in a dictionary or table 
    * from file 

* Working with a qp ensemble 
    * objdata, metadata, and ancil tables 
    * What methods can be called for ensembles of all types (i.e. the scipy ones, maybe table like in scipy, also link to scipy for more info)
    * convert to different parameterizations (more detail elsewhere)
    * calculating metrics (similarly table, more detail elsewhere)

* Writing out a qp ensemble



Troubleshooting and Common Pitfalls
===================================

* normalization (?) I think the parameterizations normalize upon initialization so may actually be ok?
* scipy updates 
* some parameterizations currently have some known bugs 







