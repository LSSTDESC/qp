# Basic Usage 


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
