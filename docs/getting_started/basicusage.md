# Basic Usage

- The main object of qp is the qp.Ensemble

  - This is a data structure that can store many distributions with the same parameterizations
  - General ensemble structure:
    - qp metadata: tells you parameterization, qp version, and the 'coordinates' of whatever the distribution parameterization is
    - qp objdata: the 'data' of the distributions
    - Optional: qp ancil: this contains whatever information you choose
  - The exact configuration of the data within these dictionaries differs by ensemble, see qp data structure for details

- Creating a qp ensemble

  - from data in a dictionary or table
  - from file

- Working with a qp ensemble

  - objdata, metadata, and ancil tables
  - Show a couple of the more important methods that can be called, link to the ensemble methods page which lists all of them
  - basics of how to convert to different parameterizations (more detail elsewhere)
    - mention you can't convert to scipy parameterizations
  - calculating metrics (basics, table of existing supported metrics)

- Writing out a qp ensemble
  - format file is normally written to
  - see data structure for more information about what's written
  - see cookbook for example
