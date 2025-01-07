# Contribution Guidelines

## Conventions to follow

### Naming

* name_pdf for parameterizations 

### Placement

* diagram or list here of basic code structure (i.e. folders and what they're meant to include)
* parameterizations go in the parameterizations folder
* if you have one that needs supporting functions, create a folder and place the class file and the supporting files in that folder
* conversion functions go in the conversion_funcs file

     

## Tests 

* what is expected in terms of code coverage, what tests need to be written, types of tests, etc 
* tests should exist purely in test folder
* test data stored in tests/test_data
* output of tests should be written to temporary path for ease of clean up 

## Documentation

* every function should have docstrings describing what the function does 
* what is expected in terms of documentation upkeep (i.e. if you change something and the documentation is now out of date)


## PR 

* steps for submitting a PR to github
    * run pylint 
    * make sure you have appropriate test coverage
