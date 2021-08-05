jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to html --execute nb/demo.ipynb; \mv nb/demo.html docs
jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to html --execute nb/practical_example.ipynb; \mv nb/practical_example.html docs
