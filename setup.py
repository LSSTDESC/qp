from setuptools import setup

import qp

setup(
    name="qp",
    version=qp.__version__,
    author="Alex Malz, Phil Marshall, Eric Charles",
    author_email="aimalz@nyu.edu, pjm@slac.stanford.edu, echarles@slac.stanford.edu",
    url = "https://github.com/aimalz/qp",
    packages=["qp", "docs/notebooks"],
    description="Quantile parametrization of probability distribution functions",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE", "*.npy"],
                      "docs/notebooks": ["*.npy"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["matplotlib", "numpy", "scipy>=1.7.0", "sklearn", "astropy", "h5py"]
)
