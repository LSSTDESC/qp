============================================
qp : quantile-parametrized PDF approximation
============================================

`qp` is a python library for the storage and manipulation of tables of probability distributions. 



Features
--------

- Read and write tables of probability distributions to/from file
- Parameterize probability distributions inferred from real data
- Convert between different methods of parameterizing probability distributions  
- Perform statistical methods on many distributions at a time

`qp` is currently a part of the `LSST DESC <https://lsstdesc.org/>`_ `RAIL <https://github.com/LSSTDESC/rail>`_ package and its sub-packages. 


.. rst-class:: center

**Useful links**: `Source repository <https://github.com/LSSTDESC/qp>`_ | `PyPI <https://pypi.org/project/qp-prob/>`_



.. _cards-clickable: 

.. grid:: 2
    :gutter: 3
    :margin: 5 5 0 0
    :padding: 0

    .. grid-item-card::
        :link: user_guide/quickstart
        :link-type: doc 
        :link-alt: quickstart
        :text-align: center 


        :fas:`fa-solid fa-rocket; fa-5x`


        **Quickstart**
        
        A quick introduction to some of the main functionality of `qp`. This guide is 
        short and sweet for when you're in a rush or need a quick refresher.


    .. grid-item-card::
        :link: user_guide/installation
        :link-type: doc
        :link-alt: user-guide-installation
        :text-align: center

        :fas:`fa-solid fa-book; fa-5x`

        **User guide**

        A good starting place for new users who have never used `qp` before. Includes a short 
        primer on statistics and `qp` terminology, and detailed usage explanations and examples. 
     


    .. grid-item-card::
        :link: user_guide/cookbook/index
        :link-type: doc
        :link-alt: cookbook
        :text-align: center

        :fas:`fa-solid fa-terminal; fa-5x`

        **Cookbook**

        Contains useful examples of `qp` usage. These examples range from quick reference one-liners 
        to in-depth tutorials of more complex usage cases.

    .. grid-item-card::
        :link: developer_docs/setup
        :link-type: doc
        :link-alt: developer-documentation
        :text-align: center

        :fas:`fa-solid fa-code-compare; fa-5x`

        **Developer Documentation**

        Detailed setup and contribution workflow for new developers, and a reference for ongoing issues in need of development.  

.. admonition:: Citation
    :class: note

    If you end up using any of the code or ideas you find here in your academic research, please cite our paper: `A. I. Malz et al 2018 AJ 156 35 <https://ui.adsabs.harvard.edu/abs/2018AJ....156...35M/abstract>`_ (`ADS - BibTex <https://ui.adsabs.harvard.edu/abs/2018AJ....156...35M/exportcitation>`_).

.. toctree:: 
    :hidden:
    :maxdepth: 3

    whatisqp
    
.. toctree::
    :hidden:
    :maxdepth: 4
    :caption: User Guide

    user_guide/quickstart
    user_guide/installation
    user_guide/qpprimer
    user_guide/basicusage
    user_guide/datastructure
    user_guide/parameterizations/index.md
    user_guide/cookbook/index
    user_guide/methods
    user_guide/troubleshooting

    

.. toctree::
    :hidden:
    :maxdepth: 4
    :caption: Developer Documentation

    developer_docs/setup
    developer_docs/codestructure
    developer_docs/contribution
    developer_docs/parameterizationcontribution
    developer_docs/techdebt
    developer_docs/roadmap

.. toctree:: 
    :hidden:
    :maxdepth: 5
    :caption: API Documentation
    
    api_docs/index.rst

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Demo Notebooks

    nb/index.md


.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: More

    license
    acknowledgements
    GitHub <https://github.com/LSSTDESC/qp>


.. rst-class:: center

:ref:`genindex` | 
:ref:`modindex` | 
:ref:`search`

