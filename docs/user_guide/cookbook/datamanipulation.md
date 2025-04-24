# Data manipulation

This notebook covers use cases of accessing and manipulating the data contained in Ensembles and Ensemble files.

## Exploring the structure of an Ensemble file

This tutorial notebook does an in-depth look of what the actual data structure of an Ensemble file looks like, and how to create one from data tables: <project:../../nb/ensemble_file.md> (download {download}`here <../../nb/ensemble_file.ipynb>`).

## Accessing Ensemble data

Below are examples of how to access the relevant metadata and data coordinates for each of the supported parameterizations, as well as the normal parameterization.

### Accessing the bins and pdf values of a histogram Ensemble

The bin edges are common to all distributions, so they are found in the {py:attr}`qp.Ensemble.metadata` dictionary. The bin values ('pdfs') are unique to each distribution, and so they are found in the {py:attr}`qp.Ensemble.objdata` dictionary.

```{doctest}

>>> ens_h.metadata["bins"]
array([-1.  , -0.76, -0.52, -0.28, -0.04,  0.2 ,  0.44,  0.68,  0.92,
        1.16,  1.4 ,  1.64,  1.88,  2.12,  2.36,  2.6 ,  2.84,  3.08,
        3.32,  3.56,  3.8 ,  4.04,  4.28,  4.52,  4.76,  5.  ])
>>> ens_h.objdata["pdfs"]
array([[9.32923960e-18, 1.97897967e-13, 9.45298950e-10, 1.02855210e-06,
        2.60941589e-04, 1.61237560e-02, 2.60412136e-01, 1.20404896e+00,
        1.73585762e+00, 8.19029064e-01, 1.25107478e-01, 5.75357860e-03,
        7.18755383e-05, 2.23188835e-07, 1.62884983e-10, 2.68303898e-14,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [1.14354160e-03, 4.21698632e-03, 1.20328160e-02, 3.02614760e-02,
        6.71100840e-02, 1.31303762e-01, 2.26760990e-01, 3.45825540e-01,
        4.65925752e-01, 5.54741107e-01, 5.83826084e-01, 5.43198413e-01,
        4.46818319e-01, 3.24914337e-01, 2.08830620e-01, 1.18600647e-01,
        5.94965882e-02, 2.63527299e-02, 1.03011111e-02, 3.55182656e-03,
        1.07971268e-03, 2.89224138e-04, 6.82367667e-05, 1.41728752e-05,
        2.59038348e-06],
       [1.25202752e-02, 2.55094001e-02, 4.02000834e-02, 6.06476803e-02,
        8.75985090e-02, 1.21145507e-01, 1.60426590e-01, 2.03437637e-01,
        2.47056497e-01, 2.87337054e-01, 3.20061620e-01, 3.41455402e-01,
        3.48899207e-01, 3.41455402e-01, 3.20061620e-01, 2.87337054e-01,
        2.47056497e-01, 2.03437637e-01, 1.60426590e-01, 1.21145507e-01,
        8.75985090e-02, 6.06476803e-02, 4.02000834e-02, 2.55094001e-02,
        1.54952237e-02]])

```

### Accessing the x and y values of an interpolation Ensemble

The $x$ values are shared across all distributions, so they are found in the {py:attr}`qp.Ensemble.metadata` dictionary. The $y$ values are unique to each distribution, so they are found in the {py:attr}`qp.Ensemble.objdata` dictionary.

```{doctest}

>>> ens.metadata["xvals"]
array([-1.        , -0.87755102, -0.75510204, -0.63265306, -0.51020408,
       -0.3877551 , -0.26530612, -0.14285714, -0.02040816,  0.10204082,
        0.2244898 ,  0.34693878,  0.46938776,  0.59183673,  0.71428571,
        0.83673469,  0.95918367,  1.08163265,  1.20408163,  1.32653061,
        1.44897959,  1.57142857,  1.69387755,  1.81632653,  1.93877551,
        2.06122449,  2.18367347,  2.30612245,  2.42857143,  2.55102041,
        2.67346939,  2.79591837,  2.91836735,  3.04081633,  3.16326531,
        3.28571429,  3.40816327,  3.53061224,  3.65306122,  3.7755102 ,
        3.89795918,  4.02040816,  4.14285714,  4.26530612,  4.3877551 ,
        4.51020408,  4.63265306,  4.75510204,  4.87755102,  5.        ])
>>> ens.objdata["yvals"]
array([[3.84729931e-22, 1.45447239e-19, 3.77974403e-17, 6.75191079e-15,
        8.29083747e-13, 6.99805751e-11, 4.06035506e-09, 1.61941410e-07,
        4.43975721e-06, 8.36696446e-05, 1.08388705e-03, 9.65178258e-03,
        5.90797206e-02, 2.48586039e-01, 7.18989314e-01, 1.42947162e+00,
        1.95360176e+00, 1.83528678e+00, 1.18516614e+00, 5.26092288e-01,
        1.60528458e-01, 3.36704974e-02, 4.85461093e-03, 4.81134752e-04,
        3.27782998e-05, 1.53501817e-06, 4.94137729e-08, 1.09342729e-09,
        1.66317981e-11, 1.73898526e-13, 1.24985604e-15, 6.17492215e-18,
        2.09705770e-20, 4.89549569e-23, 7.85579903e-26, 8.66545669e-29,
        6.57052311e-32, 3.42464720e-35, 1.22698465e-38, 3.02182853e-42,
        5.11573344e-46, 5.95324006e-50, 4.76218523e-54, 2.61858437e-58,
        9.89769939e-63, 2.57163511e-67, 4.59295122e-72, 5.63873544e-77,
...
        2.54521656e-01, 2.32452633e-01, 2.09903856e-01, 1.87405606e-01,
        1.65432547e-01, 1.44389480e-01, 1.24602387e-01, 1.06314718e-01,
        8.96884739e-02, 7.48093867e-02, 6.16952559e-02, 5.03064490e-02,
        4.05575555e-02, 3.23292847e-02, 2.54798365e-02, 1.98551602e-02,
        1.52977080e-02, 1.16534779e-02]])

```

### Accessing the x and y values of an irregular interpolation Ensemble

The $x$ and $y$ values are unique to each distribution in an irregular interpolation Ensemble, so they are both found in the {py:attr}`qp.Ensemble.objdata` dictionary.

```{doctest}

>>> ens_irr.objdata["xvals"]
array([[0.  , 0.25, 0.5 , 0.75, 1.  ],
       [1.  , 1.25, 1.5 , 1.75, 2.  ]])
>>> ens_irr.objdata["yvals"]
array([[1.93480833, 0.17750059, 1.88439427, 0.68718763, 1.25091751],
       [0.62741694, 1.04245176, 0.81553133, 0.76474132, 1.37727559]])

```

### Accessing the quantiles and locations of a quantile Ensemble

The quantiles are common to all distributions, so these are found in the {py:attr}`qp.Ensemble.metadata` dictionary. The locations are specific to each distribution, so are found in the {py:attr}`qp.Ensemble.objdata` dictionary.

```{doctest}

>>> ens_q.metadata["quants"]
array([0.   , 0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777, 0.888,
       0.999, 1.   ])
>>> ens_q.locs["locs"]
array([2.        , 2.24254695, 2.35428008, 2.44998068, 2.5417504 ,
       2.63627077, 2.74047765, 2.86619383, 3.04624481, 3.85846109,
       3.86577836])

```

### Accessing the means, standard deviations, and weights of a Gaussian mixture model Ensemble

Since each distribution has its own mean, standard deviation and weight, these values are all found in the {py:attr}`qp.Ensemble.objdata` dictionary:

```{doctest}

>>> ens_m.objdata["means"]
array([-1. , -0.5,  0. ,  0.5,  1. ])
>>> ens_m.objdata["stds"]
array([0.1  , 0.275, 0.45 , 0.625, 0.8  ])
>>> ens_m.objdata["weights"]
array([0.1, 0.3, 0.2, 0.2, 0.2])

```

### Accessing the mean and standard deviation of a normal Ensemble

Since each distribution has its own mean and standard deviation, these values are all found in the {py:attr}`qp.Ensemble.objdata` dictionary:

```{doctest}

>>> ens_n.objdata["loc"] # gives the means
array([[0],
       [1]])
>>> ens_n.objdata["scale"] # gives the standard deviations
array([[0.5 ],
       [0.25]])

```

This is true of all the {py:class}`qp.stats` distributions, though some will have different variables you can access. To find out what variables exist for a specific {py:class}`qp.stats` distribution, take a look at the <project:../parameterizations/index.md> page.

## Updating the data in an Ensemble

To update the data in an Ensemble without changing its metadata, you can use the {py:meth}`qp.Ensemble.update_objdata` method. This will recreate the Ensemble with the existing metadata and the new data you've provided. If you'd like to preserve your old Ensemble and make a new Ensemble with this updated data, you should first save a copy of the old Ensemble to a different variable, like in the example below:

```{doctest}

>>> import qp
>>> import numpy as np
>>> # create a histogram Ensemble
>>> ens_h = qp.hist.create_ensemble(bins= np.array([0,1,2,3,4,5]),
... pdfs = np.array([0,0.1,0.1,0.4,0.2]))
>>> ens_h.objdata # values before updating
{'pdfs': array([0.   , 0.125, 0.125, 0.5  , 0.25 ])}
>>> ens_h_old = ens_h # assign Ensemble to new variable to keep old version
>>> # update Ensemble with new data
>>> ens_h.update_objdata(data={'pdfs': np.array([0.05,0.09,0.2,0.3,0.15])})
>>> ens_h.objdata # values after updating
{'pdfs': array([[0.06329114, 0.11392405, 0.25316456, 0.37974684, 0.18987342]])}

```

If you'd like to change not only the data but also the metadata of the Ensemble, you can use the {py:meth}`qp.Ensemble.update` method. Let's say we want to get a new version of our histogram Ensemble where we have one less bin:

```{doctest}

>>> ens_h.update(data={'bins': np.array([0,1,2,3,4]),'pdfs': np.array([0.05,0.09,0.2,0.3])})
>>> ens_h.objdata
{'pdfs': array([0.078125, 0.140625, 0.3125  , 0.46875 ])}
>>> ens_h.shape
(1, 4)

```

## Normalizing an Ensemble

If you have an Ensemble and want to ensure it's normalized, you can use the {py:meth}`qp.Ensemble.norm` method. This method will only work for interpolation, irregular interpolation, and histogram Ensembles.

Let's say you created an Ensemble without normalizing, but now you've changed your mind and want it normalized:

```{doctest}

>>> import qp
>>> import numpy as np
>>> # create interpolated Ensemble
>>> xvals= np.array([0,0.5,1,1.5,2])
>>> yvals = np.array([[0.01, 0.2,0.3,0.2,0.01],[0.09,0.25,0.2,0.1,0.01]])
>>> ens_i = qp.interp.create_ensemble(xvals=xvals, yvals=yvals,norm=False)
>>> ens_i.objdata["yvals"] # values before normalizing
array([[0.01, 0.2 , 0.3 , 0.2 , 0.01],
       [0.09, 0.25, 0.2 , 0.1 , 0.01]])

>>> # normalize the Ensemble
>>> ens_i.norm()
>>> ens_i.objdata["yvals"] # values after normalizing
array([[0.02816901, 0.56338028, 0.84507042, 0.56338028, 0.02816901],
       [0.3       , 0.83333333, 0.66666667, 0.33333333, 0.03333333]])

```
