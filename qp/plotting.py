"""Functions to plot PDFs"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import rv_histogram
from qp import utils

from .scipy_extend import interp_dist, spline_dist, intspline_dist, kde_dist, quantile_dist


def init_matplotlib():
    """Initialize matplotlib parameters"""
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Times New Roman'
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['savefig.dpi'] = 250
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['savefig.bbox'] = 'tight'


init_matplotlib()

COLORS = {}
COLORS['truth'] = 'k'
COLORS['mix_mod'] = 'k'
COLORS['gridded'] = 'k'
COLORS['quantiles'] = 'blueviolet'
COLORS['histogram'] = 'darkorange'
COLORS['samples'] = 'forestgreen'

STYLES = {}
STYLES['truth'] = '-'
STYLES['mix_mod'] = ':'
STYLES['gridded'] = '--'
STYLES['quantiles'] = '--'#(0,(5,10))
STYLES['histogram'] = ':'#(0,(3,6))
STYLES['samples'] = '-.'#(0,(1,2))


def plot_pdf_on_axes(axes, dist, xvals, **kwargs):
    """
    Plot a PDF on a set of axes, by evaluating it a set of points

    Parameters
    ----------
    axes : The axes we want to plot the data on

    dist : `scipy.stats.rv_continuous`
        The distribution we want to plot

    xvals : `np.array`
        The locations we evaluate the PDF at for plotting

    **kwargs : passed directly to the `matplotlib` plot function

    Return
    ------
    axes : The axes the data are plotted on
    """

    yvals = dist.pdf(xvals)
    axes.plot(xvals, yvals, **kwargs)
    return axes


def plot_dist_pdf(axes, dist, **kwargs):
    """
    Plot a PDF on a set of axes, using the axes limits

    Parameters
    ----------
    axes : The axes we want to plot the data on

    dist : `scipy.stats.rv_continuous`
        The distribution we want to plot

    Keywords
    --------
    npoints : `int`
        Number of points to use in the plotting.  Evenly spaced along the axis provided.

    **kwargs : passed directly to the `matplotlib` plot function

    Return
    ------
    axes : The axes the data are plotted on
    """

    xlim = axes.get_xlim()
    npoints = kwargs.pop('npoints', 101)
    xvals = np.linspace(xlim[0], xlim[1], npoints)
    return plot_pdf_on_axes(axes, dist, xvals, **kwargs)


def plot_xval_dist_pdf(axes, dist, **kwargs):
    """
    Plot a PDF on a set of axes, by evaluating it at the points carried by the distribution

    Parameters
    ----------
    axes : The axes we want to plot the data on

    dist : `scipy.stats.rv_continuous`
        The distribution we want to plot

    Keywords
    --------
    **kwargs : passed directly to the `matplotlib` plot function

    Return
    ------
    axes : The axes the data are plotted on
    """

    return plot_pdf_on_axes(axes, dist, dist.xvals, **kwargs)


def plot_pdf_quantiles_on_axes(axes, dist, quantiles, **kwargs):
    """
    Plot a PDF on a set of axes, by evaluating at the quantiles provided

    Parameters
    ----------
    axes : The axes we want to plot the data on

    dist : `scipy.stats.rv_continuous`
        The distribution we want to plot

    quantiles : (`np.array`, `np.array`)
       The quantiles that define the distribution pdf

    Keywords
    --------
    npoints : `int`
        Number of points to use in the plotting.  Evenly spaced along the axis provided.

    **kwargs : passed directly to the `matplotlib` plot function

    Return
    ------
    axes : The axes the data are plotted on
    """
    #FIXME
    npoints = kwargs.pop('npoints', 101)
    xlim = axes.get_xlim()
    xvals = np.linspace(xlim[0], xlim[1], npoints)
    qinterpolated = utils.approximate_dist(dist, xvals)
    axes.scatter(quantiles[1], np.zeros(np.shape(quantiles[1])), color=COLORS['quantiles'], marker='|', s=100, label='Quantiles', alpha=0.75)
    axes.plot(xvals, qinterpolated[1], color=COLORS['quantiles'], lw=2.0, alpha=1.0, linestyle=STYLES['quantiles'], label='Quantile Interpolated PDF')
    return axes


def plot_quantile_dist_pdf(axes, dist, **kwargs):
    """
    Plot a PDF on a set of axes, by evaluating it at the quantiles points carried by the distribution

    Parameters
    ----------
    axes : The axes we want to plot the data on

    dist : `scipy.stats.rv_continuous`
        The distribution we want to plot

    Keywords
    --------
    **kwargs : passed directly to the `matplotlib` plot function

    Return
    ------
    axes : The axes the data are plotted on
    """

    return plot_pdf_quantiles_on_axes(axes, dist, quantiles=dist.quantiles, **kwargs)


def plot_pdf_histogram_on_axes(axes, dist, hist, **kwargs):
    """
    Plot a PDF on a set of axes, by plotting the histogrammed data

    Parameters
    ----------
    axes : The axes we want to plot the data on

    dist : `scipy.stats.rv_continuous`
        The distribution we want to plot

    Keywords
    --------
    npoints : `int`
        Number of points to use in the plotting.  Evenly spaced along the axis provided.

    **kwargs : passed directly to the `matplotlib` plot function

    Return
    ------
    axes : The axes the data are plotted on
    """

    axes.scatter(hist[0], np.zeros(np.shape(hist[0])), color=COLORS['histogram'], marker='|', s=100, label='Histogram Bin Ends', alpha=0.75)
    npoints = kwargs.pop('npoints', 101)
    xlim = axes.get_xlim()
    xvals = np.linspace(xlim[0], xlim[1], npoints)
    hinterpolated = utils.approximate_dist(dist, xvals)
    axes.plot(xvals, hinterpolated[1], color=COLORS['histogram'], lw=2.0, alpha=1.0, linestyle=STYLES['histogram'], label='Histogram Interpolated PDF')
    return axes


def plot_hist_dist_pdf(axes, dist, **kwargs):
    """
    Plot a PDF on a set of axes, by evaluating it at the histogram bins carried by the distribution

    Parameters
    ----------
    axes : The axes we want to plot the data on

    dist : `scipy.stats.rv_continuous`
        The distribution we want to plot

    Keywords
    --------
    **kwargs : passed directly to the `matplotlib` plot function

    Return
    ------
    axes : The axes the data are plotted on
    """

    return plot_pdf_histogram_on_axes(axes, dist, hist=dist._histogram[1], **kwargs)


def plot_pdf_samples_on_axes(axes, dist, samples, **kwargs):
    """
    Plot a PDF on a set of axes, by displaying a set of samples from the PDF

    Parameters
    ----------
    axes : The axes we want to plot the data on

    dist : `scipy.stats.rv_continuous`
        The distribution we want to plot

    samples : `np.array`
        Points sampled from the PDF

    Keywords
    --------
    **kwargs : passed directly to the `matplotlib` plot function

    Return
    ------
    axes : The axes the data are plotted on
    """

    axes.scatter(samples, np.zeros(np.shape(samples)), color=COLORS['samples'], marker='|', s=100, label='Samples', alpha=0.75)
    npoints = kwargs.pop('npoints', 101)
    xlim = axes.get_xlim()
    xvals = np.linspace(xlim[0], xlim[1], npoints)
    sinterpolated = utils.approximate_dist(dist, xvals)
    plt.plot(xvals, sinterpolated[1], color=COLORS['samples'], lw=2.0, alpha=1.0, linestyle=STYLES['samples'], label='Samples Interpolated PDF')
    return axes

def plot_kde_dist_pdf(axes, dist, **kwargs):
    """
    Plot a PDF on a set of axes, by displaying the samples carried around by the sample

    Parameters
    ----------
    axes : The axes we want to plot the data on

    dist : `scipy.stats.rv_continuous`
        The distribution we want to plot

    Keywords
    --------
    **kwargs : passed directly to the `matplotlib` plot function

    Return
    ------
    axes : The axes the data are plotted on
    """

    return plot_pdf_samples_on_axes(axes, dist, dist.samples, **kwargs)



def make_figure_axes(limits, **kwargs):
    """
    Build a figure and a set of figure axes to plot data on

    Parameters
    ----------
    limits : (float, float)
        The x-axis limits of the plot

    Keywords
    --------
    **kwargs : passed directly to the `matplotlib` plot function

    Return
    ------
    fig, axes : The figure and axes
    """

    xlabel = kwargs.pop('xlabel', r'$z$')
    ylabel = kwargs.pop('ylabel', r'$p(z)$')

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.set_xlim(limits[0], limits[-1])
    axes.set_xlabel(xlabel, fontsize=16)
    axes.set_ylabel(ylabel, fontsize=16)

    return (fig, axes)



NATIVE_PLOT_DICT = {}


def qp_plot_native(axes, dist, **kwargs):
    """
    Plot a PDF using the native representation for that type of distribution

    Parameters
    ----------
    axes : The axes we want to plot the data on

    dist : `scipy.stats.rv_continuous`
        The distribution we want to plot

    Keywords
    --------
    **kwargs : passed directly to the `matplotlib` plot function

    Return
    ------
    fig, axes : The figure and axes the data are plotted on

    Notes
    -----
    The native representations are defined by a dict that maps class to a particular plotting function
    """
    limits = kwargs.pop('limits', None)
    if axes is None:
        fig, axes = make_figure_axes(limits=limits)
    else:
        fig = axes.figure

    func = NATIVE_PLOT_DICT.get(type(dist), plot_dist_pdf)
    func(axes, dist, **kwargs)
    fig.legend()
    return fig, axes


def qp_add_plot_native_func(func, cls):
    """
    Add a mapping to the native representation functions.

    Parameters
    ----------
    func : `function`
        The plotting function to use for this class

    cls : `class`
        The class we are adding to the mapping
    """

    NATIVE_PLOT_DICT[cls] = func



qp_add_plot_native_func(plot_xval_dist_pdf, interp_dist)
qp_add_plot_native_func(plot_xval_dist_pdf, spline_dist)
qp_add_plot_native_func(plot_xval_dist_pdf, intspline_dist)
qp_add_plot_native_func(plot_kde_dist_pdf, kde_dist)
qp_add_plot_native_func(plot_quantile_dist_pdf, quantile_dist)
qp_add_plot_native_func(plot_hist_dist_pdf, rv_histogram)
