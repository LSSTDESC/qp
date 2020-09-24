"""This module implements tools to convert between distributions"""

import sys

from scipy import stats

from .ensemble import Ensemble


def get_val_or_default(in_dict, key):
    """Helper functions to return either an item in a dictionary or the default value of the dictionary

    Parameters
    ----------
    in_dict : `dict`
        input dictionary
    key : `str`
        key to search for

    Returns
    -------
    out : `dict` or `function`
        The requested item

    Notes
    -----
    This will first try to return:
        in_dict[key] : i.e., the requested item.
    If that fails it will try
        in_dict[None] : i.e., the default for that dictionary.
    If that failes it will return
        None
    """
    if key in in_dict:
        return in_dict[key]
    if None in in_dict:
        return in_dict[None]
    return None


def set_val_or_default(in_dict, key, val):
    """Helper functions to either get and item from or add an item to a dictionary and return that item

    Parameters
    ----------
    in_dict : `dict`
        input dictionary
    key : `str`
        key to search for
    val : `dict` or `function`
        item to add to the dictionary

    Returns
    -------
    out : `dict` or `function`
        The requested item

    Notes
    -----
    This will first try to return:
        in_dict[key] : i.e., the requested item.
    If that fails it will try
        in_dict[None] : i.e., the default for that dictionary.
    If that failes it will return
        None
    """
    if key in in_dict:
        return in_dict[key]
    in_dict[key] = val
    return val




PRINT_PREFIXS = ["To ", "  From ", "    Method "]

def pretty_print(in_dict, idx=0, stream=sys.stdout):
    """Print a level of the converstion dictionary in a human-readable format

    Parameters
    ----------
    in_dict : `dict`
        input dictionary
    idx : `int`
        The level of the input dictionary we are currently printing
    stream : `stream`
        The stream to print to
    """
    prefix = PRINT_PREFIXS[idx]
    for key, val in in_dict.items():
        if key is None:
            key_str = "default"
        else:
            key_str = key
        if isinstance(val, dict):
            stream.write("%s%s:\n" % (prefix, key_str))
            pretty_print(val, idx+1, stream)
        else:
            stream.write("%s%s : %s\n" % (prefix, key_str, val))



class ConversionDict:
    """Dictionary of possible conversions

    Notes
    -----
    This dictionary is implemented as a triply nested dictionary,

    dict[class_to][class_from][method]

    At each level, `None` is used define the default behavior,
    i.e.,  the behavior if the additional arguements are not found
    in the dictionary.
    """

    def __init__(self):
        """Class constructor, builds an empty dictionary"""
        self._conv_dict = {}

    def _get_convertor(self, class_to, class_from, method=None):

        to_dict = get_val_or_default(self._conv_dict, class_to)
        if to_dict is None:
            return None
        to_from_dict = get_val_or_default(to_dict, class_from)
        if to_from_dict is None:
            return None
        func = get_val_or_default(to_from_dict, method)
        return func


    def _convert_dist(self, dist_from, class_to, method=None, **kwargs):

        if isinstance(dist_from, stats._distn_infrastructure.rv_frozen):
            class_from = dist_from.dist
        else:
            class_from = type(dist_from)
        convert = self._get_convertor(class_to, class_from, method)
        return convert(dist_from, class_to, **kwargs)


    def _convert_ensemble(self, ensemble_from, class_to, method=None, **kwargs):

        class_from = ensemble_from.dist_class
        convert = self._get_convertor(class_to, class_from, method)

        out_dists = [convert(dist, class_to, **kwargs) for dist in ensemble_from.dists]
        out = Ensemble(out_dists)
        return out


    def convert(self, obj_from, class_to, method=None, **kwargs):
        """Convert a distribution or ensemble

        Parameters
        ----------
        obj_from :  `scipy.stats.rv_continuous or qp.ensemble`
            Input object
        class_to : sub-class of `scipy.stats.rv_continuous`
            The class we are converting to
        method : `str`
            Optional argument to specify a non-default conversion algorithm
        kwargs : keyword arguments are passed to the output class constructor

        Notes
        -----
        If obj_from is a single distribution this will return a single distribution of
        type class_to.

        If obj_from is a `qp.Ensemble` this will return a `qp.Ensemble` of distributions
        of type class_to.
        """
        if isinstance(obj_from, Ensemble):
            return self._convert_ensemble(obj_from, class_to, method, **kwargs)
        if isinstance(obj_from, (stats._distn_infrastructure.rv_frozen, stats.rv_continuous)):
            return self._convert_dist(obj_from, class_to, method, **kwargs)
        raise TypeError("Tried to convert object of type %s" % type(obj_from))


    def add_mapping(self, func, class_to, class_from, method=None):
        """Add a mapping to this dictionary

        Parameters
        ----------
        func : `function`
            The function used to do the conversion
        class_to : sub-class of `scipy.stats.rv_continuous`
            The class we are converting to
        class_from :  sub-class of `scipy.stats.rv_continuous`
            The class we are converting from
        method : `str`
            Optional argument to specify a non-default conversion algorithm
        """
        to_dict = set_val_or_default(self._conv_dict, class_to, {})
        to_from_dict = set_val_or_default(to_dict, class_from, {})
        ret_func = set_val_or_default(to_from_dict, method, func)
        return ret_func

    def pretty_print(self, stream=sys.stdout):
        """Print a level of the converstion dictionary in a human-readable format

        Parameters
        ----------
        stream : `stream`
            The stream to print to
        """
        pretty_print(self._conv_dict, stream=stream)


CONVERSIONS = ConversionDict()

def qp_convert(obj_from, class_to, method=None, **kwargs):
    """Convert a distribution or ensemble

    Parameters
    ----------
    obj_from :  `scipy.stats.rv_continuous or qp.ensemble`
        Input object
    class_to : sub-class of `scipy.stats.rv_continuous`
        The class we are converting to
    method : `str`
        Optional argument to specify a non-default conversion algorithm
    kwargs : keyword arguments are passed to the output class constructor

    Notes
    -----
    If obj_from is a single distribution this will return a single distribution of
    type class_to.

    If obj_from is a `qp.Ensemble` this will return a `qp.Ensemble` of distributions
    of type class_to.
    """
    return CONVERSIONS.convert(obj_from, class_to, method, **kwargs)


def qp_add_mapping(func, class_to, class_from, method=None):
    """
    Parameters
    ----------
    func : `function`
        The function used to do the conversion
    class_to : sub-class of `scipy.stats.rv_continuous`
        The class we are converting to
    class_from :  sub-class of `scipy.stats.rv_continuous`
        The class we are converting from
    method : `str`
        Optional argument to specify a non-default conversion algorithm
    """
    return CONVERSIONS.add_mapping(func, class_to, class_from, method)
