"""This module implements tools to convert between distributions"""

import sys

import numpy as np


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
    If that fails it will return
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
    If that fails it will return
      None
    """
    if key in in_dict:
        return in_dict[key]
    in_dict[key] = val
    return val


def pretty_print(in_dict, prefixes, idx=0, stream=sys.stdout):
    """Print a level of the converstion dictionary in a human-readable format

    Parameters
    ----------
    in_dict : `dict`
        input dictionary
    prefixs : `list`
        The prefixs to use at each level of the printing
    idx : `int`
        The level of the input dictionary we are currently printing
    stream : `stream`
        The stream to print to
    """
    prefix = prefixes[idx]
    for key, val in in_dict.items():
        if key is None:
            key_str = "default"
        else:
            key_str = key
        if isinstance(val, dict):  # pragma: no cover
            stream.write("%s%s:\n" % (prefix, key_str))
            pretty_print(val, prefixes, idx + 1, stream)
        else:
            stream.write("%s%s : %s\n" % (prefix, key_str, val))


def print_dict_shape(in_dict):
    """Print the shape of arrays in a dictionary.
    This is useful for debugging table creation.

    Parameters
    ----------
    in_dict : `dict`
        The dictionary to print
    """
    for key, val in in_dict.items():
        print(key, np.shape(val))


def slice_dict(in_dict, subslice):
    """Create a new `dict` by taking a slice of of every array in a `dict`

    Parameters
    ----------
    in_dict : `dict`
        The dictionary to conver
    subslice : `int` or `slice`
        Used to slice the arrays

    Returns
    -------
    out_dict : `dict`
        The converted dicionary
    """

    out_dict = {}
    for key, val in in_dict.items():
        try:
            out_dict[key] = val[subslice]
        except (KeyError, TypeError):
            out_dict[key] = val
    return out_dict


def check_keys(in_dicts):
    """Check that the keys in all the in_dicts match

    Raises KeyError if one does not match.
    """
    if not in_dicts:  # pragma: no cover
        return
    master_keys = in_dicts[0].keys()
    for in_dict in in_dicts[1:]:
        if in_dict.keys() != master_keys:  # pragma: no cover
            raise ValueError(
                "Keys to not match: %s != %s" % (in_dict.keys(), master_keys)
            )


def concatenate_dicts(in_dicts):
    """Create a new `dict` by concatenate each array in `in_dicts`

    Parameters
    ----------
    in_dicts : `list`
        The dictionaries to stack

    Returns
    -------
    out_dict : `dict`
        The stacked dicionary
    """
    if not in_dicts:  # pragma: no cover
        return {}
    check_keys(in_dicts)
    out_dict = {key: None for key in in_dicts[0].keys()}
    for key in out_dict.keys():
        out_dict[key] = np.concatenate([in_dict[key] for in_dict in in_dicts])
    return out_dict


def check_array_shapes(in_dict, npdf):
    """Check that all the arrays in in_dict match the number of pdfs

    Raises ValueError if one does not match.
    """
    if in_dict is None:
        return
    for key, val in in_dict.items():
        if np.size(val) == 1 and npdf == 1:  # pragma: no cover
            continue
        if np.shape(val)[0] != npdf:  # pragma: no cover
            raise ValueError(
                "First dimension of array %s does not match npdf: %i != %i"
                % (key, np.shape(val)[0], npdf)
            )


def compare_two_dicts(d1, d2):
    """Check that all the items in d1 and d2 match

    Returns
    -------
    match : `bool`
        True if they all match, False otherwise
    """
    if d1.keys() != d2.keys():  # pragma: no cover
        return False
    for k, v in d1.items():
        vv = d2[k]
        try:
            if v != vv:  # pragma: no cover
                return False
        except ValueError:
            if not np.allclose(v, vv):  # pragma: no cover
                return False
    return True


def compare_dicts(in_dicts):
    """Check that all the dicts in in_dicts match

    Returns
    -------
    match : `bool`
        True if they all match, False otherwise
    """
    if not in_dicts:  # pragma: no cover
        return True
    first_dict = in_dicts[0]
    for in_dict in in_dicts[1:]:
        if not compare_two_dicts(first_dict, in_dict):  # pragma: no cover
            return False
    return True
