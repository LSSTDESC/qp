"""This module implements tools to handle dictionaries"""

import sys
from typing import Any, Mapping

import numpy as np

from .array import reshape_to_pdf_shape


def get_val_or_default(in_dict: dict, key: str) -> Any | None:
    """Helper functions to return either an item in a dictionary or the default value of the dictionary

    Parameters
    ----------
    in_dict : dict
        input dictionary
    key : str
        key to search for

    Returns
    -------
    out : Any | None
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


def set_val_or_default(in_dict: dict, key: str, val: Any):
    """Helper functions to either get and item from or add an item to a dictionary and return that item

    Parameters
    ----------
    in_dict : dict
        input dictionary
    key : str
        key to search for
    val : Any
        item to add to the dictionary

    Returns
    -------
    out : Any
        If the key already existed, return the current value. Otherwise, return `val`.

    Notes
    -----
    This will first try to return:
      in_dict[key] : i.e., the requested item.
    If that fails it will return
      `val`
    """
    if key in in_dict:
        return in_dict[key]
    in_dict[key] = val
    return val


def pretty_print(in_dict: dict, prefixes: list, idx=0, stream=sys.stdout) -> None:
    """Print a level of the converstion dictionary in a human-readable format

    Parameters
    ----------
    in_dict : dict
        input dictionary
    prefixs : list
        The prefixs to use at each level of the printing
    idx : int
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


def print_dict_shape(in_dict: dict) -> None:
    """Print the shape of arrays in a dictionary.
    This is useful for debugging table creation.

    Parameters
    ----------
    in_dict : dict
        The dictionary to print
    """
    for key, val in in_dict.items():
        print(key, np.shape(val))


def slice_dict(in_dict: dict, subslice: int | slice) -> dict:
    """Create a new dict by taking a slice of of every array in a dict

    Parameters
    ----------
    in_dict : dict
        The dictionary to conver
    subslice : int or slice
        Used to slice the arrays

    Returns
    -------
    out_dict : dict
        The converted dictionary
    """

    out_dict = {}
    for key, val in in_dict.items():
        try:
            out_dict[key] = val[subslice]
        except (KeyError, TypeError):
            out_dict[key] = val
    return out_dict


def check_keys(in_dicts: list[dict]) -> None:
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


def concatenate_dicts(in_dicts: list[dict], add_axis: int = 0) -> dict:
    """Create a new dict by concatenate each array in `in_dicts`

    Parameters
    ----------
    in_dicts : list[dict]
        The dictionaries to stack

    add_axis: int
        The axis to add when ensuring an array is 2D

    Returns
    -------
    out_dict : dict
        The stacked dicionary
    """
    if not in_dicts:  # pragma: no cover
        return {}
    check_keys(in_dicts)
    out_dict = {key: None for key in in_dicts[0].keys()}
    for key in out_dict.keys():
        out_dict[key] = np.concatenate(
            [ensure_2d_array(in_dict[key], add_axis) for in_dict in in_dicts]
        )
    return out_dict


def check_array_shapes(in_dict: dict, npdf: int) -> None:
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


def compare_two_dicts(d1: dict, d2: dict) -> bool:
    """Check that all the items in d1 and d2 match

    Returns
    -------
    match : bool
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


def compare_dicts(in_dicts: list[dict]) -> bool:
    """Check that all the dicts in in_dicts match

    Returns
    -------
    match : bool
        True if they all match, False otherwise
    """
    if not in_dicts:  # pragma: no cover
        return True
    first_dict = in_dicts[0]
    for in_dict in in_dicts[1:]:
        if not compare_two_dicts(first_dict, in_dict):  # pragma: no cover
            return False
    return True


def reduce_arrays_to_1d(in_dict: Mapping) -> Mapping:
    """Checks if any arrays in the dictionary have ndim greater than 1, and
    if the first dimension is equal to 1 it reshapes the array to remove that dimension.

    Parameters
    ----------
    in_dict : Mapping
        A dictionary of array-like objects.

    Returns
    -------
    Mapping
        The updated dictionary.
    """

    for key, value in in_dict.items():
        if np.ndim(value) > 1:
            # if shape of any objdata is (1,n) reshape to (,n)
            if np.shape(value)[0] == 1:
                new_val = np.reshape(value, np.shape(value)[-1:])
                in_dict[key] = new_val

    return in_dict


def make_len_equal(in_dict: Mapping[str, np.ndarray], l_arr: int = 1) -> Mapping:
    """Ensures that all arrays in the dictionary have `shape[0]` of at least l_arr.

    This essentially assures that a dictionary of numpy arrays is a `numpyDict`
    `Table-like` object according to `tables_io` to allow for writing.

    Parameters
    ----------
    in_dict : Mapping
        A dictionary of array-like objects.
    l_arr : int, optional
        The value of shape[0] to ensure, by default 1

    Returns
    -------
    Mapping
        The updated dictionary.
    """

    for key, value in in_dict.items():
        if np.shape(value)[0] > l_arr:
            # add a dimension to axis=0
            in_dict[key] = np.expand_dims(value, 0)

    return in_dict


def expand_dimensions(
    in_dict: Mapping[str, np.ndarray], npdf: int, nvals: int
) -> Mapping:

    for key, value in in_dict.items():
        in_dict[key] = reshape_to_pdf_shape(value, npdf, nvals)

    return in_dict


def ensure_2d_array(arr: np.ndarray, add_axis: int = 0) -> np.ndarray:
    """Makes sure that the input array is at least 2 dimensions, by adding a new axis
    to 1D arrays. By default, the new axis is added as axis=0, so the new array will
    have shape (1, len(arr)).

    Parameters
    ----------
    arr : np.ndarray
        The input array
    add_axis : int
        Where to add the new axis.

    Returns
    -------
    np.ndarray
        Returns the input array if it's 2D, or an array with an extra dimension if
        given a 1D array.

    """

    if np.ndim(arr) == 1:
        # make array 2D by adding a dimension to given
        return np.expand_dims(arr, add_axis)
    elif np.ndim(arr) >= 2:
        # array is already 2D, no changes needed
        return arr
    elif np.ndim(arr) == 0:
        # this should return the given number with the appropriate number of dimensions
        return np.atleast_2d(arr)
