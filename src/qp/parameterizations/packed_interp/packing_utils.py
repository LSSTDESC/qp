"""Integer packing utilities for qp"""

import enum
import numpy as np


class PackingType(enum.Enum):
    linear_from_rowmax = 0
    log_from_rowmax = 1


def linear_pack_from_rowmax(input_array):
    """Pack an array into 8bit unsigned integers, using the maximum of each row as a refrence

    This packs the values onto a linear grid for each row, running from 0 to row_max

    Parameters
    ----------
    input_array : array_like
        The values we are packing

    Returns
    -------
    packed_array : array_like
        The packed values
    row_max : array_like
        The max for each row, need to unpack the array
    """
    row_max = np.expand_dims(input_array.max(axis=1), -1)
    return np.round(255 * input_array / row_max).astype(np.uint8), row_max


def linear_unpack_from_rowmax(packed_array, row_max):
    """Unpack an array into 8bit unsigned integers, using the maximum of each row as a refrence

    Parameters
    ----------
    packed_array : array_like
        The packed values
    row_max : array_like
        The max for each row, need to unpack the array


    Returns
    -------
    unpacked_array : array_like
        The unpacked values
    """
    unpacked_array = row_max * packed_array / 255.0
    return unpacked_array


def log_pack_from_rowmax(input_array, log_floor=-3.0):
    """Pack an array into 8bit unsigned integers, using the maximum of each row as a refrence

    This packs the values onto a log grid for each row, running from row_max / 10**log_floor to row_max

    Parameters
    ----------
    input_array : array_like
        The values we are packing
    log_floor: float
        The logarithmic floor used for the packing

    Returns
    -------
    packed_array : array_like
        The packed values
    row_max : array_like
        The max for each row, need to unpack the array
    """
    neg_log_floor = -1.0 * log_floor
    epsilon = np.power(10.0, 3 * log_floor)
    row_max = np.expand_dims(input_array.max(axis=1), -1)
    return (
        np.round(
            255
            * (np.log10((input_array + epsilon) / row_max) + neg_log_floor)
            / neg_log_floor
        )
        .clip(0.0, 255.0)
        .astype(np.uint8),
        row_max,
    )


def log_unpack_from_rowmax(packed_array, row_max, log_floor=-3.0):
    """Unpack an array into 8bit unsigned integers, using the maximum of each row as a refrence

    Parameters
    ----------
    packed_array : array_like
        The packed values
    row_max : array_like
        The max for each row, need to unpack the array
    log_floor: float
        The logarithmic floor used for the packing

    Returns
    -------
    unpacked_array : array_like
        The unpacked values
    """
    neg_log_floor = -1.0 * log_floor
    unpacked_array = row_max * np.where(
        packed_array == 0,
        0.0,
        np.power(10, neg_log_floor * ((packed_array / 255.0) - 1.0)),
    )
    return unpacked_array


def pack_array(packing_type, input_array, **kwargs):
    """Pack an array into 8bit unsigned integers

    Parameters
    ----------
    packing_type : PackingType
        Enum specifing the type of packing to use
    input_array : array_like
        The values we are packing

    Return values and keyword argument depend on the packing type used
    """

    if packing_type == PackingType.linear_from_rowmax:
        return linear_pack_from_rowmax(input_array)
    if packing_type == PackingType.log_from_rowmax:
        return log_pack_from_rowmax(input_array, kwargs.get("log_floor", -3))
    raise ValueError(
        f"Packing for packing type {packing_type} is not implemetned"
    )  # pragma: no cover


def unpack_array(packing_type, packed_array, **kwargs):
    """Unpack an array from 8bit unsigned integers

    Parameters
    ----------
    packing_type : PackingType
        Enum specifing the type of packing to use
    packed_array : array_like
        The packed values

    Return values and keyword argument depend on the packing type used
    """
    if packing_type == PackingType.linear_from_rowmax:
        return linear_unpack_from_rowmax(packed_array, row_max=kwargs.get("row_max"))
    if packing_type == PackingType.log_from_rowmax:
        return log_unpack_from_rowmax(
            packed_array,
            row_max=kwargs.get("row_max"),
            log_floor=kwargs.get("log_floor", -3),
        )
    raise ValueError(
        f"Unpacking for packing type {packing_type} is not implemetned"
    )  # pragma: no cover
