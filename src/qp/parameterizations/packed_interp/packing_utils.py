"""Integer packing utilities for qp"""

from __future__ import annotations

import enum

import numpy as np
from numpy.typing import ArrayLike


class PackingType(enum.Enum):
    linear_from_rowmax = 0
    log_from_rowmax = 1


def linear_pack_from_rowmax(input_array: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Pack an array into 8bit unsigned integers, using the maximum of each row as a reference

    This packs the values onto a linear grid for each row, running from 0 to row_max

    Parameters
    ----------
    input_array : ArrayLike
        The values we are packing

    Returns
    -------
    packed_array : np.ndarray
        The packed values
    row_max : np.ndarray
        The max for each row, need to unpack the array
    """
    row_max = np.expand_dims(input_array.max(axis=1), -1)
    return np.round(255 * input_array / row_max).astype(np.uint8), row_max


def linear_unpack_from_rowmax(
    packed_array: ArrayLike, row_max: ArrayLike
) -> np.ndarray[float]:
    """Unpack an array into 8bit unsigned integers, using the maximum of each row as a reference

    Parameters
    ----------
    packed_array : ArrayLike
        The packed values
    row_max : ArrayLike
        The max for each row, need to unpack the array


    Returns
    -------
    unpacked_array : np.ndarray[float]
        The unpacked values
    """
    unpacked_array = row_max * packed_array / 255.0
    return unpacked_array


def log_pack_from_rowmax(
    input_array: ArrayLike, log_floor: float = -3.0
) -> tuple[np.ndarray[np.uint8], np.ndarray]:
    """Pack an array into 8bit unsigned integers, using the maximum of each row as a reference

    This packs the values onto a log grid for each row, running from row_max / 10**log_floor to row_max

    Parameters
    ----------
    input_array : ArrayLike
        The values we are packing
    log_floor : float, optional
        The logarithmic floor used for the packing, by default -3.

    Returns
    -------
    packed_array : np.ndarray[np.uint8]
        The packed values
    row_max : np.ndarray
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


def log_unpack_from_rowmax(
    packed_array: ArrayLike, row_max: ArrayLike, log_floor: float = -3.0
) -> np.ndarray:
    """Unpack an array into 8bit unsigned integers, using the maximum of each row as a reference

    Parameters
    ----------
    packed_array : ArrayLike
        The packed values
    row_max : ArrayLike
        The max for each row, need to unpack the array
    log_floor : float, optional
        The logarithmic floor used for the packing, -3 by default.

    Returns
    -------
    unpacked_array : np.ndarray
        The unpacked values
    """
    neg_log_floor = -1.0 * log_floor
    unpacked_array = row_max * np.where(
        packed_array == 0,
        0.0,
        np.power(10, neg_log_floor * ((packed_array / 255.0) - 1.0)),
    )
    return unpacked_array


def pack_array(packing_type: PackingType, input_array: ArrayLike, **kwargs):
    """Pack an array into 8bit unsigned integers

    Parameters
    ----------
    packing_type : PackingType
        Enum specifying the type of packing to use
    input_array : ArrayLike
        The values we are packing
    kwargs
        depend on the packing type used

    Returns
    -------
    np.ndarray
        Details depend on packing type used
    """

    if packing_type == PackingType.linear_from_rowmax:
        return linear_pack_from_rowmax(input_array)
    if packing_type == PackingType.log_from_rowmax:
        return log_pack_from_rowmax(input_array, kwargs.get("log_floor", -3))
    raise ValueError(
        f"Packing for packing type {packing_type} is not implemented"
    )  # pragma: no cover


def unpack_array(packing_type: PackingType, packed_array: ArrayLike, **kwargs):
    """Unpack an array from 8bit unsigned integers

    Parameters
    ----------
    packing_type : PackingType
        Enum specifying the type of packing to use
    packed_array : ArrayLike
        The packed values
    kwargs
        depend on the packing type used

    Returns
    -------
    np.ndarray
        Details depend on packing type used
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
        f"Unpacking for packing type {packing_type} is not implemented"
    )  # pragma: no cover
