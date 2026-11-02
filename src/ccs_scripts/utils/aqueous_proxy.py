"""
CO2 in the aqueous phase, derived by a mole balance, for restart files where the
mole fraction of CO2 in the aqueous phase (AMF) is not written.
"""

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from ccs_scripts.utils.restart_header import zmf_includes_water

WATER_ONLY_SATURATION = 1e-6
MIN_WATER_ONLY_CELLS = 10
NEGLIGIBLE_FRACTION = 1e-12

# How the convention was found, in the order they are tried
BASIS_ZMFW = "ZMFW"
BASIS_ZCOMPS = "ZCOMPS"
BASIS_CELLS = "the cells with only water"


class ZmfConvention(Enum):
    INCLUDES_WATER = "includes water"
    EXCLUDES_WATER = "excludes water"


def detect_zmf_convention_with_basis(
    zmf_total: np.ndarray,
    sgas: np.ndarray,
    soil: np.ndarray,
    zmfw: Optional[np.ndarray] = None,
    zcomps: Optional[List[str]] = None,
    n_zmf: Optional[int] = None,
) -> Tuple[Optional[ZmfConvention], Optional[str]]:
    """
    Whether the overall composition ZMF counts water, and how that was found. Tried in
    this order

    Returns (None, None) if none of them can tell.
    """
    if zmfw is not None:
        if abs(float(np.median(zmf_total + zmfw)) - 1.0) < 0.01:
            return ZmfConvention.INCLUDES_WATER, BASIS_ZMFW
        return None, None
    if zcomps is not None and n_zmf is not None:
        includes = zmf_includes_water(zcomps, n_zmf)
        if includes is not None:
            convention = (
                ZmfConvention.INCLUDES_WATER
                if includes
                else ZmfConvention.EXCLUDES_WATER
            )
            return convention, BASIS_ZCOMPS
    water_only = (sgas < WATER_ONLY_SATURATION) & (soil < WATER_ONLY_SATURATION)
    if int(water_only.sum()) < MIN_WATER_ONLY_CELLS:
        return None, None
    median = float(np.median(zmf_total[water_only]))
    if median > 0.99:
        return ZmfConvention.INCLUDES_WATER, BASIS_CELLS
    if median < 0.01:
        return ZmfConvention.EXCLUDES_WATER, BASIS_CELLS
    return None, None


def detect_zmf_convention(
    zmf_total: np.ndarray,
    sgas: np.ndarray,
    soil: np.ndarray,
    zmfw: Optional[np.ndarray] = None,
    zcomps: Optional[List[str]] = None,
    n_zmf: Optional[int] = None,
) -> Optional[ZmfConvention]:
    """The convention of detect_zmf_convention_with_basis(), without how it was found"""
    return detect_zmf_convention_with_basis(
        zmf_total, sgas, soil, zmfw=zmfw, zcomps=zcomps, n_zmf=n_zmf
    )[0]


def aqueous_mole_fraction(
    convention: ZmfConvention,
    sgas: np.ndarray,
    swat: np.ndarray,
    soil: Optional[np.ndarray],
    bgas: np.ndarray,
    bwat: np.ndarray,
    boil: Optional[np.ndarray],
    xmf: np.ndarray,
    ymf: np.ndarray,
    zmf: np.ndarray,
) -> np.ndarray:
    """
    Mole fraction of CO2 in the aqueous phase, per cell. Cells with no water get
    zero, and so do negative values, which are rounding errors.

    Args:
        sgas, swat, soil: Saturations. SOIL is taken as 1 - SGAS - SWAT if None.
        bgas, bwat, boil: Molar densities. BOIL can be None if there is no oil.
        xmf, ymf, zmf: Mole fraction of CO2 in the oil phase, the gas phase and
            overall.
    """
    if soil is None:
        soil = np.clip(1.0 - sgas - swat, 0.0, None)
    n_water = swat * bwat
    n_gas = sgas * bgas
    n_oil = soil * boil if boil is not None else np.zeros_like(sgas)

    if convention == ZmfConvention.INCLUDES_WATER:
        co2 = zmf * (n_water + n_gas + n_oil) - ymf * n_gas - xmf * n_oil
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            co2 = (zmf * (n_gas + n_oil) - ymf * n_gas - xmf * n_oil) / (1.0 - zmf)

    has_water = n_water > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(has_water, co2 / np.where(has_water, n_water, 1.0), 0.0)
    fraction = np.clip(np.nan_to_num(fraction, nan=0.0, posinf=0.0), 0.0, 1.0)
    fraction[fraction < NEGLIGIBLE_FRACTION] = 0.0
    return fraction
