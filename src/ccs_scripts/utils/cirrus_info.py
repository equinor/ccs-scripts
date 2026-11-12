"""
Molar masses from the Cirrus info file, used to convert the mass densities that
Cirrus COMP writes (DGAS, DWAT, DOIL) to molar densities.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ccs_scripts.utils.utils import format_error

WATER_MOLAR_MASS = 18.0


def default_info_file(case: str) -> Optional[str]:
    """
    The info file next to the case, <case>_INFO.CSV (or .csv): the same default as
    in CO2 containment. None if there is none.
    """
    base = case[: -len(".EGRID")] if case.endswith(".EGRID") else case
    for suffix in ("_INFO.CSV", "_INFO.csv"):
        if os.path.isfile(base + suffix):
            return base + suffix
    return None


def read_molar_masses(info_file: str) -> Dict[str, Tuple[int, float]]:
    """
    Component name -> (position, molar mass), from the MW_<component> rows of the
    info file. The position is the order of the rows, starting at 1. MW_CO2 is
    required.
    """
    info = pd.read_csv(info_file)
    info.columns = info.columns.str.strip()
    info["Mnemonic"] = info["Mnemonic"].str.strip()
    rows = info.loc[info["Mnemonic"].str.startswith("MW_", na=False)]
    molar_masses = {
        row["Mnemonic"][len("MW_") :]: (i + 1, float(row["Value"]))
        for i, (_, row) in enumerate(rows.iterrows())
    }
    if "CO2" not in molar_masses:
        text = f"MW_CO2 not found in the Cirrus info file {info_file}"
        raise ValueError(format_error(text))
    return molar_masses


def molar_densities(
    molar_masses: Dict[str, Tuple[int, float]],
    xmf: Dict[int, np.ndarray],
    ymf: Dict[int, np.ndarray],
    dgas: np.ndarray,
    dwat: np.ndarray,
    doil: Optional[np.ndarray],
    water_molar_mass: float = WATER_MOLAR_MASS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Molar densities (bgas, bwat, boil) for one date, from mass densities. xmf and
    ymf are by component position. A phase that is not there (zero average molar
    mass) gets a molar density of zero.
    """
    gas_mass = np.sum([mw * ymf[pos] for pos, mw in molar_masses.values()], axis=0)
    oil_mass = np.sum([mw * xmf[pos] for pos, mw in molar_masses.values()], axis=0)
    bgas = np.divide(dgas, gas_mass, out=np.zeros_like(dgas), where=gas_mass > 0)
    boil = np.zeros_like(dgas)
    if doil is not None:
        boil = np.divide(doil, oil_mass, out=np.zeros_like(doil), where=oil_mass > 0)
    bwat = dwat / water_molar_mass
    return bgas, bwat, boil
