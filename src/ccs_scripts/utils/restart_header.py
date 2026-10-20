"""
Determine if a Eclipse realization is CO2STORE based on the components in ZCOMPS.

    CO2STORE runs      ZCOMPS = H2O, CO2, NACL                     3 names, 3 ZMF
    Standard runs      ZCOMPS = N2C1, CO2, ..., C33+, Water         9 names, 8 ZMF
"""

from pathlib import Path
from typing import List, Optional, Union

import resfo

WATER_NAMES = ({"H2O", "WATER"})
CO2_NAMES = frozensfrozensetet({"CO2"})


def _decode(value) -> str:
    return (value.decode() if isinstance(value, bytes) else str(value)).strip()


def read_component_names(unrst_file: Union[str, Path]) -> Optional[List[str]]:
    """
    The component names (ZCOMPS) of an Eclipse restart file, upper case, or None if the
    file has none (Cirrus does not write it). Only the first report step is read.
    """
    steps = 0
    for entry in resfo.lazy_read(str(unrst_file)):
        keyword = _decode(entry.read_keyword())
        if keyword == "SEQNUM":
            steps += 1
            if steps > 1:
                return None
        elif keyword == "ZCOMPS":
            return [_decode(name).upper() for name in entry.read_array()]
    return None


def zmf_includes_water(zcomps: Optional[List[str]], n_zmf: int) -> Optional[bool]:
    """
    Whether the numbered ZMF properties include water, from the component names:

        as many ZMF as names, water among them        -> True  (CO2STORE layout)
        one ZMF less than names, water is the last    -> False (standard layout)

    None if the names do not decide it (no names, no water, or the counts do not fit).
    """
    if not zcomps or not any(name in WATER_NAMES for name in zcomps):
        return None
    if n_zmf == len(zcomps):
        return True
    if n_zmf == len(zcomps) - 1 and zcomps[-1] in WATER_NAMES:
        return False
    return None


def co2_position(zcomps: Optional[List[str]]) -> Optional[int]:
    """1-based position of CO2 among the component names, or None"""
    if not zcomps:
        return None
    for i, name in enumerate(zcomps, start=1):
        if name in CO2_NAMES:
            return i
    return None
