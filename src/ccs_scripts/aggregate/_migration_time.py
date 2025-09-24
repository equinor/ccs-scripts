import datetime
from typing import List

import numpy as np
import xtgeo

MIGRATION_TIME_PNAME = "MigrationTime"


def generate_migration_time_property(
    co2_props: List[xtgeo.GridProperty],
    co2_threshold: float,
) -> xtgeo.GridProperty:
    """
    Calculates a 3D grid property reflecting the migration time. Migration time is
    defined as the first time step at which the property value exceeds its initial
    condition
    """
    # Calculate time since simulation start
    times = [datetime.datetime.strptime(_prop.date, "%Y%m%d") for _prop in co2_props]
    time_since_start = [(t - times[0]).days / 365 for t in times]
    # Duplicate first property to ensure equal actnum
    prop_name = co2_props[0].name.split("--")[0]
    t_prop = co2_props[0].copy(newname=MIGRATION_TIME_PNAME + "_" + prop_name)
    t_prop.values[~t_prop.values.mask] = np.inf
    for co2, dt in zip(
        co2_props[1:],
        time_since_start[1:],
    ):
        diff_prop = co2.values - co2_props[0].values
        above_threshold = diff_prop > co2_threshold
        t_prop.values[above_threshold] = np.minimum(t_prop.values[above_threshold], dt)
    # Mask inf values
    if not isinstance(t_prop.values.mask, np.ndarray):
        t_prop.values.mask = np.asarray(t_prop.values.mask)
    t_prop.values.mask[np.isinf(t_prop.values)] = 1
    return t_prop
