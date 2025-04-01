import datetime
from typing import List, Union

import numpy as np
import xtgeo

MIGRATION_TIME_PNAME = "MigrationTime"


def generate_migration_time_property(
    co2_props: List[xtgeo.GridProperty],
    co2_threshold: Union[float, List],
):
    """
    Calculates a 3D grid property reflecting the migration time. Migration time is
    defined as the first time step at which the property value exceeds the provided
    `lower_threshold`.
    """
    # Calculate time since simulation start
    times = [datetime.datetime.strptime(_prop.date, "%Y%m%d") for _prop in co2_props]
    time_since_start = [(t - times[0]).days / 365 for t in times]
    # Duplicate first property to ensure equal actnum
    prop_name = co2_props[0].name.split("--")[0]
    t_props = {
        prop_name: co2_props[0].copy(newname=MIGRATION_TIME_PNAME + "_" + prop_name)
    }
    t_props[prop_name].values[~t_props[prop_name].values.mask] = np.inf
    for co2, dt in zip(
        co2_props,
        time_since_start,
    ):
        above_threshold = co2.values > co2_threshold
        t_props[prop_name].values[above_threshold] = np.minimum(
            t_props[prop_name].values[above_threshold], dt
        )
    # Mask inf values
    if not isinstance(t_props[prop_name].values.mask, np.ndarray):
        t_props[prop_name].values.mask = np.asarray(t_props[prop_name].values.mask)
    t_props[prop_name].values.mask[np.isinf(t_props[prop_name].values)] = 1
    return t_props
