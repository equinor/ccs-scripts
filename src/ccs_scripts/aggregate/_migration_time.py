import datetime
import logging
from typing import List, Optional

import numpy as np
import xtgeo

from ccs_scripts.utils.utils import format_error

MIGRATION_TIME_PNAME = "MigrationTime"


def generate_migration_time_property(
    co2_props: List[xtgeo.GridProperty],
    co2_threshold: float,
    first_injection_year: Optional[int] = None,
) -> xtgeo.GridProperty:
    """
    Calculates a 3D grid property reflecting the migration time. Migration time is
    defined as the first time step at which the property value exceeds its initial
    condition or a threshold
    """
    # Calculate time since simulation start
    times = [datetime.datetime.strptime(_prop.date, "%Y%m%d") for _prop in co2_props]
    reference_date = (
        times[0]
        if first_injection_year is None
        else datetime.datetime(first_injection_year, 1, 1)
    )
    logging.info("Reference_date for migration time: %s\n", reference_date.date())
    time_since_start = [(t - reference_date).days / 365 for t in times]
    first_positive_idx = next(
        (idx for idx, t in enumerate(time_since_start) if t > 0),
        None,
    )
    if first_positive_idx is None:
        error_text = "No date occurs after the migration time reference date."
        raise ValueError(format_error(error_text))
    idx_start = max(first_positive_idx - 1, 0)
    # Duplicate first property to ensure equal actnum
    prop_name = co2_props[0].name.split("--")[0]
    t_prop = co2_props[0].copy(newname=MIGRATION_TIME_PNAME + "_" + prop_name)
    t_prop.values[~t_prop.values.mask] = np.inf
    baseline_values = co2_props[idx_start].values
    for co2, dt in zip(
        co2_props[idx_start:],
        time_since_start[idx_start:],
    ):
        if dt == 0:
            # CO2 is already meaningfully present at the baseline time
            above_threshold = baseline_values > co2_threshold
        else:
            # CO2 has increased significantly compared to baseline
            diff_prop = co2.values - baseline_values
            above_threshold = diff_prop > co2_threshold
        t_prop.values[above_threshold] = np.minimum(t_prop.values[above_threshold], dt)

    # Mask inf values
    if not isinstance(t_prop.values.mask, np.ndarray):
        t_prop.values.mask = np.asarray(t_prop.values.mask)
    t_prop.values.mask[np.isinf(t_prop.values)] = 1
    return t_prop
