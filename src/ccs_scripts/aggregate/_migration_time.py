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
    print(f"\n\n\n\ngenerate_migration_time_property()")
    print(f"co2_threshold   : {co2_threshold}")
    # Calculate time since simulation start
    times = [datetime.datetime.strptime(_prop.date, "%Y%m%d") for _prop in co2_props]
    print(f"times           : {times}")
    time_since_start = [(t - times[0]).days / 365 for t in times]
    print(f"time_since_start: {time_since_start}")

    # Duplicate first property to ensure equal actnum
    prop_name = co2_props[0].name.split("--")[0]
    print(f"prop_name       : {prop_name}")
    t_prop = co2_props[0].copy(newname=MIGRATION_TIME_PNAME + "_" + prop_name)

    if False:
        t_prop.values[~t_prop.values.mask] = np.inf

        for co2, dt in zip(
            co2_props[1:],
            time_since_start[1:],
        ):
            print(f"    dt: {dt}")
            diff_prop = co2.values - co2_props[0].values
            above_threshold = diff_prop > co2_threshold
            t_prop.values[above_threshold] = np.minimum(t_prop.values[above_threshold], dt)

    else:
        if prop_name == "SGAS":
            co2_threshold = 0.05
        elif prop_name == "AMFG":
            co2_threshold = 0.01
        t_prop.values[~t_prop.values.mask] = 0.0  # time_since_start[-1]
        dt_prev = time_since_start[-1]
        for co2, dt in zip(
            reversed(co2_props[:-1]),
            reversed(time_since_start[:-1]),
        ):
            # dt_prev... NBNB-AS
            print(f"    dt: {dt}")
            diff_prop = co2.values - co2_props[-1].values
            # print(f"    diff_prop.mean(): {diff_prop.mean()} (#values: {diff_prop.size})")
            diff_prop = abs(diff_prop)
            # print(f"    diff_prop.mean()     : {diff_prop.mean()} (#values: {diff_prop.size})")
            above_threshold = diff_prop > co2_threshold
            print(f"    above_threshold.sum(): {above_threshold.sum()} (#values: {above_threshold.size})")
            t_prop.values[above_threshold] = np.maximum(t_prop.values[above_threshold], dt_prev)
            dt_prev = dt
            # Print summary of unique values on t_prop.values:
            unique, counts = np.unique(t_prop.values, return_counts=True)
            for a, b in zip(unique, counts):
                print(f"        t_prop.values: {a:.4f} (#cells: {b})")

    # Mask inf values
    if not isinstance(t_prop.values.mask, np.ndarray):
        t_prop.values.mask = np.asarray(t_prop.values.mask)
    t_prop.values.mask[np.isinf(t_prop.values)] = 1

    return t_prop
