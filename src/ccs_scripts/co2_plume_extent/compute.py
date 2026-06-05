"""Compute service for plume extent and distance calculations."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ccs_scripts.co2_plume_extent.config import (
    DEFAULT_THRESHOLD_DISSOLVED,
    DEFAULT_THRESHOLD_GAS,
    Calculation,
    CalculationType,
    LineDirection,
)
from ccs_scripts.co2_plume_tracking.co2_plume_tracking import (
    calculate_plume_groups,
    load_plume_tracking_data,
)
from ccs_scripts.co2_plume_tracking.utils import (
    GridData,
    InjectionWellData,
    assemble_plume_groups_into_dict,
)
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import format_warning


def _calculate_grid_cell_distances(
    inj_wells: Optional[List[InjectionWellData]],
    calculation_type: CalculationType,
    x_co2: np.ndarray,
    y_co2: np.ndarray,
    config: Calculation,
) -> Dict[str, np.ndarray]:
    timer = Timer()
    timer.start("calculate_grid_cell_distances")
    dist = {}
    if calculation_type == CalculationType.PLUME_EXTENT:
        if inj_wells is None or len(inj_wells) == 0:
            dist["WELL"] = np.sqrt((x_co2 - config.x) ** 2 + (y_co2 - config.y) ** 2)
        else:
            for well in inj_wells:
                dist[well.name] = np.sqrt((x_co2 - well.x) ** 2 + (y_co2 - well.y) ** 2)
    elif calculation_type == CalculationType.POINT:
        dist["ALL"] = np.sqrt((x_co2 - config.x) ** 2 + (y_co2 - config.y) ** 2)
    elif calculation_type == CalculationType.LINE:
        if config.direction in (LineDirection.NORTH, LineDirection.SOUTH):
            line_value = config.y
            coords = y_co2
        else:
            line_value = config.x
            coords = x_co2
        factor = (
            -1 if config.direction in (LineDirection.WEST, LineDirection.SOUTH) else 1
        )
        dist["ALL"] = np.maximum(factor * (line_value - coords), 0.0)

    text = ""
    if calculation_type == CalculationType.PLUME_EXTENT:
        text = "injection point"
    elif calculation_type == CalculationType.POINT:
        text = "point          "
    elif calculation_type == CalculationType.LINE:
        text = "line           "
    for inj_well, distance in dist.items():
        logging.info(f"Injection well: {inj_well}")
        logging.info(
            f"    Smallest distance grid cell to {text} : {np.min(distance):>10.1f}"
        )
        logging.info(
            f"    Largest distance grid cell to {text}  : {np.max(distance):>10.1f}"
        )
        logging.info(
            f"    Average distance grid cell to {text}  : "
            f"{sum(distance) / len(distance):>10.1f}"
        )
    logging.info("")

    timer.stop("calculate_grid_cell_distances")
    return dist


def calculate_single_distances(
    grid_data: GridData,
    properties: Dict[str, Dict[str, np.ndarray]],
    dates: List[str],
    gasless: np.ndarray,
    threshold_gas: float,
    threshold_dissolved: float,
    config: Calculation,
    inj_wells: Optional[List[InjectionWellData]],
    plume_groups_gas: Optional[List[List[str]]],
    plume_groups_dissolved: Optional[List[List[str]]],
):
    calculation_type = config.type

    non_gasless = np.where(~gasless)[0]
    x_co2 = grid_data.x_active[non_gasless]
    y_co2 = grid_data.y_active[non_gasless]

    # Calculate distance from point/line to center of all non-gasless cells
    dist = _calculate_grid_cell_distances(
        inj_wells, calculation_type, x_co2, y_co2, config
    )

    dissolved_prop_key = next(
        (p for p in ("AMFS", "AMFG", "XMF2") if p in properties), None
    )

    gas_results = _find_distances_per_time_step(
        "SGAS",
        calculation_type,
        threshold_gas,
        properties,
        dates,
        dist,
        inj_wells,
        plume_groups_gas,
        non_gasless,
    )

    if dissolved_prop_key is not None:
        dissolved_results = _find_distances_per_time_step(
            dissolved_prop_key,
            calculation_type,
            threshold_dissolved,
            properties,
            dates,
            dist,
            inj_wells,
            plume_groups_dissolved,
            non_gasless,
        )
    else:
        dissolved_results = None
        warning_text = "WARNING: Neither AMFG nor XMF2 exists as properties."
        logging.warning(format_warning(warning_text))

    return gas_results, dissolved_results, dissolved_prop_key


def calculate_distances(
    case: str,
    distance_calculations: List[Calculation],
    injection_wells: Optional[List[InjectionWellData]] = None,
    do_plume_tracking: bool = False,
    threshold_gas: float = DEFAULT_THRESHOLD_GAS,
    threshold_dissolved: float = DEFAULT_THRESHOLD_DISSOLVED,
) -> List[Tuple[dict, Optional[dict], Optional[str]]]:
    """
    Find distance (plume extent / distance to point / distance to line) per
    date for SGAS and AMFG/XMF2.
    """
    logging.info("\nStart calculating distances")
    grid_data, properties, dates, gasless = load_plume_tracking_data(
        f"{case}.EGRID", f"{case}.UNRST"
    )

    if do_plume_tracking and injection_wells is not None:
        dissolved_prop_key = next(
            (p for p in ("AMFS", "AMFG", "XMF2") if p in properties), None
        )

        plume_groups_gas, _ = calculate_plume_groups(
            "SGAS",
            threshold_gas,
            grid_data,
            properties,
            dates,
            injection_wells,
            gasless,
        )

        if dissolved_prop_key is not None:
            plume_groups_dissolved, _ = calculate_plume_groups(
                dissolved_prop_key,
                threshold_dissolved,
                grid_data,
                properties,
                dates,
                injection_wells,
                gasless,
            )
        else:
            plume_groups_dissolved = None
    else:
        plume_groups_gas = None
        plume_groups_dissolved = None

    logging.info(f"Number of active grid cells: {grid_data.n_active}")

    all_results = []
    for i, single_config in enumerate(distance_calculations, 1):
        logging.info(f"\nCalculating distances for configuration number: {i}\n")
        a, b, c = calculate_single_distances(
            grid_data,
            properties,
            dates,
            gasless,
            threshold_gas,
            threshold_dissolved,
            single_config,
            injection_wells,
            plume_groups_gas,
            plume_groups_dissolved,
        )
        all_results.append((a, b, c))
        logging.info(f"Done calculating distances for configuration number: {i}\n")
    return all_results


def _find_distances_per_time_step(
    attribute_key: str,
    calculation_type: CalculationType,
    threshold: float,
    properties: Dict[str, Dict[str, np.ndarray]],
    dates: List[str],
    dist: Dict[str, np.ndarray],
    inj_wells: Optional[List[InjectionWellData]],
    plume_groups: Optional[List[List[str]]],
    non_gasless: np.ndarray,
) -> dict:
    """
    Find value of distance metric for each step
    """
    timer = Timer()
    timer.start("find_distances")

    do_plume_tracking = plume_groups is not None
    n_time_steps = len(dates)
    dist_per_group: Dict[str, Dict[str, np.ndarray]] = {}

    logging.info(f"\nStart calculating plume extent for {attribute_key}.\n")
    logging.info(f"Progress ({n_time_steps} time steps):")
    logging.info(f"{0:>6.1f} %")
    prop_data = properties[attribute_key]
    for i, date in enumerate(dates):
        data_co2 = prop_data[date][non_gasless]
        _find_distances_at_time_step(
            data_co2,
            i,
            threshold,
            do_plume_tracking,
            n_time_steps,
            calculation_type,
            dist,
            plume_groups[i] if plume_groups is not None else None,
            dist_per_group,
        )
        percent = (i + 1) / n_time_steps
        logging.info(f"{percent * 100:>6.1f} %")
    logging.info("")

    # Handle groups not found above, fill in zero:
    if do_plume_tracking:
        for well_name in dist.keys():
            if well_name != "ALL" and well_name not in dist_per_group:
                dist_per_group[well_name] = {well_name: np.zeros(shape=(n_time_steps,))}
    else:
        if "ALL" not in dist_per_group:
            dist_per_group["ALL"] = {
                well_name: np.zeros(shape=(n_time_steps,)) for well_name in dist.keys()
            }

    report_dates = [datetime.strptime(d, "%Y%m%d") for d in dates]
    outputs = _organize_output_with_dates(
        dist_per_group,
        calculation_type,
        do_plume_tracking,
        inj_wells,
        report_dates,
    )

    logging.info(f"Done calculating plume extent for {attribute_key}.")
    timer.stop("find_distances")
    return outputs


def _find_distances_at_time_step(
    data_co2: np.ndarray,
    i: int,
    threshold: float,
    do_plume_tracking: bool,
    n_time_steps: int,
    calculation_type: CalculationType,
    dist: Dict[str, np.ndarray],
    plume_groups: Optional[List[str]],
    # This argument will be updated:
    dist_per_group: Dict[str, Dict[str, np.ndarray]],
):
    if calculation_type == CalculationType.PLUME_EXTENT:
        if do_plume_tracking and plume_groups is not None:
            pg_dict = assemble_plume_groups_into_dict(plume_groups)
            for group_name, indices_this_group in pg_dict.items():
                # Skip calculating distances for cells that
                # have an undecided plume group
                if group_name == "undetermined":
                    continue
                # Check for new group name
                if group_name not in dist_per_group:
                    dist_per_group[group_name] = {
                        s: np.zeros(shape=(n_time_steps,))
                        for s in group_name.split("+")
                    }
                # Calculate max distance from each injection well in this group
                for well_name in group_name.split("+"):
                    dist_per_group[group_name][well_name][i] = dist[well_name][
                        indices_this_group
                    ].max()
        else:
            co2_above_threshold = np.where(data_co2 > threshold)[0]
            if i == 0:
                dist_per_group["ALL"] = {}
                for well_name in dist.keys():
                    dist_per_group["ALL"][well_name] = np.zeros(shape=(n_time_steps,))
            for well_name in dist.keys():
                if len(co2_above_threshold) > 0:
                    dist_per_group["ALL"][well_name][i] = dist[well_name][
                        co2_above_threshold
                    ].max()
                else:
                    dist_per_group["ALL"][well_name][i] = 0.0  # NBNB-AS: Or np.nan
    elif calculation_type in (
        CalculationType.POINT,
        CalculationType.LINE,
    ):
        if do_plume_tracking and plume_groups is not None:
            pg_dict = assemble_plume_groups_into_dict(plume_groups)
            for group_name, indices_this_group in pg_dict.items():
                # Skip calculating distances for cells that
                # have an undecided plume group
                if group_name == "undetermined":
                    continue
                # Check for new group name
                if group_name not in dist_per_group:
                    dist_per_group[group_name] = {"ALL": np.full(n_time_steps, np.nan)}
                # Calculate min distance in this group
                dist_per_group[group_name]["ALL"][i] = dist["ALL"][
                    indices_this_group
                ].min()
        else:
            co2_above_threshold = np.where(data_co2 > threshold)[0]
            if i == 0:
                dist_per_group["ALL"] = {}
                for well_name in dist.keys():
                    dist_per_group["ALL"][well_name] = np.full(n_time_steps, np.nan)
            if len(co2_above_threshold) > 0:
                dist_per_group["ALL"]["ALL"][i] = dist["ALL"][co2_above_threshold].min()
            else:
                dist_per_group["ALL"]["ALL"][i] = np.nan


def _organize_output_with_dates(
    dist_per_group: Dict[str, Dict[str, np.ndarray]],
    calculation_type: CalculationType,
    do_plume_tracking: bool,
    inj_wells: Optional[List[InjectionWellData]],
    report_dates: List[datetime],
) -> dict:
    outputs: dict = {}
    for group_name, single_group_distances in dist_per_group.items():
        outputs[group_name] = {}
        for single_group, distances in single_group_distances.items():
            well_name = "ALL"
            if calculation_type == CalculationType.PLUME_EXTENT:
                if do_plume_tracking and inj_wells is not None:
                    # NBNB-AS: x.name here should probably be handled earlier
                    well_name = [
                        x.name
                        for x in inj_wells
                        if x.number == single_group or x.name == single_group
                    ][0]
                else:
                    if inj_wells is not None and len(inj_wells) != 0:
                        well_name = [
                            x.name for x in inj_wells if x.name == single_group
                        ][0]
                    else:
                        well_name = "WELL"
            outputs[group_name][well_name] = []
            for i, d in enumerate(report_dates):
                date_and_result = [d.strftime("%Y-%m-%d"), distances[i]]
                outputs[group_name][well_name].append(date_and_result)
    return outputs
