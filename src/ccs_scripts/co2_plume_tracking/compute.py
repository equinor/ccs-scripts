import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ccs_scripts.co2_plume_tracking.utils import (
    DEFAULT_THRESHOLD_DISSOLVED,
    DEFAULT_THRESHOLD_GAS,
    GridData,
    InjectionWellData,
    PlumeGroups,
    Status,
    sort_well_names,
)
from ccs_scripts.utils.gridproperty_tools import GridHandler
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import (
    format_error,
    format_warning,
    identify_gas_less_cells,
    reduce_properties,
)
from ccs_scripts.utils.xtgeo_logging import setup_xtgeo_logging

setup_xtgeo_logging()

INJ_POINT_THRESHOLD_LATERAL = 80.0
INJ_POINT_THRESHOLD_VERTICAL = 10.0


def _find_cell_xy(
    grid_data: GridData, x: float, y: float, k: int
) -> Optional[Tuple[int, int]]:
    """Find (i, j) of cell at (x, y) in layer k (nearest-neighbor)."""
    layer_active = grid_data.active_index_3d[:, :, k]
    mask = layer_active >= 0
    if not mask.any():
        return None
    active_indices = layer_active[mask]
    cx = grid_data.x_active[active_indices]
    cy = grid_data.y_active[active_indices]
    dist_sq = (cx - x) ** 2 + (cy - y) ** 2
    idx = np.argmin(dist_sq)
    ij_positions = np.argwhere(mask)
    return (int(ij_positions[idx, 0]), int(ij_positions[idx, 1]))


def _fetch_properties_xtgeo(
    grid_handler: GridHandler,
    props_to_extract: List[str],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    """Fetch properties from UNRST file using xtgeo.

    Returns properties for active cells only, in C-order.
    """
    names = [p for p in props_to_extract if p in grid_handler.property_names]
    gprops = grid_handler.read_properties(names=names, dates="all")
    actnum = grid_handler.grid.actnum_array.astype(bool)

    props: Dict[str, Dict[str, np.ndarray]] = {}
    dates_ordered: List[str] = []

    for prop in gprops.props:
        parts = prop.name.split("--")
        pname = parts[0]
        pdate = str(prop.date or parts[1]) if len(parts) > 1 else str(prop.date)

        if pname not in props:
            props[pname] = {}
        props[pname][pdate] = prop.values[actnum].data

        if pdate not in dates_ordered:
            dates_ordered.append(pdate)

    logging.info(
        "Done reading properties from file"
        "\nRelevant properties extracted:"
        f"\n    {', '.join(list(props.keys()))}\n"
    )
    return props, dates_ordered


def _find_gasless_cells(
    properties: Dict[str, Dict[str, np.ndarray]],
) -> np.ndarray:
    """Identify gasless cells from property arrays for active cells."""
    dissolved_prop = None
    if "AMFS" in properties:
        dissolved_prop = properties["AMFS"]
    elif "AMFG" in properties:
        dissolved_prop = properties["AMFG"]
    elif "XMF2" in properties:
        dissolved_prop = properties["XMF2"]

    return identify_gas_less_cells(properties["SGAS"], dissolved_prop)


def load_plume_tracking_data(
    grid_file: str, unrst_file: str
) -> Tuple[GridData, Dict[str, Dict[str, np.ndarray]], List[str], np.ndarray]:
    """Load grid and properties needed for plume tracking.

    Returns:
        grid_data: Pre-computed grid lookup arrays
        properties: {prop_name: {date_str: array}} for active cells
        dates: Ordered date strings
        gasless: Boolean mask over active cells (True = gasless)
    """
    grid_handler = GridHandler(Path(grid_file), Path(unrst_file))
    grid_data = GridData.from_xtgeo_grid(grid_handler.grid)

    dissolved_prop = next(
        (p for p in ("AMFS", "AMFG", "XMF2") if p in grid_handler.property_names), None
    )

    props_to_extract = ["SGAS"]
    if dissolved_prop is not None:
        props_to_extract.append(dissolved_prop)

    properties, dates = _fetch_properties_xtgeo(grid_handler, props_to_extract)
    gasless = _find_gasless_cells(properties)
    return grid_data, properties, dates, gasless


def calculate_all_plume_groups(
    grid_data: GridData,
    properties: Dict[str, Dict[str, np.ndarray]],
    dates: List[str],
    gasless: np.ndarray,
    threshold_gas: float,
    threshold_dissolved: float,
    inj_wells: List[InjectionWellData],
) -> Tuple[List[List[str]], Optional[List[List[str]]], Optional[str]]:
    dissolved_prop_key = None
    if "AMFS" in properties:
        dissolved_prop_key = "AMFS"
    elif "AMFG" in properties:
        dissolved_prop_key = "AMFG"
    elif "XMF2" in properties:
        dissolved_prop_key = "XMF2"

    pg_prop_gas, _ = calculate_plume_groups(
        "SGAS",
        threshold_gas,
        grid_data,
        properties,
        dates,
        inj_wells,
        gasless,
    )

    if dissolved_prop_key is not None:
        pg_prop_dissolved, _ = calculate_plume_groups(
            dissolved_prop_key,
            threshold_dissolved,
            grid_data,
            properties,
            dates,
            inj_wells,
            gasless,
        )
    else:
        pg_prop_dissolved = None
        warning_text = "WARNING: Cannot find any of the following properties: "
        warning_text += "AMFS, AMFG or XMF2."
        logging.warning(format_warning(warning_text))

    return pg_prop_gas, pg_prop_dissolved, dissolved_prop_key


def load_data_and_calculate_plume_groups(
    case: str,
    injection_wells: List[InjectionWellData],
    threshold_gas: float = DEFAULT_THRESHOLD_GAS,
    threshold_dissolved: float = DEFAULT_THRESHOLD_DISSOLVED,
) -> Tuple[List[List[str]], Optional[List[List[str]]], Optional[str], List[datetime]]:
    logging.info("\nStart calculations for plume tracking")
    grid_data, properties, dates, gasless = load_plume_tracking_data(
        f"{case}.EGRID", f"{case}.UNRST"
    )
    logging.info(f"Number of active grid cells: {grid_data.n_active}")

    pg_prop_gas, pg_prop_dissolved, dissolved_prop_key = calculate_all_plume_groups(
        grid_data,
        properties,
        dates,
        gasless,
        threshold_gas,
        threshold_dissolved,
        injection_wells,
    )

    report_dates = [datetime.strptime(d, "%Y%m%d") for d in dates]
    return pg_prop_gas, pg_prop_dissolved, dissolved_prop_key, report_dates


def _log_number_of_grid_cells(
    n_grid_cells_for_logging: Dict[str, List[int]],
    report_dates: List[datetime],
    attribute_key: str,
    inj_wells: List[InjectionWellData],
):
    logging.info(
        f"Number of grid cells with {attribute_key} above threshold "
        f"for the different plumes:"
    )

    for well in inj_wells:
        if well.name not in n_grid_cells_for_logging.keys():
            n_grid_cells_for_logging[well.name] = [0] * len(report_dates)

    n_cells_sorted = sort_well_names(n_grid_cells_for_logging, inj_wells)
    sorted_cols = n_cells_sorted.keys()
    header = f"{'Date':<11}"
    widths = {}
    for col in sorted_cols:
        widths[col] = max(9, len(col))
        header += f" {col:>{widths[col]}}"
    logging.info("\n" + header)
    logging.info("-" * len(header))
    for i, d in enumerate(report_dates):
        date = d.strftime("%Y-%m-%d")
        row = f"{date:<11}"
        for col in sorted_cols:
            n_cells = str(n_cells_sorted[col][i]) if n_cells_sorted[col][i] > 0 else "-"
            row += f" {n_cells:>{widths[col]}}"
        logging.info(row)
    logging.info("")
    if "undetermined" in n_cells_sorted:
        no_groups = len(n_cells_sorted) == 1
        warning_text = (
            f"WARNING: Plume group not found for "
            f"{'any' if no_groups else 'some'} grid cells with CO2."
        )
        logging.warning(format_warning(warning_text))
        logging.warning("         See table above, under column '?'.")
        if no_groups:
            logging.warning(
                "         The reason might be incorrect coordinates "
                "for the injection wells.\n"
            )
        else:
            logging.warning("")  # Line ending


def _find_inj_wells_grid_indices(
    inj_wells_grid_indices: Dict[str, List[Tuple[int, int, Optional[int]]]],
    grid_data: GridData,
    inj_wells: List[InjectionWellData],
    print_table: bool = False,
):
    wells_with_errors = []
    for well in inj_wells:
        if well.z is not None:
            found = False
            for z in well.z:
                ijk = grid_data.find_cell(x=well.x, y=well.y, z=z)
                if ijk is not None:
                    inj_wells_grid_indices[well.name] = [ijk]
                    found = True
                    break
            if not found:
                wells_with_errors.append(well)
        else:
            inj_wells_grid_indices[well.name] = []
            for k in range(grid_data.nz):
                ij = _find_cell_xy(grid_data, x=well.x, y=well.y, k=k)
                if ij is None:
                    # Inactive layer, skip
                    continue
                active_index = int(grid_data.active_index_3d[ij[0], ij[1], k])
                if active_index != -1:
                    if ij + (None,) not in inj_wells_grid_indices[well.name]:
                        inj_wells_grid_indices[well.name].append((ij[0], ij[1], None))

    if print_table:
        logging.info("Found the following grid cell indices for injection wells:")
        logging.info(
            f"\n{'Name':<25} {'x':>12} {'y':>12} {'z':>9} "
            f"{'i':>6} {'j':>6} {'k':>6}"
        )
        logging.info("-" * 82)
        for well in inj_wells:
            x_str = f"{well.x:.2f}"
            y_str = f"{well.y:.2f}"
            z_str = f"{well.z[0]:.2f}" if well.z is not None else "-"
            if well not in wells_with_errors:
                indices = inj_wells_grid_indices[well.name]
                for idx, entry in enumerate(indices):
                    if entry is None or entry[0] is None:
                        i_str, j_str, k_str = "X", "X", "X"
                    else:
                        i_str = str(entry[0])
                        j_str = str(entry[1])
                        k_str = str(entry[2]) if entry[2] is not None else "-"
                    if idx == 0:
                        logging.info(
                            f"{well.name:<25} {x_str:>12} {y_str:>12} "
                            f"{z_str:>9} {i_str:>6} {j_str:>6} {k_str:>6}"
                        )
                    else:
                        logging.info(
                            f"{'':<25} {'':>12} {'':>12} "
                            f"{'':>9} {i_str:>6} {j_str:>6} {k_str:>6}"
                        )
            else:
                logging.info(
                    f"{well.name:<25} {x_str:>12} {y_str:>12} "
                    f"{z_str:>9} {'X':>6} {'X':>6} {'X':>6}"
                )

    if wells_with_errors:
        error_text = "\nERROR: Could not find grid cell indices for "
        error_text += "the following injection well(s):"
        for well in wells_with_errors:
            z_str = "z: " + str(well.z) if well.z is not None else "z: not provided"
            error_text += f"\n         - {well.name}  (x: {well.x}, y: {well.y}, "
            error_text += z_str
        error_text += "\n       Please check the coordinates and "
        error_text += "make sure they are within the grid."
        logging.error(format_error(error_text))
        sys.exit(1)


def calculate_plume_groups(
    attribute_key: str,
    threshold: float,
    grid_data: GridData,
    properties: Dict[str, Dict[str, np.ndarray]],
    dates: List[str],
    inj_wells: List[InjectionWellData],
    gasless: np.ndarray,
) -> Tuple[List[List[str]], Dict[int, int]]:
    """
    Calculates/tracks the plume groups for a single property.
    The result is a list over the number of time steps, where
    each element is a list over the number of active grid cells.
    The string is the name of the plume group, for instance
    "well_A+well_B" (if well_A and well_B have merged).

    Args:
        attribute_key: Property name to track (e.g. "SGAS", "AMFG", "XMF2")
        threshold: Threshold for attribute_key
        grid_data: Pre-computed grid lookup arrays (from GridData.from_xtgeo_grid)
        properties: {prop_name: {date_str: array}} for active cells
        dates: Ordered list of date strings ("YYYYMMDD")
        inj_wells: Injection well data
        gasless: Boolean mask over active cells (True = gasless)
    """
    timer = Timer()
    timer.start("plume_tracking")

    time_start = time.time()
    n_time_steps = len(dates)
    n_grid_cells_for_logging: Dict[str, List[int]] = {}

    non_gasless = np.where(~gasless)[0]
    n_cells = len(non_gasless)

    properties = reduce_properties(properties, ~gasless)
    data = properties[attribute_key]

    cell_map_gasless_to_active = {i: non_gasless[i] for i in range(0, n_cells)}
    cell_map_active_to_gasless = {v: k for k, v in cell_map_gasless_to_active.items()}

    inj_wells_grid_indices: Dict[str, List[Tuple[int, int, Optional[int]]]] = {}
    _find_inj_wells_grid_indices(
        inj_wells_grid_indices, grid_data, inj_wells, print_table=True
    )

    logging.info(f"\nStart calculating plume tracking for {attribute_key}.\n")
    logging.info(f"Progress ({n_time_steps} time steps):")
    logging.info(f"{0:>6.1f} %")

    # Plume group property
    timer.start("plume_tracking_represent_as_property", "plume_tracking")
    pg_prop = [["" for _ in range(n_cells)] for _ in range(n_time_steps)]
    timer.stop("plume_tracking_represent_as_property")
    prev_groups = PlumeGroups(n_cells)
    for i, date in enumerate(dates):
        groups = PlumeGroups(n_cells)
        _plume_groups_at_time_step(
            data[date],  # type: ignore[arg-type]
            grid_data,
            i,
            threshold,
            prev_groups,
            inj_wells,
            inj_wells_grid_indices,
            n_time_steps,
            cell_map_gasless_to_active,
            cell_map_active_to_gasless,
            groups,
            n_grid_cells_for_logging,
        )

        timer.start("plume_tracking_represent_as_property", "plume_tracking")
        for j, all_groups in enumerate(groups.all_groups):
            if all_groups:
                group_string = "+".join(
                    [
                        str(
                            [x.name for x in inj_wells if x.number == y][0]
                            if y != -1
                            else "undetermined"
                        )
                        for y in all_groups
                    ]
                )
                pg_prop[i][j] = group_string
        timer.stop("plume_tracking_represent_as_property")

        prev_groups = groups.copy()
        percent = (i + 1) / n_time_steps
        logging.info(f"{percent * 100:>6.1f} %")
    logging.info("")

    report_dates = [datetime.strptime(d, "%Y%m%d") for d in dates]
    timer.start("plume_tracking_logging", "plume_tracking")
    _log_number_of_grid_cells(
        n_grid_cells_for_logging, report_dates, attribute_key, inj_wells
    )
    timer.stop("plume_tracking_logging")
    logging.info(f"Done calculating plume tracking for {attribute_key}.")
    logging.info(
        f"Execution time {attribute_key}: {(time.time() - time_start):.1f} s\n"
    )

    timer.stop("plume_tracking")
    return pg_prop, cell_map_active_to_gasless


def _plume_groups_at_time_step(
    data: np.ndarray,
    grid_data: GridData,
    i: int,
    threshold: float,
    prev_groups: PlumeGroups,
    inj_wells: List[InjectionWellData],
    inj_wells_grid_indices: Dict[str, List[Tuple[int, int, Optional[int]]]],
    n_time_steps: int,
    cell_map_gasless_to_active: Dict[int, int],
    cell_map_active_to_gasless: Dict[int, int],
    # These arguments will be updated:
    groups: PlumeGroups,
    n_grid_cells_for_logging: Dict[str, List[int]],
):
    timer = Timer()

    cells_with_co2 = np.where(data > threshold)[0]

    logging.debug("\nPrevious group:")
    prev_groups.debug_print()

    timer.start("plume_tracking_init_groups", "plume_tracking")
    _initialize_groups_from_prev_step_and_inj_wells(
        cells_with_co2,
        prev_groups,
        grid_data,
        inj_wells,
        inj_wells_grid_indices,
        groups,
        cell_map_gasless_to_active,
    )
    timer.stop("plume_tracking_init_groups")

    logging.debug("\nCurrent group after first intialization:")
    groups.debug_print()

    timer.start("plume_tracking_resolve_undetermined", "plume_tracking")
    groups_to_merge = groups.resolve_undetermined_cells(
        grid_data, cell_map_gasless_to_active, cell_map_active_to_gasless
    )
    timer.stop("plume_tracking_resolve_undetermined")
    for full_group in groups_to_merge:
        new_group = [x for y in full_group for x in y]
        new_group.sort()
        for j in range(len(groups.status)):
            if groups.status[j] == Status.HAS_CO2:
                for g in full_group:
                    if set(groups.all_groups[j]) & set(g):
                        groups.all_groups[j] = new_group.copy()

    logging.debug("\nCurrent group after resolving undetermined cells:")
    groups.debug_print()

    timer.start("plume_tracking_find_unique_groups", "plume_tracking")
    unique_groups = groups.find_unique_groups()
    timer.stop("plume_tracking_find_unique_groups")
    for g in unique_groups:
        if g == [-1]:
            if "undetermined" not in n_grid_cells_for_logging:
                n_grid_cells_for_logging["undetermined"] = [0] * n_time_steps
            n_grid_cells_for_logging["undetermined"][i] = len(
                [j for j in cells_with_co2 if groups.all_groups[j] == [-1]]
            )
        else:
            indices_this_group = [
                j for j in cells_with_co2 if groups.all_groups[j] == g
            ]

            group_string = "+".join(
                [str([x.name for x in inj_wells if x.number == y][0]) for y in g]
            )
            if group_string not in n_grid_cells_for_logging:
                n_grid_cells_for_logging[group_string] = [0] * n_time_steps
            n_grid_cells_for_logging[group_string][i] = len(indices_this_group)


def _initialize_groups_from_prev_step_and_inj_wells(
    cells_with_co2: np.ndarray,
    prev_groups: PlumeGroups,
    grid_data: GridData,
    inj_wells: List[InjectionWellData],
    inj_wells_grid_indices: Dict[str, List[Tuple[int, int, Optional[int]]]],
    groups: PlumeGroups,
    cell_map_gasless_to_active: Dict[int, int],
):
    new_z_coords: Dict[str, List[float]] = {}
    for index in cells_with_co2:
        if prev_groups.status[index] == Status.HAS_CO2:
            groups.status[index] = prev_groups.status[index]
            groups.all_groups[index] = prev_groups.all_groups[index]
        else:
            # This grid cell did not have CO2 in the last time step
            active_ind = cell_map_gasless_to_active[index]
            i, j, k = tuple(grid_data.ijk_from_active[active_ind])
            x = float(grid_data.x_active[active_ind])
            y = float(grid_data.y_active[active_ind])
            z = float(grid_data.z_active[active_ind])

            found = False
            for well in inj_wells:
                if well.z is not None:
                    same_cell = any(
                        [
                            (i, j, k) == (wi, wj, wk)
                            for (wi, wj, wk) in inj_wells_grid_indices[well.name]
                        ]
                    )
                    xyz_close = (
                        abs(x - well.x) <= INJ_POINT_THRESHOLD_LATERAL
                        and abs(y - well.y) <= INJ_POINT_THRESHOLD_LATERAL
                        and any(
                            [
                                abs(z - well_z) <= INJ_POINT_THRESHOLD_VERTICAL
                                for well_z in well.z
                            ]
                        )
                    )
                else:
                    same_cell = False
                    for cell_i, cell_j, _ in inj_wells_grid_indices[well.name]:
                        if (i, j) == (cell_i, cell_j):
                            same_cell = True
                            break
                    xyz_close = (
                        abs(x - well.x) <= INJ_POINT_THRESHOLD_LATERAL
                        and abs(y - well.y) <= INJ_POINT_THRESHOLD_LATERAL
                    )
                if same_cell or xyz_close:
                    found = True
                    merged_group = groups.check_if_well_is_part_of_larger_group(
                        well.number
                    )
                    if merged_group is None:
                        groups.set_cell_groups(index, [well.number])
                    else:
                        groups.set_cell_groups(index, merged_group)
                    if (
                        well.name not in new_z_coords
                        or z not in new_z_coords[well.name]
                    ):
                        if well.name not in new_z_coords:
                            new_z_coords[well.name] = [z]
                        else:
                            new_z_coords[well.name].append(z)
                    break
            if not found:
                groups.status[index] = Status.UNDETERMINED
    _update_inj_z_coordinates(inj_wells, new_z_coords)
    _find_inj_wells_grid_indices(
        inj_wells_grid_indices, grid_data, inj_wells
    )  # Might need an update


def _update_inj_z_coordinates(
    inj_wells: List[InjectionWellData],
    new_z_coords: Dict[str, List[float]],
):
    for well in inj_wells:
        if well.name in new_z_coords:
            for z in new_z_coords[well.name]:
                if well.z is None or z not in well.z and len(well.z) < 5:
                    logging.debug(
                        f"Found new injection z-coordinate for well {well.name}: {z}"
                    )
                    if well.z is None:
                        well.z = [z]
                    else:
                        well.z.append(z)
