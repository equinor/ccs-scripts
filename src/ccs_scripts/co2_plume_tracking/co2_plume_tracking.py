#!/usr/bin/env python
"""
Calculations for tracking the CO2 plumes from different injection wells,
using SGAS and the dissolved property (AMFG/XMF2).
Keeps track of which grid cells belong to which
plume group at each time step, and merges plumes if they meet.
"""
import argparse
import getpass
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from resdata.grid import Grid
from resdata.resfile import ResdataFile

from ccs_scripts.co2_plume_tracking.utils import (
    InjectionWellData,
    PlumeGroups,
    Status,
    assemble_plume_groups_into_dict,
    sort_well_names,
)
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import (
    fetch_properties,
    find_active_and_gasless_cells,
    read_yaml_file,
    reduce_properties,
)

DEFAULT_THRESHOLD_GAS = 0.2
DEFAULT_THRESHOLD_DISSOLVED = 0.0005
INJ_POINT_THRESHOLD_LATERAL = 80.0
INJ_POINT_THRESHOLD_VERTICAL = 10.0

DESCRIPTION = """
Calculations for tracking the CO2 plumes from different injection wells,
using SGAS and the dissolved property (AMFG/XMF2). Keeps track of which
grid cells belong to which plume group at each time step, and merges
plumes if they meet.

Output is a table on CSV format, counting the number of grid cells in
each group at each time step. The functionality is also used by the plume
extent script, to separate the results into different plume groups.
"""

CATEGORY = "modelling.reservoir"


class Configuration:
    """
    Holds the configuration for plume tracking calculations
    """

    def __init__(
        self,
        config_file: str,
    ):
        self.injection_wells: List[InjectionWellData] = []

        input_dict = read_yaml_file(config_file)
        self.make_config_from_input_dict(input_dict)

    def make_config_from_input_dict(self, input_dict: Dict):
        if "injection_wells" not in input_dict:
            logging.error("\nERROR: No injection wells specified.")
        else:
            if not isinstance(input_dict["injection_wells"], list):
                logging.error(
                    '\nERROR: Specification under "injection_wells" in '
                    "input YAML file is not a list."
                )
                sys.exit(1)
            for i, injection_well_info in enumerate(input_dict["injection_wells"], 1):
                args_required = ["name", "x", "y"]
                for arg in args_required:
                    if arg not in injection_well_info:
                        logging.error(
                            f'\nERROR: Missing "{arg}" under "injection_wells" '
                            f"for injection well number {i}."
                        )
                        sys.exit(1)

                self.injection_wells.append(
                    InjectionWellData(
                        name=injection_well_info["name"],
                        x=injection_well_info["x"],
                        y=injection_well_info["y"],
                        z=(
                            [injection_well_info["z"]]
                            if "z" in injection_well_info
                            else None
                        ),
                        number=len(self.injection_wells) + 1,
                    )
                )


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculations for tracking plume groups"
    )
    parser.add_argument("case", help="Name of Eclipse case")
    parser.add_argument(
        "--config_file",
        help="YML file with configurations for plume tracking calculations.",
        default="",
    )
    parser.add_argument(
        "--output_csv",
        help="Path to output CSV file",
        default=None,
    )
    parser.add_argument(
        "--threshold_gas",
        default=DEFAULT_THRESHOLD_GAS,
        type=float,
        help="Threshold for gas saturation (SGAS)",
    )
    parser.add_argument(
        "--threshold_dissolved",
        default=DEFAULT_THRESHOLD_DISSOLVED,
        type=float,
        help="Threshold for aqueous mole fraction of gas (AMFG or XMF2)",
    )
    parser.add_argument(
        "--no_logging",
        help="Skip print of detailed information during execution of script",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        help="Enable print of debugging data during execution of script. "
        "Normally not necessary for most users.",
        action="store_true",
    )

    return parser


def _setup_log_configuration(arguments: argparse.Namespace) -> None:
    if arguments.debug:
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    elif arguments.no_logging:
        logging.basicConfig(format="%(message)s", level=logging.WARNING)
    else:
        logging.basicConfig(format="%(message)s", level=logging.INFO)


def _log_input_configuration(arguments: argparse.Namespace) -> None:
    version = "v0.9.2"
    is_dev_version = True
    if is_dev_version:
        version += "_dev"
        try:
            source_dir = os.path.dirname(os.path.abspath(__file__))
            short_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], cwd=source_dir
                )
                .decode("ascii")
                .strip()
            )
        except subprocess.CalledProcessError:
            short_hash = "-"
        version += " (latest git commit: " + short_hash + ")"

    now = datetime.now()
    date_time = now.strftime("%B %d, %Y %H:%M:%S")
    logging.info("CCS-scripts - Plume tracking calculations")
    logging.info("=========================================")
    logging.info(f"Version             : {version}")
    logging.info(f"Date and time       : {date_time}")
    logging.info(f"User                : {getpass.getuser()}")
    logging.info(f"Host                : {socket.gethostname()}")
    logging.info(f"Platform            : {platform.system()} ({platform.release()})")
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    logging.info(f"Python version      : {py_version}")

    logging.info(f"\nCase                    : {arguments.case}")
    if not os.path.isabs(arguments.case):
        logging.info(f"  => Absolute path      : {os.path.abspath(arguments.case)}")
    logging.info(
        f"Configuration YAML-file : "
        f"{arguments.config_file if arguments.config_file != '' else 'Not specified'}"
    )
    if arguments.output_csv is None or arguments.output_csv == "":
        text = "Not specified, using default"
    else:
        text = arguments.output_csv
    logging.info(f"Output CSV file         : {text}")
    logging.info(f"Threshold gas           : {arguments.threshold_gas}")
    logging.info(f"Threshold dissolved     : {arguments.threshold_dissolved}\n")


def _log_configuration(config: Configuration) -> None:
    logging.info("\nInjection well data:")
    logging.info(f"\n{'Number':<8} {'Name':<15} {'x':<15} {'y':<15} {'z':<15}")
    logging.info("-" * 72)
    for i, well in enumerate(config.injection_wells, 1):
        z_str = f"{well.z[0]:<15}" if well.z is not None else "-"
        logging.info(f"{i:<8} {well.name:<15} {well.x:<15} {well.y:<15} {z_str}")
    logging.info("")


def calculate_all_plume_groups(
    grid: Grid,
    unrst: ResdataFile,
    threshold_gas: float,
    threshold_dissolved: float,
    inj_wells: List[InjectionWellData],
) -> Tuple[List[List[str]], Optional[List[List[str]]], Optional[str]]:
    pg_prop_gas, _ = calculate_plume_groups(
        "SGAS",
        threshold_gas,
        unrst,
        grid,
        inj_wells,
    )
    if "AMFG" in unrst:
        pg_prop_dissolved, _ = calculate_plume_groups(
            "AMFG",
            threshold_dissolved,
            unrst,
            grid,
            inj_wells,
        )
        dissolved_prop_key = "AMFG"
    elif "XMF2" in unrst:
        pg_prop_dissolved, _ = calculate_plume_groups(
            "XMF2",
            threshold_dissolved,
            unrst,
            grid,
            inj_wells,
        )
        dissolved_prop_key = "XMF2"
    else:
        pg_prop_dissolved = None
        dissolved_prop_key = None
        logging.warning("WARNING: Neither AMFG nor XMF2 exists as properties.")

    return pg_prop_gas, pg_prop_dissolved, dissolved_prop_key


def load_data_and_calculate_plume_groups(
    case: str,
    injection_wells: List[InjectionWellData],
    threshold_gas: float = DEFAULT_THRESHOLD_GAS,
    threshold_dissolved: float = DEFAULT_THRESHOLD_DISSOLVED,
) -> Tuple[List[List[str]], Optional[List[List[str]]], Optional[str], List[datetime]]:
    logging.info("\nStart calculations for plume tracking")
    grid = Grid(f"{case}.EGRID")
    unrst = ResdataFile(f"{case}.UNRST")

    logging.info(f"Number of active grid cells: {grid.get_num_active()}")

    (pg_prop_gas, pg_prop_dissolved, dissolved_prop_key) = calculate_all_plume_groups(
        grid,
        unrst,
        threshold_gas,
        threshold_dissolved,
        injection_wells,
    )

    return pg_prop_gas, pg_prop_dissolved, dissolved_prop_key, unrst.report_dates


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
        logging.warning(
            f"WARNING: Plume group not found for "
            f"{'any' if no_groups else 'some'} grid cells with CO2."
        )
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
    grid: Grid,
    inj_wells: List[InjectionWellData],
):
    for well in inj_wells:
        if well.z is not None:
            inj_wells_grid_indices[well.name] = [
                grid.find_cell(x=well.x, y=well.y, z=well.z[0])
            ]
        else:
            inj_wells_grid_indices[well.name] = []
            for k in range(grid.get_nz()):
                xy = grid.find_cell_xy(x=well.x, y=well.y, k=k)
                if xy + (None,) not in inj_wells_grid_indices[well.name]:
                    inj_wells_grid_indices[well.name].append((xy[0], xy[1], None))


def calculate_plume_groups(
    attribute_key: str,
    threshold: float,
    unrst: ResdataFile,
    grid: Grid,
    inj_wells: List[InjectionWellData],
) -> Tuple[List[List[str]], Dict[int, int]]:
    """
    Calculates/tracks the plume groups for a single property.
    The result is a list over the number of time steps, where
    each element is a list over the number of active grid cells.
    The string is the name of the plume group, for instance
    "well_A+well_B" (if well_A and well_B have merged).
    """
    timer = Timer()
    timer.start("plume_tracking")

    time_start = time.time()
    n_time_steps = len(unrst.report_steps)
    n_grid_cells_for_logging: Dict[str, List[int]] = {}

    if "AMFG" in unrst:
        dissolved_prop = "AMFG"
    elif "XMF2" in unrst:
        dissolved_prop = "XMF2"
    props_to_extract = ["SGAS", dissolved_prop]
    if attribute_key not in props_to_extract:
        props_to_extract.append(attribute_key)
    properties, dates = fetch_properties(unrst, props_to_extract)

    active, gasless = find_active_and_gasless_cells(grid, properties, False)
    global_active_idx = active[~gasless]
    non_gasless = np.where(np.isin(active, global_active_idx))[0]
    n_cells = len(non_gasless)

    properties = reduce_properties(properties, ~gasless)
    data = properties[attribute_key]

    cell_map_gasless_to_active = {i: non_gasless[i] for i in range(0, n_cells)}
    cell_map_active_to_gasless = {v: k for k, v in cell_map_gasless_to_active.items()}

    inj_wells_grid_indices: Dict[str, List[Tuple[int, int, Optional[int]]]] = {}
    _find_inj_wells_grid_indices(inj_wells_grid_indices, grid, inj_wells)

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
            grid,
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

    timer.start("plume_tracking_logging", "plume_tracking")
    _log_number_of_grid_cells(
        n_grid_cells_for_logging, unrst.report_dates, attribute_key, inj_wells
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
    grid: Grid,
    i: int,
    threshold: float,
    prev_groups: PlumeGroups,
    inj_wells: List[InjectionWellData],
    inj_wells_grid_indices: Dict[str, List[Tuple[int, int, Optional[int]]]],
    n_time_steps: int,
    cell_map_gasless_to_active: dict[int, int],
    cell_map_active_to_gasless: dict[int, int],
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
        grid,
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
        grid, cell_map_gasless_to_active, cell_map_active_to_gasless
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
    grid: Grid,
    inj_wells: List[InjectionWellData],
    inj_wells_grid_indices: Dict[str, List[Tuple[int, int, Optional[int]]]],
    groups: PlumeGroups,
    cell_map_gasless_to_active: dict[int, int],
):
    new_z_coords: Dict[str, List[float]] = {}
    for index in cells_with_co2:
        if prev_groups.status[index] == Status.HAS_CO2:
            groups.status[index] = prev_groups.status[index]
            groups.all_groups[index] = prev_groups.all_groups[index]
        else:
            # This grid cell did not have CO2 in the last time step
            active_ind = cell_map_gasless_to_active[index]
            (i, j, k) = grid.get_ijk(active_index=active_ind)
            (x, y, z) = grid.get_xyz(active_index=active_ind)
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
        inj_wells_grid_indices, grid, inj_wells
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


def _log_results(
    df: pd.DataFrame,
) -> None:
    dfs = df.sort_values("date")
    col_width = 1 + max(31, max([len(c) for c in df]))
    logging.info("\nSummary of results:")
    logging.info("===================")
    logging.info(
        f"Number of dates {' ' * (col_width - 5)}: {len(dfs['date'].unique()):>11}"
    )
    logging.info(f"First date      {' ' * (col_width - 5)}: {dfs['date'].iloc[0]:>11}")
    logging.info(f"Last date       {' ' * (col_width - 5)}: {dfs['date'].iloc[-1]:>11}")

    for col in df.drop("date", axis=1).columns:
        logging.info(f"End state {col:<{col_width}} : {dfs[col].iloc[-1]:>11.1f}")


def _find_output_file(output: str, case: str):
    if output is None:
        p = Path(case).parents[2]
        p2 = p / "share" / "results" / "tables" / "plume_tracking.csv"
        return str(p2)
    else:
        return output


def _collect_results_into_dataframe(
    report_dates: List[datetime],
    pg_prop_gas: List[List[str]],
    pg_prop_dissolved: Optional[List[List[str]]],
    dissolved_prop_key: Optional[str],
    injection_wells: List[InjectionWellData],
) -> pd.DataFrame:
    dates = [[d.strftime("%Y-%m-%d")] for d in report_dates]
    df = pd.DataFrame.from_records(dates, columns=["date"])

    for prop_key, pg_prop in zip(
        ["SGAS", dissolved_prop_key], [pg_prop_gas, pg_prop_dissolved]
    ):
        if pg_prop is None or prop_key is None:
            continue
        results = {}
        for i, p in enumerate(pg_prop):
            pg_dict = assemble_plume_groups_into_dict(p)
            for group_name, indices in pg_dict.items():
                if group_name not in results:
                    results[group_name] = np.zeros(
                        shape=(len(dates)),
                        dtype=int,
                    )
                results[group_name][i] = len(indices)
        results_sorted = sort_well_names(results, injection_wells)
        results_sorted = {
            prop_key + "_" + key: value for key, value in results_sorted.items()
        }

        prop_df = pd.DataFrame(results_sorted)
        df = pd.concat([df, prop_df], axis=1)

    return df


def main():
    """
    Calculations for tracking plume groups.
    The method calculate_plume_groups() can be used by other scripts
    that want this functionality.
    Output from this script is a simple CSV-file counting the number of
    grid cells in each plume group for each time step.
    """
    time_start = time.time()
    args = _make_parser().parse_args()
    _setup_log_configuration(args)
    _log_input_configuration(args)

    config = Configuration(
        args.config_file,
    )
    _log_configuration(config)

    (pg_prop_gas, pg_prop_dissolved, dissolved_prop_key, dates) = (
        load_data_and_calculate_plume_groups(
            args.case,
            config.injection_wells,
            args.threshold_gas,
            args.threshold_dissolved,
        )
    )

    output_file = _find_output_file(args.output_csv, args.case)

    df = _collect_results_into_dataframe(
        dates,
        pg_prop_gas,
        pg_prop_dissolved,
        dissolved_prop_key,
        config.injection_wells,
    )

    logging.info("\nExport results to CSV file")
    logging.info(f"    - File path: {output_file}")
    if os.path.isfile(output_file):
        logging.info("Output CSV file already exists => Will overwrite existing file")
    df.to_csv(output_file, index=False)

    dt = time.time() - time_start
    logging.info(f"Total execution time for plume tracking script: {dt:.1f} s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
