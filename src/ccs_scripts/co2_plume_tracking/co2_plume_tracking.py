#!/usr/bin/env python
"""
Calculations for tracking the CO2 plumes from different injection wells,
using SGAS and AMFG/XMF2. Keeps track of which grid cells belong to which
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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from resdata.grid import Grid
from resdata.resfile import ResdataFile

from ccs_scripts.co2_plume_tracking.utils import (
    InjectionWellData,
    PlumeGroups,
    assemble_plume_groups_into_dict,
)

DEFAULT_THRESHOLD_SGAS = 0.2
DEFAULT_THRESHOLD_AMFG = 0.0005
INJ_POINT_THRESHOLD = 60.0

DESCRIPTION = """
Calculations for tracking the CO2 plumes from different injection wells,
using SGAS and AMFG/XMF2. Keeps track of which grid cells belong to which
plume group at each time step, and merges plumes if they meet.

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
        case: str,
    ):
        self.injection_wells: List[InjectionWellData] = []

        input_dict = self.read_config_file(config_file)
        self.make_config_from_input_dict(input_dict, case)

    @staticmethod
    def read_config_file(
        config_file: str,
    ) -> Dict:  # NBNB-AS: Move to common utils-file?
        with open(config_file, "r", encoding="utf8") as stream:
            try:
                config = yaml.safe_load(stream)
                return config
            except yaml.YAMLError as exc:
                logging.error(exc)
                sys.exit(1)

    def make_config_from_input_dict(self, input_dict: Dict, case: str):
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
                            injection_well_info["z"]
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
        "--threshold_sgas",
        default=DEFAULT_THRESHOLD_SGAS,
        type=float,
        help="Threshold for SGAS",
    )
    parser.add_argument(
        "--threshold_amfg",
        default=DEFAULT_THRESHOLD_AMFG,
        type=float,
        help="Threshold for AMFG",
    )
    parser.add_argument(
        "--verbose",
        help="Enable print of detailed information during execution of script",
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
    elif arguments.verbose:
        logging.basicConfig(format="%(message)s", level=logging.INFO)
    else:
        logging.basicConfig(format="%(message)s", level=logging.WARNING)


def _log_input_configuration(arguments: argparse.Namespace) -> None:
    version = "v0.7.0"
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
    logging.info(
        f"Configuration YAML-file : "
        f"{arguments.config_file if arguments.config_file != '' else 'Not specified'}"
    )
    if arguments.output_csv is None or arguments.output_csv == "":
        text = "Not specified, using default"
    else:
        text = arguments.output_csv
    logging.info(f"Output CSV file         : {text}")
    logging.info(f"Threshold SGAS          : {arguments.threshold_sgas}")
    logging.info(f"Threshold AMFG          : {arguments.threshold_amfg}\n")


def _log_configuration(config: Configuration) -> None:
    logging.info("\nInjection well data:")
    logging.info(f"\n{'Number':<8} {'Name':<15} {'x':<15} {'y':<15} {'z':<15}")
    logging.info("-" * 72)
    for i, well in enumerate(config.injection_wells, 1):
        z_str = f"{well.z:<15}" if well.z is not None else "-"
        logging.info(f"{i:<8} {well.name:<15} {well.x:<15} {well.y:<15} {z_str}")
    logging.info("")


def calculate_all_plume_groups(
    grid: Grid,
    unrst: ResdataFile,
    threshold_sgas: float,
    threshold_amfg: float,
    inj_wells: List[InjectionWellData],
) -> Tuple[List[List[str]], Optional[List[List[str]]], Optional[str]]:
    pg_prop_sgas = calculate_plume_groups(
        "SGAS",
        threshold_sgas,
        unrst,
        grid,
        inj_wells,
    )
    if "AMFG" in unrst:
        pg_prop_amfg = calculate_plume_groups(
            "AMFG",
            threshold_amfg,
            unrst,
            grid,
            inj_wells,
        )
        amfg_key = "AMFG"
    elif "XMF2" in unrst:
        pg_prop_amfg = calculate_plume_groups(
            "XMF2",
            threshold_amfg,
            unrst,
            grid,
            inj_wells,
        )
        amfg_key = "XMF2"
    else:
        pg_prop_amfg = None
        amfg_key = None
        logging.warning("WARNING: Neither AMFG nor XMF2 exists as properties.")

    return pg_prop_sgas, pg_prop_amfg, amfg_key


def load_data_and_calculate_plume_groups(
    case: str,
    injection_wells: List[InjectionWellData],
    threshold_sgas: float = DEFAULT_THRESHOLD_SGAS,
    threshold_amfg: float = DEFAULT_THRESHOLD_AMFG,
) -> Tuple[List[List[str]], Optional[List[List[str]]], Optional[str], List[datetime]]:
    logging.info("\nStart calculations for plume tracking")
    grid = Grid(f"{case}.EGRID")
    unrst = ResdataFile(f"{case}.UNRST")

    logging.info(f"Number of active grid cells: {grid.get_num_active()}")

    (pg_prop_sgas, pg_prop_amfg, amfg_key) = calculate_all_plume_groups(
        grid,
        unrst,
        threshold_sgas,
        threshold_amfg,
        injection_wells,
    )

    return pg_prop_sgas, pg_prop_amfg, amfg_key, unrst.report_dates


def _log_number_of_grid_cells(
    n_grid_cells_for_logging: Dict[str, List[int]],
    report_dates: List[datetime],
    attribute_key: str,
):
    logging.info(
        f"Number of grid cells with {attribute_key} above threshold "
        f"for the different plumes:"
    )
    cols = [c for c in n_grid_cells_for_logging]
    header = f"{'Date':<11}"
    widths = {}
    for col in cols:
        widths[col] = max(9, len(col))
        header += f" {col:>{widths[col]}}"
    logging.info("\n" + header)
    logging.info("-" * len(header))
    for i, d in enumerate(report_dates):
        date = d.strftime("%Y-%m-%d")
        row = f"{date:<11}"
        for col in cols:
            n_cells = (
                str(n_grid_cells_for_logging[col][i])
                if n_grid_cells_for_logging[col][i] > 0
                else "-"
            )
            row += f" {n_cells:>{widths[col]}}"
        logging.info(row)
    logging.info("")
    if "?" in n_grid_cells_for_logging:
        no_groups = len(n_grid_cells_for_logging) == 1
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
    grid: Grid, inj_wells: List[InjectionWellData]
) -> Dict[str, List[Tuple[int, int, Optional[int]]]]:
    inj_wells_grid_indices = {}
    for well in inj_wells:
        if well.z is not None:
            inj_wells_grid_indices[well.name] = [
                grid.find_cell(x=well.x, y=well.y, z=well.z)
            ]
        else:
            inj_wells_grid_indices[well.name] = []
            for k in range(grid.get_nz()):
                xy = grid.find_cell_xy(x=well.x, y=well.y, k=k)
                if xy not in inj_wells_grid_indices[well.name]:
                    inj_wells_grid_indices[well.name].append((xy[0], xy[1], None))
    return inj_wells_grid_indices


def calculate_plume_groups(
    attribute_key: str,
    threshold: float,
    unrst: ResdataFile,
    grid: Grid,
    inj_wells: List[InjectionWellData],
) -> List[List[str]]:
    """
    Calculates/tracks the plume groups for a single property.
    The result is a list over the number of time steps, where
    each element is a list over the number of active grid cells.
    The string is the name of the plume group, for instance
    "well_A+well_B" (if well_A and well_B have merged).
    """
    n_time_steps = len(unrst.report_steps)
    n_grid_cells_for_logging: Dict[str, List[int]] = {}
    n_cells = len(unrst[attribute_key][0])

    inj_wells_grid_indices = _find_inj_wells_grid_indices(grid, inj_wells)

    logging.info(f"\nStart calculating plume tracking for {attribute_key}.\n")
    logging.info(f"Progress ({n_time_steps} time steps):")
    logging.info(f"{0:>6.1f} %")

    # Plume group property
    pg_prop = [["" for _ in range(n_cells)] for _ in range(n_time_steps)]
    prev_groups = PlumeGroups(n_cells)
    for i in range(n_time_steps):
        groups = PlumeGroups(n_cells)
        _plume_groups_at_time_step(
            unrst,
            grid,
            attribute_key,
            i,
            threshold,
            prev_groups,
            inj_wells,
            inj_wells_grid_indices,
            n_time_steps,
            groups,
            n_grid_cells_for_logging,
        )

        for j, cell in enumerate(groups.cells):
            all_groups = cell.all_groups
            if all_groups:
                group_string = "+".join(
                    [
                        str(
                            [x.name for x in inj_wells if x.number == y][0]
                            if y != -1
                            else "?"
                        )
                        for y in all_groups
                    ]
                )
                pg_prop[i][j] = group_string

        prev_groups = groups.copy()
        percent = (i + 1) / n_time_steps
        logging.info(f"{percent*100:>6.1f} %")
    logging.info("")

    _log_number_of_grid_cells(
        n_grid_cells_for_logging, unrst.report_dates, attribute_key
    )
    logging.info(f"Done calculating plume tracking for {attribute_key}.")

    return pg_prop


def _plume_groups_at_time_step(
    unrst: ResdataFile,
    grid: Grid,
    attribute_key: str,
    i: int,
    threshold: float,
    prev_groups: PlumeGroups,
    inj_wells: List[InjectionWellData],
    inj_wells_grid_indices: Dict[str, List[Tuple[int, int, Optional[int]]]],
    n_time_steps: int,
    # These arguments will be updated:
    groups: PlumeGroups,
    n_grid_cells_for_logging: Dict[str, List[int]],
):
    data = unrst[attribute_key][i].numpy_view()
    cells_with_co2 = np.where(data > threshold)[0]

    logging.debug("\nPrevious group:")
    prev_groups.debug_print()

    _initialize_groups_from_prev_step_and_inj_wells(
        cells_with_co2,
        prev_groups,
        grid,
        inj_wells,
        inj_wells_grid_indices,
        groups,
    )

    logging.debug("\nCurrent group after first intialization:")
    groups.debug_print()

    groups_to_merge = groups.resolve_undetermined_cells(grid)
    for full_group in groups_to_merge:
        new_group = [x for y in full_group for x in y]
        new_group.sort()
        for cell in groups.cells:
            if cell.has_co2():
                for g in full_group:
                    if set(cell.all_groups) & set(g):
                        cell.all_groups = new_group

    logging.debug("\nCurrent group after resolving undetermined cells:")
    groups.debug_print()

    unique_groups = groups.find_unique_groups()
    for g in unique_groups:
        if g == [-1]:
            if "?" not in n_grid_cells_for_logging:
                n_grid_cells_for_logging["?"] = [0] * n_time_steps
            n_grid_cells_for_logging["?"][i] = len(
                [j for j in cells_with_co2 if groups.cells[j].all_groups == [-1]]
            )
            continue
        indices_this_group = [
            j for j in cells_with_co2 if groups.cells[j].all_groups == g
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
):
    for index in cells_with_co2:
        if prev_groups.cells[index].has_co2():
            groups.cells[index] = prev_groups.cells[index]
        else:
            # This grid cell did not have CO2 in the last time step
            (i, j, k) = grid.get_ijk(active_index=index)
            (x, y, z) = grid.get_xyz(active_index=index)
            found = False
            for well in inj_wells:
                if well.z is not None:
                    same_cell = (i, j, k) == inj_wells_grid_indices[well.name]
                    xyz_close = (
                        abs(x - well.x) <= INJ_POINT_THRESHOLD
                        and abs(y - well.y) <= INJ_POINT_THRESHOLD
                        and abs(z - well.z) <= INJ_POINT_THRESHOLD
                    )
                else:
                    same_cell = False
                    for cell_i, cell_j, _ in inj_wells_grid_indices[well.name]:
                        if (i, j) == (cell_i, cell_j):
                            same_cell = True
                            break
                    xyz_close = (
                        abs(x - well.x) <= INJ_POINT_THRESHOLD
                        and abs(y - well.y) <= INJ_POINT_THRESHOLD
                    )
                if same_cell or xyz_close:
                    found = True
                    groups.cells[index].set_cell_groups(new_groups=[well.number])
                    break
            if not found:
                groups.cells[index].set_undetermined()


def _log_results(
    df: pd.DataFrame,
) -> None:
    dfs = df.sort_values("date")
    col_width = 1 + max(31, max([len(c) for c in df]))
    logging.info("\nSummary of results:")
    logging.info("===================")
    logging.info(
        f"Number of dates {' '*(col_width-5)}: {len(dfs['date'].unique()):>11}"
    )
    logging.info(f"First date      {' '*(col_width-5)}: {dfs['date'].iloc[0]:>11}")
    logging.info(f"Last date       {' '*(col_width-5)}: {dfs['date'].iloc[-1]:>11}")

    for col in df.drop("date", axis=1).columns:
        logging.info(f"End state {col:<{col_width}} : {dfs[col].iloc[-1]:>11.1f}")


def _find_dates(all_results: List[Tuple[Dict, Optional[Dict], Optional[str]]]):
    one_dict = all_results[0][0][next(iter(all_results[0][0]))]
    one_array = one_dict[next(iter(one_dict))]
    dates = [[date] for (date, _) in one_array]
    return dates


def _find_output_file(output: str, case: str):
    if output is None:
        p = Path(case).parents[2]
        p2 = p / "share" / "results" / "tables" / "plume_tracking.csv"
        return str(p2)
    else:
        return output


def _collect_results_into_dataframe(
    report_dates: List[datetime],
    pg_prop_sgas: List[List[str]],
    pg_prop_amfg: Optional[List[List[str]]],
    amfg_key: Optional[str],
) -> pd.DataFrame:
    dates = [[d.strftime("%Y-%m-%d")] for d in report_dates]
    df = pd.DataFrame.from_records(dates, columns=["date"])

    for prop_key, pg_prop in zip(["SGAS", amfg_key], [pg_prop_sgas, pg_prop_amfg]):
        if pg_prop is None or prop_key is None:
            continue
        results = {}
        for i, p in enumerate(pg_prop):
            pg_dict = assemble_plume_groups_into_dict(p)
            for group_name, indices in pg_dict.items():
                group_name = prop_key + "_" + group_name
                if group_name not in results:
                    results[group_name] = np.zeros(
                        shape=(len(dates)),
                        dtype=int,
                    )
                results[group_name][i] = len(indices)
        prop_df = pd.DataFrame(results)
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
    args = _make_parser().parse_args()
    _setup_log_configuration(args)
    _log_input_configuration(args)

    config = Configuration(
        args.config_file,
        args.case,
    )
    _log_configuration(config)

    (pg_prop_sgas, pg_prop_amfg, amfg_key, dates) = (
        load_data_and_calculate_plume_groups(
            args.case,
            config.injection_wells,
            args.threshold_sgas,
            args.threshold_amfg,
        )
    )

    output_file = _find_output_file(args.output_csv, args.case)

    df = _collect_results_into_dataframe(
        dates,
        pg_prop_sgas,
        pg_prop_amfg,
        amfg_key,
    )
    df.to_csv(output_file, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
