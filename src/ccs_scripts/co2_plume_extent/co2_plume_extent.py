#!/usr/bin/env python
"""
Calculates the plume extent from a given coordinate, or well point,
using SGAS and AMFG/XMF2.
"""
import argparse
import getpass
import logging
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from resdata.grid import Grid
from resdata.resfile import ResdataFile

from ccs_scripts.co2_plume_extent._utils import CellGroup, PlumeGroups


DEFAULT_THRESHOLD_SGAS = 0.2
DEFAULT_THRESHOLD_AMFG = 0.0005
INJ_POINT_LATERAL_THRESHOLD = 60.0

DESCRIPTION = """
Calculates the maximum lateral distance of the CO2 plume from a given location,
for instance an injection point. It is also possible to instead calculate the
distance to a point or a line (north-south or east-west). The distances are
calculated for each time step, for both SGAS and AMFG (Pflotran) / YMF2
(Eclipse).

Output is a table on CSV format. Multiple calculations specified in the
YAML-file will be combined to a single CSV-file with many columns.
"""

CATEGORY = "modelling.reservoir"


class CalculationType(Enum):
    """
    Type of distance calculation
    """

    PLUME_EXTENT = 0
    POINT = 1
    LINE = 2

    @classmethod
    def check_for_key(cls, key: str):
        """
        Check if key is in enum
        """
        if key not in cls.__members__:
            error_text = "Illegal calculation type: " + key
            error_text += "\nValid options:"
            for calc_type in CalculationType:
                error_text += "\n  * " + calc_type.name.lower()
            error_text += "\nExiting"
            raise ValueError(error_text)


class LineDirection(Enum):
    """
    Line direction used in distance calculations. We currently only allow
    north/south/east/west.
    """

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

    @classmethod
    def check_for_key(cls, key: str):
        """
        Check if key is in enum
        """
        if key not in cls.__members__:
            error_text = "Illegal line direction: " + key
            error_text += "\nValid options:"
            for line in LineDirection:
                error_text += "\n  * " + line.name.lower()
            error_text += "\nExiting"
            raise ValueError(error_text)


@dataclass
class Calculation:
    type: CalculationType
    direction: Optional[LineDirection]
    name: str
    x: Optional[float]
    y: Optional[float]


@dataclass
class InjectionWellData:
    name: str
    x: float
    y: float
    number: int


class Configuration:
    """
    Holds the configuration for all distance calculations
    """

    def __init__(
        self,
        config_file: str,
        calculation_type: str,
        injection_point_info: str,
        name: str,
        case: str,
    ):
        self.distance_calculations: List[Calculation] = []
        self.injection_wells: List[InjectionWellData] = []
        self.do_plume_tracking: bool = False  # Only available when using a config file

        if config_file != "":
            input_dict = self.read_config_file(config_file)
            self.make_config_from_input_dict(input_dict, case)
        if injection_point_info != "":
            self.make_config_from_input_args(
                calculation_type, injection_point_info, name, case
            )

        if len(self.distance_calculations) == 0:
            logging.warning(
                "WARNING: No CO2 plume distance/extent calculations"
                " specified in the input. Terminating script"
            )
            sys.exit(1)

    def read_config_file(self, config_file: str) -> Dict:
        with open(config_file, "r", encoding="utf8") as stream:
            try:
                config = yaml.safe_load(stream)
                return config
            except yaml.YAMLError as exc:
                logging.error(exc)
                sys.exit(1)

    def make_config_from_input_dict(self, input_dict: Dict, case: str):
        if "do_plume_tracking" in input_dict:
            self.do_plume_tracking = bool(input_dict["do_plume_tracking"])
        else:
            self.do_plume_tracking = False
        if "injection_wells" in input_dict:
            if not isinstance(input_dict["injection_wells"], list):
                logging.error(
                    '\nERROR: Specification under "injection_wells" in '
                    "input YAML file is not a list."
                )
                sys.exit(1)
        elif self.do_plume_tracking:
            logging.warning(
                "\nWARNING: Plume tracking activated, but no injection_wells specified."
                "\n         Plume tracking will therefore be switched off."
            )
        for i, injection_well_info in enumerate(input_dict["injection_wells"], 1):
            args_required = ["name", "x", "y"]
            for arg in args_required:
                if arg not in injection_well_info:
                    logging.error(
                        f'\nERROR: Missing "{arg}" under "injection_wells" for injection well number {i}.'
                    )
                    sys.exit(1)

            self.injection_wells.append(
                InjectionWellData(
                    name=injection_well_info["name"],
                    x=injection_well_info["x"],
                    y=injection_well_info["y"],
                    number=len(self.injection_wells) + 1,
                )
            )

        if "distance_calculations" not in input_dict:
            logging.error(
                '\nERROR: No instance of "distance_calculations" in input YAML file.'
            )
            sys.exit(1)
        if not isinstance(input_dict["distance_calculations"], list):
            logging.error(
                '\nERROR: Specification under "distance_calculations" in '
                "input YAML file is not a list."
            )
            sys.exit(1)
        for i, single_calculation in enumerate(input_dict["distance_calculations"], 1):
            if "type" not in single_calculation:
                logging.error(
                    f'\nERROR: Missing "type" for distance calculation number {i}.'
                )
                sys.exit(1)
            type_str = single_calculation["type"].upper()
            CalculationType.check_for_key(type_str)
            calculation_type = CalculationType[type_str]

            name = single_calculation["name"] if "name" in single_calculation else ""

            direction = None
            if calculation_type == CalculationType.LINE:
                if "direction" not in single_calculation:
                    logging.error(
                        f'\nERROR: Missing "direction" for distance '
                        f'calculation number {i}. Needed when "type" = "line".'
                    )
                    sys.exit(1)
                else:
                    direction_str = single_calculation["direction"].upper()
                    LineDirection.check_for_key(direction_str)
                    direction = LineDirection[direction_str]
            else:
                if "direction" in single_calculation:
                    logging.warning(
                        f'\nWARNING: No need to specify "direction" when '
                        f'"type" is not "line" (distance calculation number '
                        f"{i})."
                    )

            x = single_calculation["x"] if "x" in single_calculation else None
            y = single_calculation["y"] if "y" in single_calculation else None
            well_name = (
                single_calculation["well_name"]
                if "well_name" in single_calculation
                else None
            )

            if calculation_type == CalculationType.POINT or (
                calculation_type == CalculationType.PLUME_EXTENT and well_name is None
            ):
                if x is None:
                    logging.error(
                        f'\nERROR: Missing "x" for distance calculation number {i}.'
                    )
                    sys.exit(1)
                if y is None:
                    logging.error(
                        f'\nERROR: Missing "y" for distance calculation number {i}.'
                    )
                    sys.exit(1)
            elif calculation_type == CalculationType.LINE:
                if direction in (LineDirection.EAST, LineDirection.WEST):
                    if x is None:
                        logging.error(
                            f'\nERROR: Missing "x" for distance calculation number {i}.'
                        )
                        sys.exit(1)
                    if y is not None:
                        logging.warning(
                            f'\nWARNING: No need to specify "y" for distance '
                            f"calculation number {i}."
                        )
                elif direction in (LineDirection.NORTH, LineDirection.SOUTH):
                    if y is None:
                        logging.error(
                            f'\nERROR: Missing "y" for distance calculation number {i}.'
                        )
                        sys.exit(1)
                    if x is not None:
                        logging.warning(
                            f'\nWARNING: No need to specify "x" for distance '
                            f"calculation number {i}."
                        )

            if well_name is not None:
                (x, y) = self.calculate_well_coordinates(case, well_name)

            calculation = Calculation(
                type=calculation_type,
                direction=direction,
                name=name,
                x=x,
                y=y,
            )
            self.distance_calculations.append(calculation)

    def make_config_from_input_args(
        self, calculation_type_str: str, injection_point_info: str, name: str, case: str
    ):
        type_str = calculation_type_str.upper()
        CalculationType.check_for_key(type_str)
        calculation_type = CalculationType[type_str]

        direction = None
        x = None
        y = None

        if (
            len(injection_point_info) > 0
            and injection_point_info[0] == "["
            and injection_point_info[-1] == "]"
        ):
            values = injection_point_info[1:-1].split(",")
            if len(values) != 2:
                if calculation_type == CalculationType.PLUME_EXTENT:
                    logging.error(
                        "ERROR: Invalid input. injection_point_info must be on"
                        ' the format "[x,y]" or "well_name" when '
                        "calculation_type is 'plume_extent'."
                    )
                elif calculation_type == CalculationType.POINT:
                    logging.error(
                        "ERROR: Invalid input. injection_point_info must be on"
                        ' the format "[x,y]" when calculation_type is '
                        "'point'."
                    )
                elif calculation_type == CalculationType.LINE:
                    logging.error(
                        "Invalid input: injection_point_info must be on the "
                        'format "[direction, x or y]" when '
                        "calculation_type is 'line'."
                    )
                sys.exit(1)

            if calculation_type in (
                CalculationType.PLUME_EXTENT,
                CalculationType.POINT,
            ):
                try:
                    (x, y) = (float(values[0]), float(values[1]))
                    logging.info(f"Using injection coordinates: [{x}, {y}]")
                except ValueError:
                    logging.error(
                        "ERROR: Invalid input. When providing two arguments "
                        "(x and y coordinates) for injection point info they "
                        "need to be floats."
                    )
                    sys.exit(1)
            elif calculation_type == CalculationType.LINE:
                try:
                    (direction_str, coord) = (str(values[0]), float(values[1]))
                    logging.info(f"Using injection info: [{direction_str}, {coord}]")
                except ValueError:
                    logging.error(
                        "ERROR: Invalid input. When providing two arguments "
                        "(direction and x or y) for injection point, the "
                        "direction needs to be a string and the coordinate "
                        "needs to be a float."
                    )
                    sys.exit(1)

                direction_str = direction_str.upper()
                LineDirection.check_for_key(direction_str)
                direction = LineDirection[direction_str]

                if direction in (LineDirection.EAST, LineDirection.WEST):
                    x = coord
                elif direction in (LineDirection.NORTH, LineDirection.SOUTH):
                    y = coord
        else:
            # Specification is now either a well name (for plume extent) or incorrect
            if calculation_type != CalculationType.PLUME_EXTENT:
                logging.error(
                    "ERROR: Invalid input. For plume_extent, the injection "
                    f'point info specified ("{injection_point_info}") is '
                    'incorrect. It should be on the format "[x,y]" or '
                    '"well_name".'
                )
                sys.exit(1)

            (x, y) = self.calculate_well_coordinates(case, injection_point_info)

        calculation = Calculation(
            type=calculation_type,
            direction=direction,
            name=name,
            x=x,
            y=y,
        )
        self.distance_calculations.append(calculation)

    def calculate_well_coordinates(
        self, case: str, well_name: str, well_picks_path: Optional[str] = None
    ):
        logging.info(f"Using well to find coordinates: {well_name}")

        if well_picks_path is None:
            p = Path(case).parents[2]
            p2 = p / "share" / "results" / "wells" / "well_picks.csv"
            logging.info(f"Using default well picks path : {p2}")
        else:
            p2 = Path(well_picks_path)

        df = pd.read_csv(p2)
        logging.info("Done reading well picks CSV file")
        logging.debug("Well picks read from CSV file:")
        logging.debug(df)

        if well_name not in list(df["WELL"]):
            logging.error(
                f"No matches for well name {well_name}, input is either mistyped "
                "or well does not exist."
            )
            sys.exit(1)

        df = df[df["WELL"] == well_name]
        logging.info(f"Number of well picks for well {well_name}: {len(df)}")
        logging.info("Using the well pick with the largest measured depth.")

        df = df[df["X_UTME"].notna()]
        df = df[df["Y_UTMN"].notna()]

        max_id = df["MD"].idxmax()
        max_md_row = df.loc[max_id]
        x = max_md_row["X_UTME"]
        y = max_md_row["Y_UTMN"]
        md = max_md_row["MD"]
        surface = max_md_row["HORIZON"] if "HORIZON" in max_md_row else "-"
        logging.info(
            f"Injection coordinates: [{x:.2f}, {y:.2f}] (surface: {surface}, "
            f"MD: {md:.2f})"
        )
        return (x, y)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate plume extent (distance)")
    parser.add_argument("case", help="Name of Eclipse case")
    parser.add_argument(
        "--config_file",
        help="YML file with configurations for distance calculations.",
        default="",
    )
    parser.add_argument(
        "--injection_point_info",
        help="Input depends on calculation_type. \
        For 'plume_extent': Either the name of the injection well (string) or \
        the x and y coordinates (two floats, '[x,y]') to calculate plume extent from. \
        For 'point': the x and y coordinates (two floats, '[x,y]'). \
        For 'line': [direction, value] where direction must be \
        'east'/'west'/'north'/'south' and value is the \
        corresponding x or y value that defines this line.",
        default="",
    )
    parser.add_argument(
        "--calculation_type",
        help="Options: \
        'plume_extent': Maximum distance of plume from input (injection) coordinate. \
        'point': Minimum distance from plume to a point, e.g. plume approaching \
        a dangerous area. \
        'line': Minimum distance from plume to an \
        eastern/western/northern/southern line.",
        default="plume_extent",
        type=str,
    )
    parser.add_argument(
        "--output",
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
        "--name",
        default="",
        type=str,
        help="Name that will be included in the column of the CSV file",
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
    logging.info("CCS-scripts - Plume extent calculations")
    logging.info("=======================================")
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
    if arguments.injection_point_info != "":
        logging.info("Configuration from args :")
        logging.info(f"    Injection point info: {arguments.injection_point_info}")
        logging.info(f"    Calculation type    : {arguments.calculation_type}")
        if arguments.name != "":
            logging.info(
                f"    Column name         : "
                f"{arguments.name if arguments.name != '' else 'Not specified'}"
            )
    else:
        logging.info("Configuration from args : Not specified")
    if arguments.name != "":
        text = "Not specified, using default"
    else:
        text = arguments.output
    logging.info(f"Output CSV file         : {text}")
    logging.info(f"Threshold SGAS          : {arguments.threshold_sgas}")
    logging.info(f"Threshold AMFG          : {arguments.threshold_amfg}\n")


def _log_distance_calculation_configurations(config: Configuration) -> None:
    logging.info("\nDistance calculation configurations:")
    logging.info(
        f"\n{'Number':<8} {'Type':<14} {'Name':<15} {'Direction':<12} "
        f"{'x':<15} {'y':<15}"
    )
    logging.info("-" * 84)
    for i, calc in enumerate(config.distance_calculations, 1):
        name = calc.name if calc.name != "" else "-"
        direction = calc.direction.name.lower() if calc.direction is not None else "-"
        x = calc.x if calc.x is not None else "-"
        y = calc.y if calc.y is not None else "-"
        logging.info(
            f"{i:<8} {calc.type.name.lower():<14} {name:<15} {direction:<12} "
            f"{x:<15} {y:<15}"
        )
    logging.info("")

    logging.info(f"\nPlume tracking activated: {'yes' if config.do_plume_tracking else 'no'}")
    logging.info("\nInjection well data:")
    logging.info(f"\n{'Number':<8} {'Name':<15} {'x':<15} {'y':<15}")
    logging.info("-" * 56)
    for i, well in enumerate(config.injection_wells, 1):
        logging.info(f"{i:<8} {well.name:<15} {well.x:<15} {well.y:<15}")
    logging.info("")


def _calculate_grid_cell_distances(
    inj_wells: list[InjectionWellData],
    nactive: int,
    calculation_type: CalculationType,
    grid: Grid,
    config: Calculation,
):
    # Note: Currently we loop over injection wells even for POINT and LINE,
    #       but this should not be necessary.
    dist = {}
    for well in inj_wells:
        name = well.name
        dist[name] = np.zeros(shape=(nactive,))
        if calculation_type in (CalculationType.PLUME_EXTENT, CalculationType.POINT):
            if calculation_type == CalculationType.PLUME_EXTENT:
                x0 = well.x
                y0 = well.y
            else:
                x0 = config.x
                y0 = config.y
            for i in range(nactive):
                center = grid.get_xyz(active_index=i)
                dist[well.name][i] = np.sqrt(
                    (center[0] - x0) ** 2 + (center[1] - y0) ** 2
                )
        elif calculation_type == CalculationType.LINE:
            line_value = config.x
            ind = 0  # Use x-coordinate
            if config.direction in (LineDirection.NORTH, LineDirection.SOUTH):
                line_value = config.y
                ind = 1  # Use y-coordinate

            factor = 1
            if config.direction in (LineDirection.WEST, LineDirection.SOUTH):
                factor = -1

            for i in range(nactive):
                center = grid.get_xyz(active_index=i)
                dist[name][i] = factor * (line_value - center[ind])
            dist[name][dist[name] < 0] = 0.0

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
            f"    Smallest distance grid cell to {text} : {min(distance):>10.1f}"
        )
        logging.info(
            f"    Largest distance grid cell to {text}  : {max(distance):>10.1f}"
        )
        logging.info(
            f"    Average distance grid cell to {text}  : {sum(distance) / len(distance):>10.1f}"
        )
    logging.info("")

    return dist


def calculate_single_distances(
    nactive: int,
    grid: Grid,
    unrst: ResdataFile,
    threshold_sgas: float,
    threshold_amfg: float,
    config: Calculation,
    inj_wells: list[InjectionWellData],
    do_plume_tracking: bool,
):
    calculation_type = config.type

    # First calculate distance from point/line to center of all cells
    dist = _calculate_grid_cell_distances(
        inj_wells, nactive, calculation_type, grid, config
    )

    sgas_results = _find_distances_per_time_step(
        "SGAS", calculation_type, threshold_sgas, unrst, grid, dist, inj_wells
    )
    if False and "AMFG" in unrst:
        amfg_results = _find_distances_per_time_step(
            "AMFG", calculation_type, threshold_amfg, unrst, grid, dist, inj_wells
        )
        amfg_key = "AMFG"
    elif "XMF2" in unrst:
        amfg_results = _find_distances_per_time_step(
            "XMF2", calculation_type, threshold_amfg, unrst, grid, dist, inj_wells
        )
        amfg_key = "XMF2"
    else:
        amfg_results = None
        amfg_key = None
        logging.warning("WARNING: Neither AMFG nor XMF2 exists as properties.")

    return (sgas_results, amfg_results, amfg_key)


def calculate_distances(
    case: str,
    config: Configuration,
    threshold_sgas: float = DEFAULT_THRESHOLD_SGAS,
    threshold_amfg: float = DEFAULT_THRESHOLD_AMFG,
) -> List[Tuple[dict, Optional[dict], Optional[str]]]:
    """
    Find distance (plume extent / distance to point / distance to line) per
    date for SGAS and AMFG/XMF2.
    """
    logging.info("\nStart calculating distances")
    grid = Grid(f"{case}.EGRID")
    unrst = ResdataFile(f"{case}.UNRST")

    nactive = grid.get_num_active()
    logging.info(f"Number of active grid cells: {nactive}")

    all_results = []
    for i, single_config in enumerate(config.distance_calculations, 1):
        logging.info(f"\nCalculating distances for configuration number: {i}\n")
        (a, b, c) = calculate_single_distances(
            nactive,
            grid,
            unrst,
            threshold_sgas,
            threshold_amfg,
            single_config,
            config.injection_wells,
            config.do_plume_tracking,
        )
        all_results.append((a, b, c))
        logging.info(f"Done calculating distances for configuration number: {i}\n")
    return all_results


def _temp_add_well3(i, plumeix):
    # Temp add cells with co2 for made up injection well 3:
    if i > 11 and True:
        temp_inj3_list = [
            584,
            585,
            586,
            609,
            610,
            611,
            636,
            637,
            638,
            1333,
            1334,
            1335,
            1357,
            1358,
            1379,
            1380,
        ]
        if i > 14:
            temp_inj3_list += [587, 612]
        if i > 20:
            temp_inj3_list += [588, 613]
        plumeix = np.concatenate((plumeix, np.array(temp_inj3_list)))
    return plumeix


def _log_number_of_grid_cells(
    n_grid_cells_for_logging: dict[str, list[int]], report_dates, attribute_key
):
    logging.info(
        f"Number of grid cells with {attribute_key} above threshold for the different plumes:"
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
        logging.warning("Warning: Plume group not found for some grid cells with CO2.")
        logging.warning("See table above, under column '?'.\n")


def _find_distances_per_time_step(
    attribute_key: str,
    calculation_type: CalculationType,
    threshold: float,
    unrst: ResdataFile,
    grid: Grid,
    dist: dict[str, np.ndarray],
    inj_wells: list[InjectionWellData],
) -> dict:
    """
    Find value of distance metric for each step
    """
    n_time_steps = len(unrst.report_steps)
    dist_per_group = {}
    n_grid_cells_for_logging = {}

    n_cells = len(unrst[attribute_key][0].numpy_view())
    # print(f"n_cells = {n_cells}")
    prev_groups = PlumeGroups(n_cells)

    logging.info(f"\nStart calculating plume extent for {attribute_key}.\n")
    logging.info(f"Progress ({n_time_steps} time steps):")
    logging.info(f"{0:>6.1f} %")
    for i in range(n_time_steps):
        data = unrst[attribute_key][i].numpy_view()
        cells_with_co2 = np.where(data > threshold)[0]
        cells_with_co2 = _temp_add_well3(i, cells_with_co2)
        # print(f"Number of grid cells with CO2 this time step: {len(cells_with_co2)}")

        # print("Previous group:")
        # prev_groups._temp_print()
        groups = PlumeGroups(n_cells)
        for index in cells_with_co2:
            if prev_groups.cells[index].has_co2():
                groups.cells[index] = prev_groups.cells[index]
            else:
                # This grid cell did not have CO2 in the last time step
                (x, y, _) = grid.get_xyz(active_index=index)
                found = False
                for well in inj_wells:
                    if (
                        abs(x - well.x) <= INJ_POINT_LATERAL_THRESHOLD
                        and abs(y - well.y) <= INJ_POINT_LATERAL_THRESHOLD
                    ):
                        found = True
                        groups.cells[index].set_cell_groups(new_groups=[well.number])
                        break
                if not found:
                    groups.cells[index].set_undetermined()
        # print("Current group:")
        # groups._temp_print()

        groups_to_merge = groups.resolve_undetermined_cells(grid)
        # print(f"Groups to merge: {groups_to_merge}")

        for full_group in groups_to_merge:
            new_group = [x for y in full_group for x in y]
            new_group.sort()
            for cell in groups.cells:
                if cell.has_co2():
                    for g in full_group:
                        if set(cell.all_groups) & set(g):
                            cell.all_groups = new_group

        # print("Current group after resolving undetermined cells:")
        # groups._temp_print()

        unique_groups = groups._find_unique_groups()
        for g in unique_groups:
            if g == [-1]:
                if "?" not in n_grid_cells_for_logging:
                    n_grid_cells_for_logging["?"] = [0] * n_time_steps
                n_grid_cells_for_logging["?"][i] = len(
                    [i for i in cells_with_co2 if groups.cells[i].all_groups == [-1]]
                )
                continue
            # Calculate distance metric for this group
            indices_this_group = [
                i for i in cells_with_co2 if groups.cells[i].all_groups == g
            ]
            if len(indices_this_group) == 0:
                result = {}
                for single_inj_number in g:  # Can do this in a better way?
                    result[single_inj_number] = np.nan
            else:
                result = {}
                for single_inj_number in g:
                    well_name = [
                        x.name for x in inj_wells if x.number == single_inj_number
                    ][0]
                    result[single_inj_number] = 0.0
                    if calculation_type == CalculationType.PLUME_EXTENT:
                        result[single_inj_number] = dist[well_name][
                            indices_this_group
                        ].max()
                    elif calculation_type in (
                        CalculationType.POINT,
                        CalculationType.LINE,
                    ):
                        result[single_inj_number] = dist[well_name][
                            indices_this_group
                        ].min()

            group_string = "+".join(
                [str([x.name for x in inj_wells if x.number == y][0]) for y in g]
            )
            if group_string not in dist_per_group:
                dist_per_group[group_string] = {
                    s: np.zeros(shape=(n_time_steps,)) for s in g
                }
                n_grid_cells_for_logging[group_string] = [
                    0
                ] * n_time_steps  # np.zeros(shape=(n_time_steps,))
            for s in g:
                dist_per_group[group_string][s][i] = result[s]
                n_grid_cells_for_logging[group_string][i] = len(indices_this_group)

        prev_groups = groups.copy()
        percent = (i + 1) / n_time_steps
        logging.info(f"{percent*100:>6.1f} %")
    logging.info("")

    _log_number_of_grid_cells(
        n_grid_cells_for_logging, unrst.report_dates, attribute_key
    )

    outputs = {}
    for group_name, single_group_distances in dist_per_group.items():
        outputs[group_name] = {}
        for single_group, distances in single_group_distances.items():
            # Find well name:
            well_name = [x.name for x in inj_wells if x.number == single_group][0]
            outputs[group_name][well_name] = []
            for i, d in enumerate(unrst.report_dates):
                date_and_result = [d.strftime("%Y-%m-%d"), distances[i]]
                outputs[group_name][well_name].append(date_and_result)

    logging.info(f"Done calculating plume extent for {attribute_key}.")
    return outputs


def _find_output_file(output: str, case: str):
    if output is None:
        p = Path(case).parents[2]
        p2 = p / "share" / "results" / "tables" / "plume_extent.csv"
        return str(p2)
    else:
        return output


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


def _find_dates(all_results: List[Tuple[dict, Optional[dict], Optional[str]]]):
    one_dict = all_results[0][0][next(iter(all_results[0][0]))]
    one_array = one_dict[next(iter(one_dict))]
    dates = [[date] for (date, _) in one_array]
    return dates


def _find_column_name(single_config, n_calculations: int, calculation_number: int):
    if single_config.type == CalculationType.PLUME_EXTENT:
        col = "MAX_"
    elif single_config.type in (CalculationType.POINT, CalculationType.LINE):
        col = "MIN_"
    else:
        col = "?"

    if single_config.name != "":
        col = col + single_config.name
    else:
        calc_number = "" if n_calculations == 1 else str(calculation_number)
        col = col + f"{single_config.type.name.upper()}{calc_number}"

    return col


def _collect_results_into_dataframe(
    all_results: List[Tuple[dict, Optional[dict], Optional[str]]],
    config: Configuration,
) -> pd.DataFrame:
    dates = _find_dates(all_results)
    df = pd.DataFrame.from_records(dates, columns=["date"])
    for i, (result, single_config) in enumerate(
        zip(all_results, config.distance_calculations), 1
    ):
        (sgas_results, amfg_results, amfg_key) = result

        # NBNB-AS:
        # for key, value in sgas_results.items():
        #     print(f"key: {key}")
        #     for key2, value2 in value.items():
        #         print(f"  key2: {key2}")
        #         print(f"   {value2}")
        # exit()

        col = _find_column_name(single_config, len(config.distance_calculations), i)

        for group_str, sgas_results in sgas_results.items():
            for well_name, sgas_result in sgas_results.items():
                full_col_name = (
                    col + "_SGAS_" + "GROUP_" + group_str + "_FROM_" + well_name
                )
                sgas_df = pd.DataFrame.from_records(
                    sgas_result, columns=["date", full_col_name]
                )
                df = pd.merge(df, sgas_df, on="date")
        if amfg_results is not None:
            for group_str, amfg_results in amfg_results.items():
                for well_name, amfg_result in amfg_results.items():
                    if amfg_result is not None:
                        if amfg_key is None:
                            amfg_key_str = "?"
                        else:
                            amfg_key_str = amfg_key
                        full_col_name = (
                            col
                            + "_"
                            + amfg_key_str
                            + "_GROUP_"
                            + group_str
                            + "_FROM_"
                            + well_name
                        )
                        amfg_df = pd.DataFrame.from_records(
                            amfg_result, columns=["date", full_col_name]
                        )
                        df = pd.merge(df, amfg_df, on="date")
    return df


def _calculate_well_coordinates(
    case: str, injection_point_info: str, well_picks_path: Optional[str] = None
) -> Tuple[float, float]:
    """
    Find coordinates of injection point
    """
    if (
        len(injection_point_info) > 0
        and injection_point_info[0] == "["
        and injection_point_info[-1] == "]"
    ):
        coords = injection_point_info[1:-1].split(",")
        if len(coords) == 2:
            try:
                coordinates = (float(coords[0]), float(coords[1]))
                logging.info(
                    f"Using injection coordinates: [{coordinates[0]}, {coordinates[1]}]"
                )
                return coordinates
            except ValueError:
                logging.error(
                    "Invalid input: When providing two arguments (x and y coordinates)\
                    for injection point info they need to be floats."
                )
                sys.exit(1)
    well_name = injection_point_info
    logging.info(f"Using well to find coordinates: {well_name}")

    if well_picks_path is None:
        p = Path(case).parents[2]
        p2 = p / "share" / "results" / "wells" / "well_picks.csv"
        logging.info(f"Using default well picks path : {p2}")
    else:
        p2 = Path(well_picks_path)

    df = pd.read_csv(p2)
    logging.info("Done reading well picks CSV file")
    logging.debug("Well picks read from CSV file:")
    logging.debug(df)

    if well_name not in list(df["WELL"]):
        logging.error(
            f"No matches for well name {well_name}, input is either mistyped "
            "or well does not exist."
        )
        sys.exit(1)

    df = df[df["WELL"] == well_name]
    logging.info(f"Number of well picks for well {well_name}: {len(df)}")
    logging.info("Using the well pick with the largest measured depth.")

    df = df[df["X_UTME"].notna()]
    df = df[df["Y_UTMN"].notna()]

    max_id = df["MD"].idxmax()
    max_md_row = df.loc[max_id]
    x = max_md_row["X_UTME"]
    y = max_md_row["Y_UTMN"]
    md = max_md_row["MD"]
    surface = max_md_row["HORIZON"] if "HORIZON" in max_md_row else "-"
    logging.info(
        f"Injection coordinates: [{x:.2f}, {y:.2f}] (surface: {surface}, MD: {md:.2f})"
    )

    return (x, y)


def _find_input_point(injection_point_info: str) -> Tuple[float, float]:
    if (
        len(injection_point_info) > 0
        and injection_point_info[0] == "["
        and injection_point_info[-1] == "]"
    ):
        coords = injection_point_info[1:-1].split(",")
        if len(coords) == 2:
            try:
                coordinates = (float(coords[0]), float(coords[1]))
                logging.info(
                    f"Using point coordinates: [{coordinates[0]}, {coordinates[1]}]"
                )
                return coordinates
            except ValueError:
                logging.error(
                    "Invalid input: When providing two arguments (x and y coordinates) "
                    "for point they need to be floats."
                )
                sys.exit(1)
    logging.error(
        "Invalid input: injection_point_info must be on the format [x,y]"
        "when calculation_type is 'point'"
    )
    sys.exit(1)


def _find_input_line(injection_point_info: str) -> Tuple[str, float]:
    if (
        len(injection_point_info) > 0
        and injection_point_info[0] == "["
        and injection_point_info[-1] == "]"
    ):
        coords = injection_point_info[1:-1].split(",")
        if len(coords) == 2:
            try:
                direction = coords[0]
                direction = direction.lower()
                if direction not in ["east", "west", "north", "south"]:
                    raise ValueError(
                        "Invalid line direction. Choose from "
                        "'east'/'west'/'north'/'south'"
                    )
                value = float(coords[1])
                coordinates = (direction, value)
                logging.info(f"Using line data: [{direction}, {value}]")
                return coordinates
            except ValueError as error:
                logging.error(
                    "Invalid input: injection_point_info must be on the format "
                    "[direction, value] when calculation_type is 'line'."
                )
                logging.error(error)
                sys.exit(1)
    logging.error(
        "Invalid input: injection_point_info must be on the format "
        "[direction, value] when calculation_type is 'line'"
    )
    sys.exit(1)


def main():
    """
    Calculate plume extent or distance to point/line using EGRID and
    UNRST-files. Calculated for SGAS and AMFG/XMF2. Output is distance per
    date written to a CSV file.
    """
    args = _make_parser().parse_args()
    args.name = args.name.upper() if args.name is not None else None
    _setup_log_configuration(args)
    _log_input_configuration(args)

    config = Configuration(
        args.config_file,
        args.calculation_type,
        args.injection_point_info,
        args.name,
        args.case,
    )
    _log_distance_calculation_configurations(config)

    all_results = calculate_distances(
        args.case,
        config,
        args.threshold_sgas,
        args.threshold_amfg,
    )

    output_file = _find_output_file(args.output, args.case)

    df = _collect_results_into_dataframe(
        all_results,
        config,
    )
    _log_results(df)
    df.to_csv(output_file, index=False, na_rep="0.0")  # How to handle nan-values?
    logging.info("\nDone exporting results to CSV file.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
