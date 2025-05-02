#!/usr/bin/env python
"""
Calculates the plume extent from a given coordinate, or well point,
using SGAS and the dissolved property (AMFG/XMF2).
"""
import argparse
import getpass
import logging
import os
import platform
import socket
import string
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

from ccs_scripts.co2_containment.co2_containment import str_to_bool
from ccs_scripts.co2_plume_tracking.co2_plume_tracking import calculate_plume_groups
from ccs_scripts.co2_plume_tracking.utils import (
    InjectionWellData,
    assemble_plume_groups_into_dict,
    sort_well_names,
)

DEFAULT_THRESHOLD_GAS = 0.2
DEFAULT_THRESHOLD_DISSOLVED = 0.0005
INJ_POINT_THRESHOLD = 60.0

DESCRIPTION = """
Calculates the maximum lateral distance of the CO2 plume from a given location,
for instance an injection point. It is also possible to instead calculate the
distance to a point or a line (north-south or east-west). The distances are
calculated for each time step, for both SGAS and AMFG (Pflotran) / XMF2
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
    column_name: str
    x: Optional[float]
    y: Optional[float]


class Configuration:
    """
    Holds the configuration for all distance calculations
    """

    def __init__(
        self,
        config_file: str,
        calculation_type: str,
        injection_point_info: str,
        column_name: str,
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
                calculation_type, injection_point_info, column_name, case
            )

        if len(self.distance_calculations) == 0:
            logging.warning(
                "WARNING: No CO2 plume distance/extent calculations"
                " specified in the input. Terminating script"
            )
            sys.exit(1)

    @staticmethod
    def read_config_file(config_file: str) -> Dict:
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
            self.do_plume_tracking = False
        if "injection_wells" in input_dict:
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

            column_name = (
                single_calculation["column_name"]
                if "column_name" in single_calculation
                else ""
            )

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
                calculation_type == CalculationType.PLUME_EXTENT
                and well_name is None
                and len(self.injection_wells) == 0
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
                column_name=column_name,
                x=x,
                y=y,
            )
            self.distance_calculations.append(calculation)

    def make_config_from_input_args(
        self,
        calculation_type_str: str,
        injection_point_info: str,
        column_name: str,
        case: str,
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
                        "ERROR: Invalid input. inj_point must be on"
                        ' the format "[x,y]" or "well_name" when '
                        "calc_type is 'plume_extent'."
                    )
                elif calculation_type == CalculationType.POINT:
                    logging.error(
                        "ERROR: Invalid input. inj_point must be on"
                        ' the format "[x,y]" when calc_type is '
                        "'point'."
                    )
                elif calculation_type == CalculationType.LINE:
                    logging.error(
                        "Invalid input: inj_point must be on the "
                        'format "[direction, x or y]" when '
                        "calc_type is 'line'."
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
            column_name=column_name,
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
        "--inj_point",
        help="Input depends on calc_type. \
        For 'plume_extent': Either the name of the injection well (string) or \
        the x and y coordinates (two floats, '[x,y]') to calculate plume extent from. \
        For 'point': the x and y coordinates (two floats, '[x,y]'). \
        For 'line': [direction, value] where direction must be \
        'east'/'west'/'north'/'south' and value is the \
        corresponding x or y value that defines this line.",
        default="",
    )
    parser.add_argument(
        "--calc_type",
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
        "--column_name",
        default="",
        type=str,
        help="Name that will be included in the column of the CSV file",
    )
    parser.add_argument(
        "--no_logging",
        help="Skip print of detailed information during execution of script",
        type=str_to_bool,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--debug",
        help="Enable print of debugging data during execution of script. "
        "Normally not necessary for most users.",
        type=str_to_bool,
        nargs="?",
        const=True,
    )

    return parser


def _replace_default_dummies_from_ert(args):
    if args.no_logging == "-1":
        args.no_logging = False
    if args.debug == "-1":
        args.debug = False


def _setup_log_configuration(arguments: argparse.Namespace) -> None:
    if arguments.debug:
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    elif arguments.no_logging:
        logging.basicConfig(format="%(message)s", level=logging.WARNING)
    else:
        logging.basicConfig(format="%(message)s", level=logging.INFO)


def _log_input_configuration(arguments: argparse.Namespace) -> None:
    version = "v0.9.1"
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
    if not os.path.isabs(arguments.case):
        logging.info(f"  => Absolute path      : {os.path.abspath(arguments.case)}")
    logging.info(
        f"Configuration YAML-file : "
        f"{arguments.config_file if arguments.config_file != '' else 'Not specified'}"
    )
    if arguments.inj_point != "":
        logging.info("Configuration from args :")
        logging.info(f"    Injection point info: {arguments.inj_point}")
        logging.info(f"    Calculation type    : {arguments.calc_type}")
        col = arguments.column_name
        if col != "":
            logging.info(
                f"    Column name         : " f"{col if col != '' else 'Not specified'}"
            )
    else:
        logging.info("Configuration from args : Not specified")
    if arguments.output_csv is None or arguments.output_csv == "":
        text = "Not specified, using default"
    else:
        text = arguments.output_csv
    logging.info(f"Output CSV file         : {text}")
    logging.info(f"Threshold gas           : {arguments.threshold_gas}")
    logging.info(f"Threshold dissolved     : {arguments.threshold_dissolved}\n")


def _log_distance_calculation_configurations(config: Configuration) -> None:
    logging.info("\nDistance calculation configurations:")
    logging.info(
        f"\n{'Number':<8} {'Type':<14} {'Name':<15} {'Direction':<12} "
        f"{'x':<15} {'y':<15}"
    )
    logging.info("-" * 84)
    for i, calc in enumerate(config.distance_calculations, 1):
        column_name = calc.column_name if calc.column_name != "" else "-"
        direction = calc.direction.name.lower() if calc.direction is not None else "-"
        x = calc.x if calc.x is not None else "-"
        y = calc.y if calc.y is not None else "-"
        logging.info(
            f"{i:<8} {calc.type.name.lower():<14} {column_name:<15} {direction:<12} "
            f"{x:<15} {y:<15}"
        )
    logging.info("")

    logging.info(
        f"\nPlume tracking activated: {'yes' if config.do_plume_tracking else 'no'}"
    )
    logging.info("\nInjection well data:")
    logging.info(f"\n{'Number':<8} {'Name':<15} {'x':<15} {'y':<15} {'z':<15}")
    logging.info("-" * 72)
    for i, well in enumerate(config.injection_wells, 1):
        z_str = f"{well.z[0]:<15}" if well.z is not None else "-"
        logging.info(f"{i:<8} {well.name:<15} {well.x:<15} {well.y:<15} {z_str}")
    logging.info("")


def _calculate_grid_cell_distances(
    inj_wells: Optional[List[InjectionWellData]],
    nactive: int,
    calculation_type: CalculationType,
    grid: Grid,
    config: Calculation,
):
    dist = {}
    if calculation_type == CalculationType.PLUME_EXTENT:
        if inj_wells is None or len(inj_wells) == 0:
            # Also needed when no config file is used
            x0 = config.x
            y0 = config.y
            dist["WELL"] = np.zeros(shape=(nactive,))
            for i in range(nactive):
                center = grid.get_xyz(active_index=i)
                dist["WELL"][i] = np.sqrt((center[0] - x0) ** 2 + (center[1] - y0) ** 2)
        else:
            for well in inj_wells:
                name = well.name
                x0 = well.x
                y0 = well.y
                dist[name] = np.zeros(shape=(nactive,))
                for i in range(nactive):
                    center = grid.get_xyz(active_index=i)
                    dist[well.name][i] = np.sqrt(
                        (center[0] - x0) ** 2 + (center[1] - y0) ** 2
                    )
    elif calculation_type == CalculationType.POINT:
        dist["ALL"] = np.zeros(shape=(nactive,))
        x0 = config.x
        y0 = config.y
        for i in range(nactive):
            center = grid.get_xyz(active_index=i)
            dist["ALL"][i] = np.sqrt((center[0] - x0) ** 2 + (center[1] - y0) ** 2)
    elif calculation_type == CalculationType.LINE:
        dist["ALL"] = np.zeros(shape=(nactive,))
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
            dist["ALL"][i] = factor * (line_value - center[ind])
        dist["ALL"][dist["ALL"] < 0] = 0.0

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
            f"    Average distance grid cell to {text}  : "
            f"{sum(distance) / len(distance):>10.1f}"
        )
    logging.info("")

    return dist


def calculate_single_distances(
    nactive: int,
    grid: Grid,
    unrst: ResdataFile,
    threshold_gas: float,
    threshold_dissolved: float,
    config: Calculation,
    inj_wells: Optional[List[InjectionWellData]],
    plume_groups_gas: Optional[List[List[str]]],
    plume_groups_dissolved: Optional[List[List[str]]],
):
    calculation_type = config.type

    # Calculate distance from point/line to center of all cells
    dist = _calculate_grid_cell_distances(
        inj_wells, nactive, calculation_type, grid, config
    )

    gas_results = _find_distances_per_time_step(
        "SGAS",
        calculation_type,
        threshold_gas,
        unrst,
        dist,
        inj_wells,
        plume_groups_gas,
    )

    if "AMFG" in unrst:
        dissolved_results = _find_distances_per_time_step(
            "AMFG",
            calculation_type,
            threshold_dissolved,
            unrst,
            dist,
            inj_wells,
            plume_groups_dissolved,
        )
        dissolved_prop_key = "AMFG"
    elif "XMF2" in unrst:
        dissolved_results = _find_distances_per_time_step(
            "XMF2",
            calculation_type,
            threshold_dissolved,
            unrst,
            dist,
            inj_wells,
            plume_groups_dissolved,
        )
        dissolved_prop_key = "XMF2"
    else:
        dissolved_results = None
        dissolved_prop_key = None
        logging.warning("WARNING: Neither AMFG nor XMF2 exists as properties.")

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
    grid = Grid(f"{case}.EGRID")
    unrst = ResdataFile(f"{case}.UNRST")

    if do_plume_tracking and injection_wells is not None:
        plume_groups_gas = calculate_plume_groups(
            "SGAS",
            threshold_gas,
            unrst,
            grid,
            injection_wells,
        )

        if "AMFG" in unrst:
            dissolved_prop_key = "AMFG"
        elif "XMF2" in unrst:
            dissolved_prop_key = "XMF2"
        else:
            dissolved_prop_key = None
        if dissolved_prop_key is not None:
            plume_groups_dissolved = calculate_plume_groups(
                dissolved_prop_key,
                threshold_dissolved,
                unrst,
                grid,
                injection_wells,
            )
        else:
            plume_groups_dissolved = None
    else:
        plume_groups_gas = None
        plume_groups_dissolved = None

    nactive = grid.get_num_active()
    logging.info(f"Number of active grid cells: {nactive}")

    all_results = []
    for i, single_config in enumerate(distance_calculations, 1):
        logging.info(f"\nCalculating distances for configuration number: {i}\n")
        (a, b, c) = calculate_single_distances(
            nactive,
            grid,
            unrst,
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
    unrst: ResdataFile,
    dist: Dict[str, np.ndarray],
    inj_wells: Optional[List[InjectionWellData]],
    plume_groups: Optional[List[List[str]]],
) -> dict:
    """
    Find value of distance metric for each step
    """
    do_plume_tracking = plume_groups is not None
    n_time_steps = len(unrst.report_steps)
    dist_per_group: Dict[str, Dict[str, np.ndarray]] = {}

    logging.info(f"\nStart calculating plume extent for {attribute_key}.\n")
    logging.info(f"Progress ({n_time_steps} time steps):")
    logging.info(f"{0:>6.1f} %")
    for i in range(n_time_steps):
        _find_distances_at_time_step(
            unrst,
            attribute_key,
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

    outputs = _organize_output_with_dates(
        dist_per_group,
        calculation_type,
        do_plume_tracking,
        inj_wells,
        unrst.report_dates,
    )

    logging.info(f"Done calculating plume extent for {attribute_key}.")
    return outputs


def _find_distances_at_time_step(
    unrst: ResdataFile,
    attribute_key: str,
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
    # NBNB-AS: Only needed if plume tracking is deactivated?
    #          If not, we have already done this
    data = unrst[attribute_key][i].numpy_view()
    cells_with_co2 = np.where(data > threshold)[0]

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
            if i == 0:
                dist_per_group["ALL"] = {}
                for well_name in dist.keys():
                    dist_per_group["ALL"][well_name] = np.zeros(shape=(n_time_steps,))
            for well_name in dist.keys():
                if len(cells_with_co2) > 0:
                    dist_per_group["ALL"][well_name][i] = dist[well_name][
                        cells_with_co2
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
            if i == 0:
                dist_per_group["ALL"] = {}
                for well_name in dist.keys():
                    dist_per_group["ALL"][well_name] = np.full(n_time_steps, np.nan)
            if len(cells_with_co2) > 0:
                dist_per_group["ALL"]["ALL"][i] = dist["ALL"][cells_with_co2].min()
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
        f"Number of dates {' ' * (col_width - 5)}: {len(dfs['date'].unique()):>11}"
    )
    logging.info(f"First date      {' ' * (col_width - 5)}: {dfs['date'].iloc[0]:>11}")
    logging.info(f"Last date       {' ' * (col_width - 5)}: {dfs['date'].iloc[-1]:>11}")

    for col in df.drop("date", axis=1).columns:
        logging.info(f"End state {col:<{col_width}} : {dfs[col].iloc[-1]:>11.1f}")


def _log_results_detailed(df: pd.DataFrame):
    dist_cols = [col for col in df.columns if col != "date"]
    letter_names = list(string.ascii_uppercase)[: len(dist_cols)]
    col_mapping = dict(zip(dist_cols, letter_names))
    col_mapping["date"] = "date"
    df = df.rename(columns=col_mapping)
    pd.options.display.float_format = "{:.1f}".format
    for col in df.columns:
        if col != "date":
            df[col] = df[col].round(1)

    logging.info("\nDetailed summary of results:")
    logging.info("============================")
    logging.info("Columns:")
    for key, value in col_mapping.items():
        if key != "date":
            logging.info(f"  {value}: {key}")

    def custom_format(x):
        if x == 0.0:
            return "-"
        else:
            return f"{x:.1f}"

    formatters = {
        col: custom_format if col != "date" else "{: >10}".format for col in df.columns
    }

    logging.info("\nResults:")
    logging.info(df.to_string(index=False, formatters=formatters))


def _find_dates(all_results: List[Tuple[dict, Optional[dict], Optional[str]]]):
    one_dict = all_results[0][0][next(iter(all_results[0][0]))]
    one_array = one_dict[next(iter(one_dict))]
    dates = [[date] for (date, _) in one_array]
    return dates


def _find_column_name(
    single_config: Calculation, n_calculations: int, calculation_number: int
):
    if single_config.type == CalculationType.PLUME_EXTENT:
        col = "MAX_"
    elif single_config.type in (CalculationType.POINT, CalculationType.LINE):
        col = "MIN_"
    else:
        col = "?"

    if single_config.column_name != "":
        col = col + single_config.column_name
    else:
        calc_number = "" if n_calculations == 1 else str(calculation_number)
        col = col + f"{single_config.type.name.upper()}{calc_number}"

    return col


def _collect_results_into_dataframe(
    all_results: List[Tuple[dict, Optional[dict], Optional[str]]],
    config: Configuration,
    injection_wells: Optional[List[InjectionWellData]] = None,
) -> pd.DataFrame:
    dates = _find_dates(all_results)
    df = pd.DataFrame.from_records(dates, columns=["date"])
    for i, (result, single_config) in enumerate(
        zip(all_results, config.distance_calculations), 1
    ):
        (gas_results, dissolved_results, _) = result

        col = _find_column_name(single_config, len(config.distance_calculations), i)

        if injection_wells is not None and config.do_plume_tracking:
            gas_results = sort_well_names(gas_results, injection_wells)

        for group_str, results in gas_results.items():
            for well_name, result2 in results.items():
                full_col_name = col + "_GAS"
                if group_str != "ALL":
                    full_col_name += "_PLUME_" + group_str
                if well_name != "ALL" and well_name != "WELL":
                    full_col_name += "_FROM_INJ_" + well_name
                gas_df = pd.DataFrame.from_records(
                    result2, columns=["date", full_col_name]
                )
                df = pd.merge(df, gas_df, on="date")
        if dissolved_results is not None:
            if injection_wells is not None and config.do_plume_tracking:
                dissolved_results_sorted = sort_well_names(
                    dissolved_results, injection_wells
                )
            else:
                dissolved_results_sorted = dissolved_results
            for group_str, results in dissolved_results_sorted.items():
                for well_name, result2 in results.items():
                    if result2 is not None:
                        full_col_name = col + "_DISSOLVED"
                        if group_str != "ALL":
                            full_col_name += "_PLUME_" + group_str
                        if well_name != "ALL" and well_name != "WELL":
                            full_col_name += "_FROM_INJ_" + well_name
                        dissolved_df = pd.DataFrame.from_records(
                            result2, columns=["date", full_col_name]
                        )
                        df = pd.merge(df, dissolved_df, on="date")
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
        "Invalid input: inj_point must be on the format [x,y]"
        "when calc_type is 'point'"
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
                    "Invalid input: inj_point must be on the format "
                    "[direction, value] when calc_type is 'line'."
                )
                logging.error(error)
                sys.exit(1)
    logging.error(
        "Invalid input: inj_point must be on the format "
        "[direction, value] when calc_type is 'line'"
    )
    sys.exit(1)


def main():
    """
    Calculate plume extent or distance to point/line using EGRID and
    UNRST-files. Calculated for SGAS and AMFG/XMF2. Output is distance per
    date written to a CSV file.
    """
    args = _make_parser().parse_args()
    _replace_default_dummies_from_ert(args)
    args.column_name = (
        args.column_name.upper() if args.column_name is not None else None
    )
    _setup_log_configuration(args)
    _log_input_configuration(args)

    config = Configuration(
        args.config_file,
        args.calc_type,
        args.inj_point,
        args.column_name,
        args.case,
    )
    _log_distance_calculation_configurations(config)

    all_results = calculate_distances(
        args.case,
        config.distance_calculations,
        config.injection_wells,
        config.do_plume_tracking,
        args.threshold_gas,
        args.threshold_dissolved,
    )

    df = _collect_results_into_dataframe(
        all_results,
        config,
        config.injection_wells,
    )
    _log_results(df)
    _log_results_detailed(df)

    output_file = _find_output_file(args.output_csv, args.case)
    logging.info("\nExport results to CSV file")
    logging.info(f"    - File path: {output_file}")
    if os.path.isfile(output_file):
        logging.info("Output CSV file already exists => Will overwrite existing file")
    df.to_csv(output_file, index=False, na_rep="0.0")

    return 0


if __name__ == "__main__":
    sys.exit(main())
