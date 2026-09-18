"""Configuration and validation for CO2 plume extent calculations."""

import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ccs_scripts.co2_plume_tracking.utils import InjectionWellData
from ccs_scripts.utils.utils import format_error, format_warning, read_yaml_file

DEFAULT_THRESHOLD_GAS = 0.2
DEFAULT_THRESHOLD_DISSOLVED = 0.0005
INJ_POINT_THRESHOLD = 60.0


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
            raise ValueError(format_error(error_text))


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
            raise ValueError(format_error(error_text))


@dataclass
class Calculation:
    type: CalculationType
    direction: Optional[LineDirection]
    column_name: str
    x: Optional[float]
    y: Optional[float]


def calculate_well_coordinates(
    case: str,
    injection_point_info: str,
    well_picks_path: Optional[str] = None,
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
                error_text = (
                    "Invalid input: When providing two arguments (x and y coordinates)"
                    " for injection point info they need to be floats."
                )
                logging.error(format_error(error_text))
                sys.exit(1)
    well_name = injection_point_info
    return _calculate_well_coordinates(case, well_name, well_picks_path)


def _calculate_well_coordinates(
    case: str, well_name: str, well_picks_path: Optional[str] = None
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
        error_text = (
            f"No matches for well name {well_name}, input is either mistyped "
            "or well does not exist."
        )
        logging.error(format_error(error_text))
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


def find_input_point(injection_point_info: str) -> Tuple[float, float]:
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
                error_text = (
                    "Invalid input: When providing two arguments (x and y coordinates) "
                    "for point they need to be floats."
                )
                logging.error(format_error(error_text))
                sys.exit(1)
    error_text = (
        "Invalid input: inj_point must be on the format [x,y]"
        "when calc_type is 'point'"
    )
    logging.error(format_error(error_text))
    sys.exit(1)


def find_input_line(injection_point_info: str) -> Tuple[str, float]:
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
                    error_text = (
                        "Invalid line direction. Choose from "
                        "'east'/'west'/'north'/'south'"
                    )
                    raise ValueError(format_error(error_text))
                value = float(coords[1])
                coordinates = (direction, value)
                logging.info(f"Using line data: [{direction}, {value}]")
                return coordinates
            except ValueError as error:
                error_text = (
                    "Invalid input: inj_point must be on the format "
                    "[direction, value] when calc_type is 'line'."
                )
                logging.error(format_error(error_text))
                logging.error(format_error(error))
                sys.exit(1)
    error_text = (
        "Invalid input: inj_point must be on the format "
        "[direction, value] when calc_type is 'line'"
    )
    logging.error(format_error(error_text))
    sys.exit(1)


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
            input_dict = read_yaml_file(config_file)
            self.make_config_from_input_dict(input_dict, case)
        if injection_point_info != "":
            self.make_config_from_input_args(
                calculation_type, injection_point_info, column_name, case
            )

        if len(self.distance_calculations) == 0:
            warning_text = (
                "WARNING: No CO2 plume distance/extent calculations"
                " specified in the input. Terminating script"
            )
            logging.warning(format_warning(warning_text))
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
            warning_text = (
                "\nWARNING: Plume tracking activated, but no injection_wells specified."
                "\n         Plume tracking will therefore be switched off."
            )
            logging.warning(format_warning(warning_text))
            self.do_plume_tracking = False
        if "injection_wells" in input_dict:
            for i, injection_well_info in enumerate(input_dict["injection_wells"], 1):
                args_required = ["name", "x", "y"]
                for arg in args_required:
                    if arg not in injection_well_info:
                        error_text = (
                            f'\nERROR: Missing "{arg}" under "injection_wells" '
                            f"for injection well number {i}."
                        )
                        logging.error(format_error(error_text))
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
            error_text = (
                '\nERROR: No instance of "distance_calculations" in input YAML file.'
            )
            logging.error(format_error(error_text))
            sys.exit(1)
        if not isinstance(input_dict["distance_calculations"], list):
            error_text = (
                '\nERROR: Specification under "distance_calculations" in '
                "input YAML file is not a list."
            )
            logging.error(format_error(error_text))
            sys.exit(1)
        for i, single_calculation in enumerate(input_dict["distance_calculations"], 1):
            if "type" not in single_calculation:
                error_text = (
                    f'\nERROR: Missing "type" for distance calculation number {i}.'
                )
                logging.error(format_error(error_text))
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
                    error_text = (
                        f'\nERROR: Missing "direction" for distance '
                        f'calculation number {i}. Needed when "type" = "line".'
                    )
                    logging.error(format_error(error_text))
                    sys.exit(1)
                else:
                    direction_str = single_calculation["direction"].upper()
                    LineDirection.check_for_key(direction_str)
                    direction = LineDirection[direction_str]
            else:
                if "direction" in single_calculation:
                    warning_text = (
                        '\nWARNING: No need to specify "direction" when "type" is not'
                        f' "line" (distance calculation number {i}).'
                    )
                    logging.warning(format_warning(warning_text))

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
                    error_text = (
                        f'\nERROR: Missing "x" for distance calculation number {i}.'
                    )
                    logging.error(format_error(error_text))
                    sys.exit(1)
                if y is None:
                    error_text = (
                        f'\nERROR: Missing "y" for distance calculation number {i}.'
                    )
                    logging.error(format_error(error_text))
                    sys.exit(1)
            elif calculation_type == CalculationType.LINE:
                if direction in (LineDirection.EAST, LineDirection.WEST):
                    if x is None:
                        error_text = (
                            f'\nERROR: Missing "x" for distance calculation number {i}.'
                        )
                        logging.error(format_error(error_text))
                        sys.exit(1)
                    if y is not None:
                        warning_text = (
                            '\nWARNING: No need to specify "y" for distance '
                            f"calculation number {i}."
                        )
                        logging.warning(format_warning(warning_text))
                elif direction in (LineDirection.NORTH, LineDirection.SOUTH):
                    if y is None:
                        logging.error(
                            f'\nERROR: Missing "y" for distance calculation number {i}.'
                        )
                        sys.exit(1)
                    if x is not None:
                        warning_text = (
                            '\nWARNING: No need to specify "x" for distance '
                            f"calculation number {i}."
                        )
                        logging.warning(format_warning(warning_text))

            if well_name is not None:
                x, y = _calculate_well_coordinates(case, well_name)

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
                    error_text = (
                        "ERROR: Invalid input. inj_point must be on"
                        ' the format "[x,y]" or "well_name" when '
                        "calc_type is 'plume_extent'."
                    )
                    logging.error(format_error(error_text))
                elif calculation_type == CalculationType.POINT:
                    error_text = (
                        "ERROR: Invalid input. inj_point must be on"
                        ' the format "[x,y]" when calc_type is '
                        "'point'."
                    )
                    logging.error(format_error(error_text))
                elif calculation_type == CalculationType.LINE:
                    error_text = (
                        "Invalid input: inj_point must be on the "
                        'format "[direction, x or y]" when '
                        "calc_type is 'line'."
                    )
                    logging.error(format_error(error_text))
                sys.exit(1)

            if calculation_type in (
                CalculationType.PLUME_EXTENT,
                CalculationType.POINT,
            ):
                try:
                    x, y = (float(values[0]), float(values[1]))
                    logging.info(f"Using injection coordinates: [{x}, {y}]")
                except ValueError:
                    error_text = (
                        "ERROR: Invalid input. When providing two arguments "
                        "(x and y coordinates) for injection point info they "
                        "need to be floats."
                    )
                    logging.error(format_error(error_text))
                    sys.exit(1)
            elif calculation_type == CalculationType.LINE:
                try:
                    direction_str, coord = (str(values[0]), float(values[1]))
                    logging.info(f"Using injection info: [{direction_str}, {coord}]")
                except ValueError:
                    error_text = (
                        "ERROR: Invalid input. When providing two arguments "
                        "(direction and x or y) for injection point, the "
                        "direction needs to be a string and the coordinate "
                        "needs to be a float."
                    )
                    logging.error(format_error(error_text))
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
                error_text = (
                    "ERROR: Invalid input. For plume_extent, the injection "
                    f'point info specified ("{injection_point_info}") is '
                    'incorrect. It should be on the format "[x,y]" or '
                    '"well_name".'
                )
                logging.error(format_error(error_text))
                sys.exit(1)

            x, y = _calculate_well_coordinates(case, injection_point_info)

        calculation = Calculation(
            type=calculation_type,
            direction=direction,
            column_name=column_name,
            x=x,
            y=y,
        )
        self.distance_calculations.append(calculation)
