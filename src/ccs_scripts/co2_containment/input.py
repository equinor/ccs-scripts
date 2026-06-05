"""CLI parsing, validation, and logging setup for CO2 containment."""

import argparse
import getpass
import logging
import os
import pathlib
import platform
import socket
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely.geometry
import yaml

from ccs_scripts.co2_plume_tracking.co2_plume_tracking import (
    DEFAULT_THRESHOLD_DISSOLVED,
    Configuration,
    calculate_plume_groups,
    load_plume_tracking_data,
)
from ccs_scripts.co2_plume_tracking.utils import InjectionWellData
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import (
    format_error,
    format_warning,
    str_to_bool,
)


@dataclass
class ZoneInfo:
    """
    Dataclass holding information about zones.
    """
    source: Optional[str]
    zranges: Optional[Dict[str, List[int]]]
    int_to_zone: Optional[List[Optional[str]]]


@dataclass
class RegionInfo:
    """
    Dataclass holding information about regions.
    """
    source: Optional[str]
    int_to_region: Optional[List[Optional[str]]]
    property_name: Optional[str]


class CalculationType(Enum):
    """
    Which type of CO2 calculation is made
    """

    MASS = 0
    CELL_VOLUME = 1
    ACTUAL_VOLUME = 2

    @classmethod
    def check_for_key(cls, key: str):
        """
        Check if key in enum
        """
        if key not in cls.__members__:
            error_text = "Illegal calculation type: " + key
            error_text += "\nValid options:"
            for calc_type in CalculationType:
                error_text += "\n  * " + calc_type.name.lower()
            error_text += "\nExiting"
            raise ValueError(format_error(error_text))


def process_input() -> Tuple[
    argparse.Namespace,
    ZoneInfo,
    RegionInfo,
    CalculationType,
    Optional[shapely.geometry.Polygon],
    Optional[shapely.geometry.Polygon],
    Optional[List[List[str]]],
]:
    """
    Process input arguments, check that they are valid, and log the provided
    input.

    Returns the processed arguments, and additional data used for CO2 and containment calculations.
    """
    args = _process_args()
    _check_input(args)

    zone_info = ZoneInfo(
        source=args.zonefile,
        zranges=None,
        int_to_zone=None,
    )
    region_info = RegionInfo(
        source=args.regionfile,
        int_to_region=None,  # set during calculation if source or property is given
        property_name=args.region_property,
    )
    if zone_info.source is not None:
        zone_info.zranges = _process_zonefile_if_yaml(zone_info.source)

    calc_type = set_calc_type_from_input_string(args.calc_type_input)

    _log_input_configuration(args)

    if args.config_plume_tracking == "":
        plume_groups = None
    else:
        config = Configuration(args.config_plume_tracking)
        injection_wells = config.injection_wells
        plume_groups = _find_plume_groups(args.egrid, args.unrst, injection_wells)

    cont_polygon = _read_polygon(args.containment_polygon)
    nogo_polygon = _read_polygon(args.nogo_polygon)

    return (
        args,
        zone_info,
        region_info,
        calc_type,
        cont_polygon,
        nogo_polygon,
        plume_groups,
    )


def _get_parser() -> argparse.ArgumentParser:
    """
    Make parser and define arguments

    Returns:
        argparse.ArgumentParser
    """
    path_name = pathlib.Path(__file__).name
    parser = argparse.ArgumentParser(path_name)
    parser.add_argument(
        "case",
        help="Path to Eclipse case (EGRID, INIT and UNRST files), including base name,\
        but excluding the file extension (.EGRID, .INIT, .UNRST)",
        metavar="<CASE>",
    )
    parser.add_argument(
        "calc_type_input",
        help="CO2 calculation options: mass / cell_volume / actual_volume. "
        "Mass is calculated in tons, volume in cubic metres.",
        metavar="<CALC_TYPE_INPUT>",
    )
    parser.add_argument(
        "--root_dir",
        help="Path to root directory. The other paths can be provided relative \
        to this or as absolute paths. Default is 2 levels up from Eclipse case.",
        default=None,
        metavar="<ROOT_DIR>",
    )
    parser.add_argument(
        "--out_dir",
        help="Path to output directory (file name is set to \
        'plume_<calculation type>.csv'). \
        Defaults to <root_dir>/share/results/tables.",
        default=None,
        metavar="<OUT_DIR>",
    )
    parser.add_argument(
        "--containment_polygon",
        help="Path to polygon that determines the bounds of the containment area. \
        Count all CO2 as contained if polygon is not provided.",
        default=None,
        metavar="<CONTAINMENT_POLYGON>",
    )
    parser.add_argument(
        "--nogo_polygon",
        help="Path to polygon that determines the bounds of the no-go area.",
        default=None,
        metavar="<NOGO_POLYGON>",
    )
    parser.add_argument(
        "--hazardous_polygon",
        help="Deprecated: use --nogo_polygon instead.",
        default=None,
        metavar="<HAZARDOUS_POLYGON>",
    )
    parser.add_argument(
        "--egrid",
        help="Path to EGRID file. Overwrites <case> if provided.",
        default=None,
        metavar="<EGRID>",
    )
    parser.add_argument(
        "--unrst",
        help="Path to UNRST file. Overwrites <case> if provided.",
        default=None,
        metavar="<UNRST>",
    )
    parser.add_argument(
        "--init",
        help="Path to INIT file. Overwrites <case> if provided.",
        default=None,
        metavar="<INIT>",
    )
    parser.add_argument(
        "--zonefile",
        help="Path to yaml or roff file containing zone information.",
        default=None,
        metavar="<ZONEFILE>",
    )
    parser.add_argument(
        "--regionfile",
        help="Path to roff file containing region information. "
        "Use either 'regionfile' or 'region_property', not both.",
        default=None,
        metavar="<REGIONFILE>",
    )
    parser.add_argument(
        "--region_property",
        help="Property in INIT file containing integer grid of regions. "
        "Use either 'regionfile' or 'region_property', not both.",
        default=None,
        metavar="<REGION_PROPERTY>",
    )
    parser.add_argument(
        "--no_logging",
        help="Skip print of detailed information during execution of script",
        type=str_to_bool,
        nargs="?",
        const=True,
        metavar="<NO_LOGGING>",
    )
    parser.add_argument(
        "--debug",
        help="Enable print of debugging data during execution of script. "
        "Normally not necessary for most users.",
        type=str_to_bool,
        nargs="?",
        const=True,
        metavar="<DEBUG>",
    )
    parser.add_argument(
        "--residual_trapping",
        help="Compute mass/volume of trapped CO2 in gas phase.",
        type=str_to_bool,
        nargs="?",
        const=True,
        metavar="<RESIDUAL_TRAPPING>",
    )
    parser.add_argument(
        "--readable_output",
        help="Generate output text-file that is easier to parse than the standard"
        " output.",
        type=str_to_bool,
        nargs="?",
        const=True,
        metavar="<READABLE_OUTPUT>",
    )
    parser.add_argument(
        "--config_plume_tracking",
        help="YML file with configurations for plume tracking calculations.",
        default="",
        metavar="<CONFIG_PLUME_TRACKING>",
    )
    parser.add_argument(
        "--cirrus_info_file",
        help="Path to Cirrus info file. Relevant for COMP3/4",
        default=None,
        metavar="<CIRRUS_INFO_FILE>",
    )

    return parser


def _handle_deprecated_args(args):
    if args.hazardous_polygon is not None:
        warning_text = (
            "'--hazardous_polygon' / '<HAZARDOUS_POLYGON>' is deprecated and "
            "will be removed in a future "
            "release.\nPlease use '--nogo_polygon' / '<NOGO_POLYGON>' instead."
        )
        logging.warning(format_warning(warning_text))
        warnings.warn(warning_text, DeprecationWarning)
        if args.nogo_polygon is None:
            args.nogo_polygon = args.hazardous_polygon


def _replace_default_dummies_from_ert(args):
    if args.root_dir == "-1":
        args.root_dir = None
    if args.egrid == "-1":
        args.egrid = None
    if args.unrst == "-1":
        args.unrst = None
    if args.init == "-1":
        args.init = None
    if args.out_dir == "-1":
        args.out_dir = None
    if args.zonefile == "-1":
        args.zonefile = None
    if args.regionfile == "-1":
        args.regionfile = None
    if args.region_property == "-1":
        args.region_property = None
    if args.containment_polygon == "-1":
        args.containment_polygon = None
    if args.nogo_polygon == "-1":
        args.nogo_polygon = None
    if args.hazardous_polygon == "-1":
        args.hazardous_polygon = None
    if args.no_logging == "-1":
        args.no_logging = False
    if args.debug == "-1":
        args.debug = False
    if args.residual_trapping == "-1":
        args.residual_trapping = False
    if args.readable_output == "-1":
        args.readable_output = False
    if args.cirrus_info_file == "-1":
        args.cirrus_info_file = None


class InputError(Exception):
    """Raised for various mistakes in the provided input."""


# pylint: disable-msg=too-many-branches
def _process_args() -> argparse.Namespace:
    """
    Process arguments and do some minor conversions.
    Create absolute paths if relative paths are provided.

    Returns:
        argparse.Namespace
    """
    args = _get_parser().parse_args()

    if args.debug:
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    elif args.no_logging:
        logging.basicConfig(format="%(message)s", level=logging.WARNING)
    else:
        logging.basicConfig(format="%(message)s", level=logging.INFO)

    _replace_default_dummies_from_ert(args)

    _handle_deprecated_args(args)

    args.calc_type_input = args.calc_type_input.lower()
    if args.residual_trapping and args.calc_type_input == "cell_volume":
        args.residual_trapping = False

    if args.root_dir is None:
        p = pathlib.Path(args.case).parents
        if len(p) < 3:
            error_text = "Invalid input, <case> must have at least two parent levels \
            if <root_dir> is not provided."
            raise InputError(format_error(error_text))
        args.root_dir = p[2]
    adict = vars(args)
    paths = [
        "case",
        "out_dir",
        "egrid",
        "unrst",
        "init",
        "zonefile",
        "regionfile",
        "containment_polygon",
        "nogo_polygon",
        "cirrus_info_file",
    ]
    for key in paths:
        if adict[key] is not None and not pathlib.Path(adict[key]).is_absolute():
            adict[key] = os.path.join(args.root_dir, adict[key])
    if args.out_dir is None:
        args.out_dir = os.path.join(args.root_dir, "share", "results", "tables")

    if args.egrid is None:
        args.egrid = args.case
        if not args.egrid.endswith(".EGRID"):
            args.egrid += ".EGRID"
    if args.unrst is None:
        args.unrst = args.case
        if args.unrst.endswith(".EGRID"):
            args.unrst = args.unrst.replace(".EGRID", ".UNRST")
        else:
            args.unrst += ".UNRST"
    if args.init is None:
        args.init = args.case
        if args.init.endswith(".EGRID"):
            args.init = args.init.replace(".EGRID", ".INIT")
        else:
            args.init += ".INIT"
    if args.cirrus_info_file is None:
        args.cirrus_info_file = args.case
        if args.cirrus_info_file.endswith(".EGRID"):
            args.cirrus_info_file = args.cirrus_info_file.replace(".EGRID", "_INFO.CSV")
        else:
            args.cirrus_info_file += "_INFO.CSV"
    return args


def _check_input(arguments: argparse.Namespace):
    """
    Checks that input arguments are valid. Checks if files exist etc.

    Args:
        arguments (argparse.Namespace): Input arguments

    Raises:
        ValueError: If calc_type_input is invalid
        FileNotFoundError: If one or more input files are not found
    """
    CalculationType.check_for_key(arguments.calc_type_input.upper())

    files_not_found = []
    if not os.path.isfile(arguments.egrid):
        files_not_found.append(arguments.egrid)
    if not os.path.isfile(arguments.unrst):
        files_not_found.append(arguments.unrst)
    if arguments.zonefile is not None and not os.path.isfile(arguments.zonefile):
        files_not_found.append(arguments.zonefile)
    if arguments.regionfile is not None and not os.path.isfile(arguments.regionfile):
        files_not_found.append(arguments.regionfile)
    if arguments.containment_polygon is not None and not os.path.isfile(
        arguments.containment_polygon
    ):
        files_not_found.append(arguments.containment_polygon)
    if arguments.nogo_polygon is not None and not os.path.isfile(
        arguments.nogo_polygon
    ):
        files_not_found.append(arguments.nogo_polygon)
    if files_not_found:
        error_text = "The following file(s) were not found:"
        for file in files_not_found:
            error_text += "\n  * " + file
        raise FileNotFoundError(format_error(error_text))

    if arguments.regionfile is not None and arguments.region_property is not None:
        error_text = (
            "Both 'regionfile' and 'region_property' have been provided. "
            "Please provide only one of the two options."
        )
        raise InputError(format_error(error_text))

    if not os.path.isdir(arguments.out_dir):
        warning_text = "Output directory doesn't exist. Creating a new folder."
        logging.warning(format_warning(warning_text))
        os.mkdir(arguments.out_dir)

    if not os.path.isfile(arguments.init):
        logging.info(f"The INIT-file {arguments.init} was not found\n")


def set_calc_type_from_input_string(calc_type_input: str) -> CalculationType:
    """
    Creates a CalculationType object from an input string

    Args:
      calc_type_input (str): Input string with calculation type to perform

    Returns:
      CalculationType

    """
    calc_type_input = calc_type_input.upper()
    CalculationType.check_for_key(calc_type_input)
    return CalculationType[calc_type_input]


def _process_zonefile_if_yaml(zonefile: str) -> Optional[Dict[str, List[int]]]:
    """
    Processes zone_file if it is provided as a yaml file, ex:
    zranges:
        - Zone1: [1, 5]
        - Zone2: [6, 10]
        - Zone3: [11, 14]

    Returns:
        Dictionary connecting names of zones to their layers:
    {
        "Zone1": [1,5]
        "Zone2": [6,10]
        "Zone3": [11,14]
    }
    """
    if zonefile.split(".")[-1].lower() in ["yml", "yaml"]:
        with open(zonefile, "r", encoding="utf8") as stream:
            try:
                zfile = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logging.error(format_error(exc))
                sys.exit(1)
        if "zranges" not in zfile:
            error_text = "The yaml zone file must be in the format:\nzranges:\
            \n    - Zone1: [1, 5]\n    - Zone2: [6, 10]\n    - Zone3: [11, 14])"
            raise InputError(format_error(error_text))
        zranges = zfile["zranges"]
        if len(zranges) > 1:
            zranges_ = zranges[0]
            for zr in zranges[1:]:
                zranges_.update(zr)
            zranges = zranges_
        return zranges
    return None


def _log_input_configuration(args: argparse.Namespace) -> None:
    """
    Log the provided input
    """
    version = "v0.16.0"
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

    col1 = 24
    now = datetime.now()
    date_time = now.strftime("%B %d, %Y %H:%M:%S")
    logging.info("CCS-scripts - Containment calculations")
    logging.info("======================================")
    logging.info(f"{'Version':<{col1}} : {version}")
    logging.info(f"{'Date and time':<{col1}} : {date_time}")
    logging.info(f"{'User':<{col1}} : {getpass.getuser()}")
    logging.info(f"{'Host':<{col1}} : {socket.gethostname()}")
    logging.info(f"{'Platform':<{col1}} : {platform.system()} ({platform.release()})")
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    logging.info(f"{'Python version':<{col1}} : {py_version}")

    logging.info(f"\n{'Case':<{col1}} : {args.case}")
    if not os.path.isabs(args.case):
        logging.info(
            f"{'  => Absolute path':<{col1}} : " f"{os.path.abspath(args.case)}"
        )
    logging.info(f"{'Calculation type':<{col1}} : {args.calc_type_input}")
    unit_str = "tons" if args.calc_type_input == "mass" else "cubic metres"
    logging.info(f"{'Unit':<{col1}} : {unit_str}")
    logging.info(f"{'Root directory':<{col1}} : {args.root_dir}")
    logging.info(f"{'Output directory':<{col1}} : {args.out_dir}")
    logging.info(f"{'Containment polygon':<{col1}} : {args.containment_polygon}")
    logging.info(f"{'No-go polygon':<{col1}} : {args.nogo_polygon}")
    logging.info(f"{'EGRID file':<{col1}} : {args.egrid}")
    logging.info(f"{'UNRST file':<{col1}} : {args.unrst}")
    logging.info(f"{'INIT file':<{col1}} : {args.init}")
    logging.info(f"{'Zone file':<{col1}} : {args.zonefile}")
    regionfile_str = args.regionfile if args.regionfile is not None else "-"
    logging.info(f"{'Region file':<{col1}} : " f"{regionfile_str}")
    region_property_str = (
        args.region_property if args.region_property is not None else "-"
    )
    logging.info(f"{'Region property':<{col1}} : " f"{region_property_str}")
    logging.info(
        f"{'Residual trapping':<{col1}} : "
        f"{'yes' if args.residual_trapping else 'no'}"
    )
    readable_output_str = (
        "yes" if args.readable_output is not None and args.readable_output else "no"
    )
    logging.info(f"{'Readable output':<{col1}} : " f"{readable_output_str}")
    config_plume_tracking_str = (
        args.config_plume_tracking if args.config_plume_tracking != "" else "-"
    )
    logging.info(
        f"{'Plume tracking YAML-file':<{col1}} : " f"{config_plume_tracking_str}\n"
    )


def _find_plume_groups(
    grid_file: str,
    unrst_file: str,
    injection_wells: List[InjectionWellData],
) -> Optional[List[List[str]]]:
    if len(injection_wells) == 0:
        return None
    grid_data, properties, dates, gasless = load_plume_tracking_data(
        grid_file, unrst_file
    )

    dissolved_prop = next(
        (p for p in ("AMFS", "AMFG", "XMF2") if p in properties), None
    )
    if dissolved_prop is None:
        return None

    plume_groups, _ = calculate_plume_groups(
        attribute_key=dissolved_prop,
        threshold=0.1 * DEFAULT_THRESHOLD_DISSOLVED,
        grid_data=grid_data,
        properties=properties,
        dates=dates,
        inj_wells=injection_wells,
        gasless=gasless,
    )
    return plume_groups


def _read_polygon(polygon_file: Optional[str]) -> Optional[shapely.geometry.Polygon]:
    """
    Reads a polygon from file.

    Args:
        polygon_file (str): Path to polygon file

    Returns:
        shapely.geometry.Polygon
    """
    if not polygon_file:
        return None
    poly_xy = np.genfromtxt(polygon_file, skip_header=1, delimiter=",")[:, :2]
    return shapely.geometry.Polygon(poly_xy)


def init_timer():
    timer = Timer()
    timer.reset_timings()
    timer.code_parts = {
        "extract_source_data": "Extract source data",
        "calculate_co2": "Calculate CO2 per grid cell from source data",
        "plume_tracking": "Plume tracking",
        "plume_tracking_represent_as_property": "Represent as property",
        "plume_tracking_init_groups": "Initialize groups from previous step",
        "plume_tracking_resolve_undetermined": "Resolve undetermined cells",
        "plume_tracking_find_unique_groups": "Find unique groups",
        "plume_tracking_logging": "Various logging",
        "conversion_active_to_gasless_cells": "Convert active to gasless cells",
        "calculate_co2_containment": "Calculate CO2 containment",
        "make_location_filters": "Make location filters for polygons",
        "plume_group_mapping": "Map plume groups",
        "sum_and_store": "Sum and store amount of CO2",
        "export_results": "Export results",
        "logging": "Various logging",
    }


# pylint: disable = too-many-statements
