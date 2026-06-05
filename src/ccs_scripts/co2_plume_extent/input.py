"""CLI and runtime setup for CO2 plume extent calculations."""

import argparse
import getpass
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from typing import Tuple

from ccs_scripts.co2_plume_extent.config import (
    DEFAULT_THRESHOLD_DISSOLVED,
    DEFAULT_THRESHOLD_GAS,
    Configuration,
)
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import setup_log_configuration, str_to_bool


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate plume extent (distance)")
    parser.add_argument("case", help="Name of Eclipse case", metavar="<CASE>")
    parser.add_argument(
        "--config_plume_extent",
        help="YML file with configurations for distance calculations.",
        default="",
        metavar="<CONFIG_PLUME_EXTENT>",
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
        metavar="<INJ_POINT>",
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
        metavar="<CALC_TYPE>",
    )
    parser.add_argument(
        "--output_csv",
        help="Path to output CSV file",
        default=None,
        metavar="<OUTPUT_CSV>",
    )
    parser.add_argument(
        "--threshold_gas",
        default=DEFAULT_THRESHOLD_GAS,
        type=float,
        help="Threshold for gas saturation (SGAS)",
        metavar="<THRESHOLD_GAS>",
    )
    parser.add_argument(
        "--threshold_dissolved",
        default=DEFAULT_THRESHOLD_DISSOLVED,
        type=float,
        help="Threshold for aqueous mole fraction of gas (AMFG or XMF2)",
        metavar="<THRESHOLD_DISSOLVED>",
    )
    parser.add_argument(
        "--column_name",
        default="",
        type=str,
        help="Name that will be included in the column of the CSV file",
        metavar="<COLUMN_NAME>",
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

    return parser


def _replace_default_dummies_from_ert(args):
    if args.no_logging == "-1":
        args.no_logging = False
    if args.debug == "-1":
        args.debug = False


def _log_input_configuration(args: argparse.Namespace) -> None:
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

    logging.info(f"\nCase                    : {args.case}")
    if not os.path.isabs(args.case):
        logging.info(f"  => Absolute path      : {os.path.abspath(args.case)}")
    if args.config_plume_extent == "":
        config_str = "Not specified"
    else:
        config_str = args.config_plume_extent
    logging.info(f"Configuration YAML-file : {config_str}")
    if args.inj_point != "":
        logging.info("Configuration from args :")
        logging.info(f"    Injection point info: {args.inj_point}")
        logging.info(f"    Calculation type    : {args.calc_type}")
        col = args.column_name
        if col != "":
            logging.info(
                f"    Column name         : " f"{col if col != '' else 'Not specified'}"
            )
    else:
        logging.info("Configuration from args : Not specified")
    if args.output_csv is None or args.output_csv == "":
        text = "Not specified, using default"
    else:
        text = args.output_csv
    logging.info(f"Output CSV file         : {text}")
    logging.info(f"Threshold gas           : {args.threshold_gas}")
    logging.info(f"Threshold dissolved     : {args.threshold_dissolved}\n")


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


def init_timer() -> None:
    timer = Timer()
    timer.reset_timings()
    timer.code_parts = {
        "plume_tracking": "Plume tracking",
        "plume_tracking_represent_as_property": "Represent as property",
        "plume_tracking_init_groups": "Initialize groups from previous step",
        "plume_tracking_resolve_undetermined": "Resolve undetermined cells",
        "plume_tracking_find_unique_groups": "Find unique groups",
        "plume_tracking_logging": "Various logging",
        "calculate_grid_cell_distances": "Precompute grid cells distances",
        "find_distances": "Find distances",
        "export_results": "Export results",
        "logging": "Various logging",
    }


def process_input() -> Tuple[argparse.Namespace, Configuration]:
    args = _make_parser().parse_args()
    _replace_default_dummies_from_ert(args)
    args.column_name = (
        args.column_name.upper() if args.column_name is not None else None
    )
    setup_log_configuration(args)
    _log_input_configuration(args)

    config = Configuration(
        args.config_plume_extent,
        args.calc_type,
        args.inj_point,
        args.column_name,
        args.case,
    )
    _log_distance_calculation_configurations(config)
    return args, config
