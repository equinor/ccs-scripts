"""CLI and runtime setup for CO2 plume tracking calculations."""

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

from ccs_scripts.co2_plume_tracking.utils import (
    DEFAULT_THRESHOLD_DISSOLVED,
    DEFAULT_THRESHOLD_GAS,
    Configuration,
)
from ccs_scripts.utils.utils import setup_log_configuration


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
        help="Threshold for aqueous mole fraction of gas (AMFS, AMFG or XMF2)",
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


def _log_input_configuration(arguments: argparse.Namespace) -> None:
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


def process_input() -> Tuple[argparse.Namespace, Configuration]:
    args = _make_parser().parse_args()
    setup_log_configuration(args)
    _log_input_configuration(args)

    config = Configuration(
        args.config_file,
    )
    _log_configuration(config)
    return args, config
