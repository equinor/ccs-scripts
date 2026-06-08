import argparse
import getpass
import logging
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import yaml

THRESHOLD_GAS = 1e-16
THRESHOLD_DISSOLVED = 1e-16  # Used also in co2_calculation to avoid numerical issues


def format_warning(txt: Union[str, Exception]) -> str:
    return f"\x1b[37;45m\x1b[1m{txt}\x1b[0m"


def format_error(txt: Union[str, Exception]) -> str:
    return f"\x1b[37;41m\x1b[1m{txt}\x1b[0m"


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "no", "0"}:
        return False
    elif value.lower() in {"true", "yes", "1"}:
        return True
    elif value == "-1":
        return "-1"
    raise ValueError(format_error(f"{value} is not a valid boolean value"))


def setup_log_configuration(arguments: argparse.Namespace) -> None:
    if arguments.debug:
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    elif arguments.no_logging:
        logging.basicConfig(format="%(message)s", level=logging.WARNING)
    else:
        logging.basicConfig(format="%(message)s", level=logging.INFO)


def log_input_banner(
    script: str,
    calculation: str,
    is_dev_version: bool = True,
    col_width: int = 24,
) -> None:
    version = "v0.16.0"  # NBNB: Fetch from version.py?
    if is_dev_version:
        version += "_dev"
        try:
            source_dir = os.path.dirname(os.path.abspath(script))
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
    calc_header = f"CCS-scripts - {calculation}"
    logging.info(calc_header)
    logging.info("=" * len(calc_header))
    logging.info(f"{'Version':<{col_width}} : {version}")
    logging.info(f"{'Date and time':<{col_width}} : {date_time}")
    logging.info(f"{'User':<{col_width}} : {getpass.getuser()}")
    logging.info(f"{'Host':<{col_width}} : {socket.gethostname()}")
    logging.info(
        f"{'Platform':<{col_width}} : {platform.system()} ({platform.release()})"
    )
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    logging.info(f"{'Python version':<{col_width}} : {py_version}")


def log_saturation_summaries(props: Dict) -> None:
    sgas = props["SGAS"]
    swat = props["SWAT"]
    soil = props["SOIL"] if "SOIL" in props else None

    first_timestep = next(iter(sgas))
    saturations_first_timestep = [
        ("sgas", sgas[first_timestep]),
        ("swat", swat[first_timestep]),
    ]
    last_timestep = next(reversed(sgas))
    saturations_last_timestep = [
        ("sgas", sgas[last_timestep]),
        ("swat", swat[last_timestep]),
    ]

    if soil is not None:
        saturations_first_timestep.append(("soil", soil[first_timestep]))
        saturations_last_timestep.append(("soil", soil[last_timestep]))

    header = (
        f"\n{'Property':<15} {'Min':>12} {'P10':>12} "
        f"{'Median':>12} {'Mean':>12} {'P90':>12} {'Max':>12}"
    )
    logging.info("\nPhase saturation summaries for first timestep - Active cells only")
    logging.info(header)
    logging.info(f"{'-' * 93}")
    for label, values in saturations_first_timestep:
        row = (
            f"{label:<15} "
            f"{values.min():>12.1f} "
            f"{np.percentile(values, 10):>12.1f} "
            f"{np.median(values):>12.1f} "
            f"{values.mean():>12.1f} "
            f"{np.percentile(values, 90):>12.1f} "
            f"{values.max():>12.1f}"
        )
        logging.info(row)

    logging.info("\nPhase saturation summaries for last timestep - Active cells only")
    logging.info(header)
    logging.info(f"{'-' * 93}")

    for label, values in saturations_last_timestep:
        row = (
            f"{label:<15} "
            f"{values.min():>12.1f} "
            f"{np.percentile(values, 10):>12.1f} "
            f"{np.median(values):>12.1f} "
            f"{values.mean():>12.1f} "
            f"{np.percentile(values, 90):>12.1f} "
            f"{values.max():>12.1f}"
        )
        logging.info(row)


def identify_gas_less_cells(
    sgas: dict, dissolved_prop: Optional[dict] = None
) -> np.ndarray:
    """
    Identifies those cells that do not have gas. This is done based on thresholds for
    SGAS and AMFG/XMF2 (dissolved property).

    Args:
      sgas (dict): The values of SGAS for each grid cell
      dissolved_prop (dict): The values of AMFG or XMF2 for each grid cell

    Returns:
      np.ndarray

    """
    return identify_gas_less_cells_from_iterator(
        sgas.values(),
        dissolved_prop.values() if dissolved_prop is not None else None,
    )


def identify_gas_less_cells_from_iterator(
    sgas_iter: Iterable[np.ndarray], dissolved_iter: Iterable[np.ndarray] | None
) -> np.ndarray:
    gas_less = np.logical_and.reduce([np.abs(s) < THRESHOLD_GAS for s in sgas_iter])
    if dissolved_iter is not None:
        gas_less &= np.logical_and.reduce(
            [np.abs(d) < THRESHOLD_DISSOLVED for d in dissolved_iter]
        )
    return gas_less


def reduce_properties(
    properties: Dict[str, Dict[str, np.ndarray]], keep_idx: np.ndarray
) -> Dict:
    """
    Reduces the data of given properties by indices in keep_idx

    Args:
      properties (Dict): Data with values of properties
      keep_idx (np.ndarray): Which indices are retained

    Returns:
      Dict

    """
    return {
        p: {d: properties[p][d][keep_idx] for d in properties[p]} for p in properties
    }


def is_subset(first: List[str], second: List[str]) -> bool:
    """
    Determines if the elements of a list (first) are part of
    another list (second)

    Args:
      first (List): The list whose elements are searched in second
      second (List): The list where elements of first are searched

    Returns:
      bool

    """
    return all(x in second for x in first)


def read_yaml_file(
    file_name: str,
) -> Dict:
    with open(file_name, "r", encoding="utf8") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            logging.error(format_error(exc))
            sys.exit(1)
