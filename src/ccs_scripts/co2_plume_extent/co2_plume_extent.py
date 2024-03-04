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
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from resdata.grid import Grid
from resdata.resfile import ResdataFile

DEFAULT_THRESHOLD_SGAS = 0.2
DEFAULT_THRESHOLD_AMFG = 0.0005


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate plume extent (distance)")
    parser.add_argument("case", help="Name of Eclipse case")
    parser.add_argument(
        "injection_point_info",
        help="Either the name of the injection well (string) or \
        the x and y coordinates (two floats) to calculate plume extent from "
        "(format: [x,y]).",
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
        "--verbose",
        help="Log information to screen",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        help="Log debug information to screen",
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
    version = "v0.5.0"
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

    logging.info(f"\nCase                 : {arguments.case}")
    logging.info(f"Injection point info : {arguments.injection_point_info}")
    logging.info(f"Output CSV file      : {arguments.output}")
    logging.info(f"Threshold SGAS       : {arguments.threshold_sgas}")
    logging.info(f"Threshold AMFG       : {arguments.threshold_amfg}\n")


def calculate_plume_extents(
    case: str,
    injxy: Tuple[float, float],
    threshold_sgas: float = DEFAULT_THRESHOLD_SGAS,
    threshold_amfg: float = DEFAULT_THRESHOLD_AMFG,
) -> Tuple[List[List], Optional[List[List]], Optional[str]]:
    """
    Find plume extents per date for SGAS and AMFG/XMF2.
    """
    logging.info("\nStart calculating plume extent")
    grid = Grid(f"{case}.EGRID")
    unrst = ResdataFile(f"{case}.UNRST")

    # First calculate distance from injection point to center of all cells
    nactive = grid.get_num_active()
    logging.info(f"Number of active grid cells                    : {nactive:>10}")
    dist = np.zeros(shape=(nactive,))
    for i in range(nactive):
        center = grid.get_xyz(active_index=i)
        dist[i] = np.sqrt((center[0] - injxy[0]) ** 2 + (center[1] - injxy[1]) ** 2)
    logging.info(f"Smallest distance grid cell to injection point : {min(dist):>10.1f}")
    logging.info(f"Largest distance grid cell to injection point  : {max(dist):>10.1f}")
    logging.info(
        f"Average distance grid cell to injection point  : {sum(dist)/len(dist):>10.1f}"
    )

    sgas_results = _find_max_distances_per_time_step(
        "SGAS", threshold_sgas, unrst, dist
    )
    logging.info("Done calculating plume extent for SGAS.")

    if "AMFG" in unrst:
        amfg_results = _find_max_distances_per_time_step(
            "AMFG", threshold_amfg, unrst, dist
        )
        amfg_key = "AMFG"
        logging.info("Done calculating plume extent for AMFG.")
    elif "XMF2" in unrst:
        amfg_results = _find_max_distances_per_time_step(
            "XMF2", threshold_amfg, unrst, dist
        )
        amfg_key = "XMF2"
        logging.info("Done calculating plume extent for XMF2.")
    else:
        amfg_results = None
        amfg_key = None
        logging.warning("WARNING: Neither AMFG nor XMF2 exists as properties.")

    return (sgas_results, amfg_results, amfg_key)


def _find_max_distances_per_time_step(
    attribute_key: str, threshold: float, unrst: ResdataFile, dist: np.ndarray
) -> List[List]:
    """
    Find max plume distance for each step
    """
    nsteps = len(unrst.report_steps)
    dist_vs_date = np.zeros(shape=(nsteps,))
    for i in range(nsteps):
        data = unrst[attribute_key][i].numpy_view()
        plumeix = np.where(data > threshold)[0]
        maxdist = 0.0
        if len(plumeix) > 0:
            maxdist = dist[plumeix].max()

        dist_vs_date[i] = maxdist

    output = []
    for i, d in enumerate(unrst.report_dates):
        temp = [d.strftime("%Y-%m-%d"), dist_vs_date[i]]
        output.append(temp)

    return output


def _log_results(df: pd.DataFrame, amfg_key: str) -> None:
    dfs = df.sort_values("date")
    logging.info("\nSummary of results:")
    logging.info("===================")
    logging.info(f"Number of dates             : {len(dfs['date'].unique()):>11}")
    logging.info(f"First date                  : {dfs['date'].iloc[0]:>11}")
    logging.info(f"Last date                   : {dfs['date'].iloc[-1]:>11}")
    logging.info(
        f"End state max distance SGAS : {dfs['MAX_DISTANCE_SGAS'].iloc[-1]:>11.1f}"
    )
    if amfg_key is not None:
        value = dfs["MAX_DISTANCE_" + amfg_key].iloc[-1]
        logging.info(f"End state max distance {amfg_key} : {value:>11.1f}")


def _collect_results_into_dataframe(
    sgas_results: List[List],
    amfg_results: Optional[List[List]],
    amfg_key: str,
) -> pd.DataFrame:
    sgas_df = pd.DataFrame.from_records(
        sgas_results, columns=["date", "MAX_DISTANCE_SGAS"]
    )
    if amfg_results is not None:
        amfg_df = pd.DataFrame.from_records(
            amfg_results, columns=["date", "MAX_DISTANCE_" + amfg_key]
        )
        df = pd.merge(sgas_df, amfg_df, on="date")
    else:
        df = sgas_df
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
                print(
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
        print(
            f"No matches for well name {well_name}, input is either mistyped \
            or well does not exist."
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


def main():
    """
    Calculate plume extent using EGRID and UNRST-files. Calculated for SGAS
    and AMFG/XMF2. Output is plume extent per date written to a CSV file.
    """
    args = _make_parser().parse_args()
    _setup_log_configuration(args)
    _log_input_configuration(args)

    injxy = _calculate_well_coordinates(
        args.case,
        args.injection_point_info,
    )
    (sgas_results, amfg_results, amfg_key) = calculate_plume_extents(
        args.case,
        injxy,
        args.threshold_sgas,
        args.threshold_amfg,
    )

    if args.output is None:
        p = Path(args.case).parents[2]
        p2 = p / "share" / "results" / "tables" / "plume_extent.csv"
        output_file = str(p2)
    else:
        output_file = args.output

    df = _collect_results_into_dataframe(sgas_results, amfg_results, amfg_key)
    _log_results(df, amfg_key)
    df.to_csv(output_file, index=False)
    logging.info("\nDone exporting results to CSV file.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
