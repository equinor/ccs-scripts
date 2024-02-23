#!/usr/bin/env python


################################################################################
# Script calculating the area extent of the plume depending on which map / date
# are present in the share/results/maps folder
#
# Created by : Jorge Sicacha (NR), Oct 2022
# Modified by: Floriane Mortier (fmmo), Nov 2022 - To fit FMU workflow
#
################################################################################

import argparse
import getpass
import glob
import logging
import os
import pathlib
import platform
import socket
import subprocess
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xtgeo


def _make_parser():
    parser = argparse.ArgumentParser(description="Calculate plume area")
    parser.add_argument("input", help="Path to maps created through grid3d-maps")
    parser.add_argument(
        "--output",
        help="Path to output CSV file",
        default=None,
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


def _find_formations(search_path: str, rskey: str) -> Optional[Tuple[np.ndarray, str]]:
    # Try different capitalizations of rskey:
    file_names_1 = glob.glob(search_path + "*max_" + rskey + "*.gri")
    file_names_2 = glob.glob(search_path + "*max_" + rskey.lower() + "*.gri")
    file_names_3 = glob.glob(search_path + "*max_" + rskey.upper() + "*.gri")

    if file_names_1:
        rskey_updated = rskey
    elif file_names_2:
        rskey_updated = rskey.lower()
    elif file_names_3:
        rskey_updated = rskey.upper()
    else:
        logging.info(f"No surface files found for {rskey}.")
        return None

    formation_list = []
    for file in glob.glob(search_path + "*max_" + rskey_updated + "*.gri"):
        fm_name = pathlib.Path(file).stem.split("--")[0]

        if fm_name in formation_list:
            pass
        else:
            formation_list.append(fm_name)

    return np.array(formation_list), rskey_updated


def _find_dates(search_path: str, fm: np.ndarray, rskey: str) -> List[str]:
    date_list = []

    for file in glob.glob(search_path + fm[0] + "*max_" + rskey + "*.gri"):
        full_date = pathlib.Path(file).stem.split("--")[2]
        date = f"{full_date[0:4]}-{full_date[4:6]}-{full_date[6:8]}"

        if date in date_list:
            pass
        else:
            date_list.append(date)

    return date_list


def _neigh_nodes(x: Tuple[np.int64, np.int64]) -> set:
    # If all the four nodes of the cell are not masked we count the area
    sq_vert = {(x[0] + 1, x[1]), (x[0], int(x[1]) + 1), (x[0] + 1, x[1] + 1)}

    return sq_vert


def calculate_plume_area(path: str, rskey: str) -> Optional[List[List[float]]]:
    """
    Finds plume area for each formation and year for a given rskey (for instance
    SGAS or AMFG). The plume areas are found using data from surface files (.gri).
    """
    logging.info(f"Calculating plume area for           : {rskey}")

    if path[-1] != "/":
        path = path + "/"
    out = _find_formations(path, rskey)
    if not out:
        return None
    else:
        formations, rskey_updated = out
        logging.info(f"Formations extracted from input maps : {', '.join(formations)}")

    dates = np.array(_find_dates(path, formations, rskey_updated))
    logging.info(f"Number of dates                      : {len(dates)}")
    logging.info(f"First date                           : {min(dates)}")
    logging.info(f"Last date                            : {max(dates)}\n")

    var = "max_" + rskey_updated
    logging.info(f"Looking for maps with the following text: {var}\n")
    list_out = []
    for fm in formations:
        for date in dates:
            year = date[0:4]
            path_file = glob.glob(path + fm + "--" + var + "--" + year + "*.gri")
            mysurf = xtgeo.surface_from_file(path_file[0])
            use_nodes = np.ma.nonzero(mysurf.values)  # Indexes of the existing nodes
            use_nodes = set(list(tuple(zip(use_nodes[0], use_nodes[1]))))
            all_neigh_nodes = list(map(_neigh_nodes, use_nodes))
            test0 = [xx.issubset(use_nodes) for xx in all_neigh_nodes]
            list_out_temp = [
                date,
                float(sum(t * mysurf.xinc * mysurf.yinc for t in test0)),
                fm,
            ]
            list_out.append(list_out_temp)

    return list_out


def _read_args() -> Tuple[str, str]:
    args = _make_parser().parse_args()
    _setup_log_configuration(args)

    input_path = args.input
    output_path = args.output

    if not os.path.isdir(input_path):
        text = f"Input surface directory not found: {input_path}"
        raise FileNotFoundError(text)

    return input_path, output_path


def _log_input_configuration(input_path: str, output_path: str) -> None:
    version = "v0.4.0"

    source_dir = os.path.dirname(os.path.abspath(__file__))
    is_dev_version = True
    if is_dev_version:
        version += "_dev"

        short_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=source_dir
            )
            .decode("ascii")
            .strip()
        )
        version += " (latest git commit: " + short_hash + ")"

    now = datetime.now()
    d = now.strftime("%B %d, %Y %H:%M:%S")
    logging.info("CCS-scripts - Plume area calculations")
    logging.info("=====================================")
    logging.info(f"Version             : {version}")
    logging.info(f"Date and time       : {d}")
    logging.info(f"User                : {getpass.getuser()}")
    logging.info(f"Host                : {socket.gethostname()}")
    logging.info(f"Platform            : {platform.system()} ({platform.release()})")
    logging.info(
        f"Python version      : {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    logging.info(f"\nInput path  : {input_path}")
    logging.info(f"Output path : {output_path}\n")


def _convert_to_data_frame(results: List[List[float]], rskey: str) -> pd.DataFrame:
    # Convert plume area results into a Pandas DataFrame
    df = pd.DataFrame.from_records(
        results, columns=["date", "AREA_" + rskey, "FORMATION_" + rskey]
    )
    df = df.pivot(index="date", columns="FORMATION_" + rskey, values="AREA_" + rskey)
    df.reset_index(inplace=True)
    df.columns.name = None
    df.columns = [x + "_" + rskey if x != "date" else x for x in df.columns]

    return df


def _log_results(df: pd.DataFrame) -> None:
    logging.info("\nSummary of results:")
    logging.info("===================")
    logging.info(f"Number of dates : {len(df['date'].unique()):>11}")
    logging.info(f"First date      : {df['date'].iloc[0]:>11}")
    logging.info(f"Last date       : {df['date'].iloc[-1]:>11}")
    columns = [c for c in df if c != "date"]
    n = max(len(c) for c in columns)
    logging.info(f"End state plume area:")
    for c in columns:
        logging.info(f"    * {c:<{n+1 }}: {df[c].iloc[-1]:>11}")


def main():
    """
    Reads directory of input surface files (.gri) and calculates plume area
    for SGAS and AMFG per formation and year. Collects the results into a CSV
    file.
    """
    input_path, output_path = _read_args()

    if output_path is None:
        p = pathlib.Path("share") / "results" / "tables" / "plume_area.csv"
        output_path = str(p)
    _log_input_configuration(input_path, output_path)

    sgas_df, amfg_df, xmf2_df = None, None, None
    sgas_results = calculate_plume_area(input_path, "sgas")
    if sgas_results:
        logging.info("\nDone calculating SGAS plume areas.")
        sgas_df = _convert_to_data_frame(sgas_results, "SGAS")

    amfg_results = calculate_plume_area(input_path, "amfg")
    if amfg_results:
        logging.info("\nDone calculating AMFG plume areas.")
        amfg_df = _convert_to_data_frame(amfg_results, "AMFG")

    xmf2_results = calculate_plume_area(input_path, "xmf2")
    if xmf2_results:
        logging.info("\nDone calculating XMF2 plume areas.")
        xmf2_df = _convert_to_data_frame(xmf2_results, "XMF2")

    # Merge the data frames
    df = None
    for df_prop in [sgas_df, amfg_df, xmf2_df]:
        if df_prop is not None:
            if df is None:
                df = df_prop
            else:
                df = pd.merge(df, df_prop)

    _log_results(df)

    if df is not None:
        df.to_csv(output_path, index=False)
        logging.info("\nDone exporting results to CSV file.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
