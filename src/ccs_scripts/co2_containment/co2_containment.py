#!/usr/bin/env python
"""
Calculates the amount of CO2 inside and outside a given perimeter,
and separates the result per formation and phase (gas/dissolved).
Output is a table in CSV format.
"""
import argparse
import dataclasses
import getpass
import logging
import os
import pathlib
import platform
import socket
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shapely.geometry
import yaml

from ccs_scripts.co2_containment.calculate import (
    ContainedCo2,
    calculate_co2_containment,
)
from ccs_scripts.co2_containment.co2_calculation import (
    CalculationType,
    Co2Data,
    _set_calc_type_from_input_string,
    calculate_co2,
)

DESCRIPTION = """
Calculates the amount of CO2 inside and outside a given perimeter, and
separates the result per formation and phase (gas/dissolved). Output is a table
on CSV format.

The most common use of the script is to calculate CO2 mass. Options for
calculation type input:

"mass": CO2 mass (kg), the default option
"cell_volume": CO2 volume (m3), a simple calculation finding the grid cells
with some CO2 and summing the volume of those cells
"actual_volume": CO2 volume (m3), an attempt to calculate a more precise
representative volume of CO2
"""

CATEGORY = "modelling.reservoir"


# pylint: disable=too-many-arguments
def calculate_out_of_bounds_co2(
    grid_file: str,
    unrst_file: str,
    init_file: str,
    compact: bool,
    calc_type_input: str,
    zone_info: Dict,
    region_info: Dict,
    file_containment_polygon: Optional[str] = None,
    file_hazardous_polygon: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculates sum of co2 mass or volume at each time step. Use polygons
    to divide into different categories (inside / outside / hazardous). Result
    is a data frame.

    Args:
        grid_file (str): Path to EGRID-file
        unrst_file (str): Path to UNRST-file
        init_file (str): Path to INIT-file
        compact (bool): Write the output to a single file as compact as possible
        calc_type_input (str): Choose mass / cell_volume / actual_volume
        file_containment_polygon (str): Path to polygon defining the
            containment area
        file_hazardous_polygon (str): Path to polygon defining the
            hazardous area
        zone_info (Dict): Dictionary containing path to zone-file,
            or zranges (if the zone-file is provided as a YAML-file
            with zones defined through intervals in depth)
            as well as a list connecting zone-numbers to names
        region_info (Dict): Dictionary containing path to potential region-file,
            and list connecting region-numbers to names, if available

    Returns:
        pd.DataFrame
    """
    co2_data = calculate_co2(
        grid_file, unrst_file, zone_info, region_info, calc_type_input, init_file
    )
    if file_containment_polygon is not None:
        containment_polygon = _read_polygon(file_containment_polygon)
    else:
        containment_polygon = None
    if file_hazardous_polygon is not None:
        hazardous_polygon = _read_polygon(file_hazardous_polygon)
    else:
        hazardous_polygon = None
    return calculate_from_co2_data(
        co2_data,
        containment_polygon,
        hazardous_polygon,
        compact,
        calc_type_input,
        zone_info,
        region_info,
    )


def calculate_from_co2_data(
    co2_data: Co2Data,
    containment_polygon: shapely.geometry.Polygon,
    hazardous_polygon: Union[shapely.geometry.Polygon, None],
    compact: bool,
    calc_type_input: str,
    zone_info: Dict,
    region_info: Dict,
) -> Union[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Use polygons to divide co2 mass or volume into different categories
    (inside / outside / hazardous). Result is a data frame.

    Args:
        co2_data (Co2Data): Mass/volume of CO2 at each time step
        containment_polygon (shapely.geometry.Polygon): Polygon defining the
            containment area
        hazardous_polygon (shapely.geometry.Polygon): Polygon defining the
            hazardous area
        compact (bool):
        calc_type_input (str): Choose mass / cell_volume / actual_volume
        zone_info (Dict): Dictionary containing zone information
        region_info (Dict): Dictionary containing region information

    Returns:
        pd.DataFrame
    """
    calc_type = _set_calc_type_from_input_string(calc_type_input.lower())
    contained_co2 = calculate_co2_containment(
        co2_data,
        containment_polygon,
        hazardous_polygon,
        zone_info,
        region_info,
        calc_type=calc_type,
    )
    data_frame = _construct_containment_table(contained_co2)
    if compact:
        return data_frame
    logging.info("\nMerge data rows for data frame")
    if co2_data.zone is None and co2_data.region is None:
        return _merge_date_rows(data_frame, calc_type)
    if co2_data.region is None:
        return {
            "zone": {
                z: _merge_date_rows(g, calc_type) for z, g in data_frame.groupby("zone")
            }
        }
    if co2_data.zone is None:
        return {
            "region": {
                z: _merge_date_rows(g, calc_type)
                for z, g in data_frame.groupby("region")
            }
        }
    return {
        "zone": {
            z: _merge_date_rows(g, calc_type) for z, g in data_frame.groupby("zone")
        },
        "region": {
            z: _merge_date_rows(g, calc_type) for z, g in data_frame.groupby("region")
        },
    }


def _read_polygon(polygon_file: str) -> shapely.geometry.Polygon:
    """
    Reads a polygon from file.

    Args:
        polygon_file (str): Path to polygon file

    Returns:
        shapely.geometry.Polygon
    """
    poly_xy = np.genfromtxt(polygon_file, skip_header=1, delimiter=",")[:, :2]
    return shapely.geometry.Polygon(poly_xy)


def _construct_containment_table(
    contained_co2: List[ContainedCo2],
) -> pd.DataFrame:
    """
    Creates a data frame from calculated CO2 data.

    Args:
        contained_co2 (list of ContainedCo2): CO2 data divided into phases/locations

    Returns:
        pd.DataFrame
    """
    records = [dataclasses.asdict(c) for c in contained_co2]
    return pd.DataFrame.from_records(records)


# pylint: disable-msg=too-many-locals
def _merge_date_rows(
    data_frame: pd.DataFrame, calc_type: CalculationType
) -> pd.DataFrame:
    """
    Uses input dataframe to calculate various new columns and renames/merges
    some columns.

    Args:
        data_frame (pd.DataFrame): Input data frame
        calc_type (CalculationType): Choose mass / cell_volume /
            actual_volume from enum CalculationType

    Returns:
        pd.DataFrame: Output data frame
    """
    data_frame = data_frame.drop(columns=["zone", "region"], axis=1, errors="ignore")
    # Total
    df1 = (
        data_frame.drop(["phase", "location"], axis=1)
        .groupby(["date"])
        .sum()
        .rename(columns={"amount": "total"})
    )
    total_df = df1.copy()
    if calc_type == CalculationType.CELL_VOLUME:
        df2 = data_frame.drop("phase", axis=1).groupby(["location", "date"]).sum()
        df2a = df2.loc[("contained",)].rename(columns={"amount": "total_contained"})
        df2b = df2.loc[("outside",)].rename(columns={"amount": "total_outside"})
        df2c = df2.loc[("hazardous",)].rename(columns={"amount": "total_hazardous"})
        for _df in [df2a, df2b, df2c]:
            total_df = total_df.merge(_df, on="date", how="left")
    else:
        df2 = data_frame.drop("location", axis=1).groupby(["phase", "date"]).sum()
        df2a = df2.loc["gas"].rename(columns={"amount": "total_gas"})
        df2b = df2.loc["aqueous"].rename(columns={"amount": "total_aqueous"})
        # Total by containment
        df3 = data_frame.drop("phase", axis=1).groupby(["location", "date"]).sum()
        df3a = df3.loc[("contained",)].rename(columns={"amount": "total_contained"})
        df3b = df3.loc[("outside",)].rename(columns={"amount": "total_outside"})
        df3c = df3.loc[("hazardous",)].rename(columns={"amount": "total_hazardous"})
        # Total by containment and phase
        df4 = data_frame.groupby(["phase", "location", "date"]).sum()
        df4a = df4.loc["gas", "contained"].rename(columns={"amount": "gas_contained"})
        df4b = df4.loc["aqueous", "contained"].rename(
            columns={"amount": "aqueous_contained"}
        )
        df4c = df4.loc["gas", "outside"].rename(columns={"amount": "gas_outside"})
        df4d = df4.loc["aqueous", "outside"].rename(
            columns={"amount": "aqueous_outside"}
        )
        df4e = df4.loc["gas", "hazardous"].rename(columns={"amount": "gas_hazardous"})
        df4f = df4.loc["aqueous", "hazardous"].rename(
            columns={"amount": "aqueous_hazardous"}
        )
        for _df in [df2a, df2b, df3a, df3b, df3c, df4a, df4b, df4c, df4d, df4e, df4f]:
            total_df = total_df.merge(_df, on="date", how="left")
    return total_df.reset_index()


def get_parser() -> argparse.ArgumentParser:
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
    )
    parser.add_argument(
        "calc_type_input",
        help="CO2 calculation options: mass / cell_volume / actual_volume.",
    )
    parser.add_argument(
        "--root_dir",
        help="Path to root directory. The other paths can be provided relative \
        to this or as absolute paths. Default is 2 levels up from Eclipse case.",
        default=None,
    )
    parser.add_argument(
        "--out_dir",
        help="Path to output directory (file name is set to \
        'plume_<calculation type>.csv'). \
        Defaults to <root_dir>/share/results/tables.",
        default=None,
    )
    parser.add_argument(
        "--containment_polygon",
        help="Path to polygon that determines the bounds of the containment area. \
        Count all CO2 as contained if polygon is not provided.",
        default=None,
    )
    parser.add_argument(
        "--hazardous_polygon",
        help="Path to polygon that determines the bounds of the hazardous area.",
        default=None,
    )
    parser.add_argument(
        "--egrid",
        help="Path to EGRID file. Overwrites <case> if provided.",
        default=None,
    )
    parser.add_argument(
        "--unrst",
        help="Path to UNRST file. Overwrites <case> if provided.",
        default=None,
    )
    parser.add_argument(
        "--init",
        help="Path to INIT file. Overwrites <case> if provided.",
        default=None,
    )
    parser.add_argument(
        "--zonefile", help="Path to file containing zone information.", default=None
    )
    parser.add_argument(
        "--regionfile", help="Path to file containing region information.", default=None
    )
    parser.add_argument(
        "--region_property",
        help="Property in INIT file containing integer grid of regions.",
        default=None,
    )
    parser.add_argument(
        "--compact",
        help="Write the output to a single file as compact as possible.",
        action="store_true",
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
    if args.hazardous_polygon == "-1":
        args.hazardous_polygon = None


class InputError(Exception):
    """Raised when relative paths are provided when absolute ones are expected"""


def process_args() -> argparse.Namespace:
    """
    Process arguments and do some minor conversions.
    Create absolute paths if relative paths are provided.

    Returns:
        argparse.Namespace
    """
    args = get_parser().parse_args()
    args.calc_type_input = args.calc_type_input.lower()

    _replace_default_dummies_from_ert(args)

    if args.root_dir is None:
        p = pathlib.Path(args.case).parents
        if len(p) < 3:
            error_text = "Invalid input, <case> must have at least two parent levels \
            if <root_dir> is not provided."
            raise InputError(error_text)
        args.root_dir = p[2]
    if args.out_dir is None:
        args.out_dir = os.path.join(args.root_dir, "share", "results", "tables")
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
        "hazardous_polygon",
    ]
    for key in paths:
        if adict[key] is not None and not pathlib.Path(adict[key]).is_absolute():
            adict[key] = os.path.join(args.root_dir, adict[key])

    if args.egrid is None:
        if args.case.endswith(".EGRID"):
            args.egrid = args.case
        else:
            args.egrid = args.case + ".EGRID"
    if args.unrst is None:
        args.unrst = args.egrid.replace(".EGRID", ".UNRST")
    if args.init is None:
        args.init = args.egrid.replace(".EGRID", ".INIT")

    if args.debug:
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(format="%(message)s", level=logging.INFO)
    else:
        logging.basicConfig(format="%(message)s", level=logging.WARNING)

    return args


def check_input(arguments: argparse.Namespace):
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
    if not os.path.isfile(arguments.init):
        files_not_found.append(arguments.init)
    if arguments.zonefile is not None and not os.path.isfile(arguments.zonefile):
        files_not_found.append(arguments.zonefile)
    if arguments.regionfile is not None and not os.path.isfile(arguments.regionfile):
        files_not_found.append(arguments.regionfile)
    if arguments.containment_polygon is not None and not os.path.isfile(
        arguments.containment_polygon
    ):
        files_not_found.append(arguments.containment_polygon)
    if arguments.hazardous_polygon is not None and not os.path.isfile(
        arguments.hazardous_polygon
    ):
        files_not_found.append(arguments.hazardous_polygon)
    if files_not_found:
        error_text = "The following file(s) were not found:"
        for file in files_not_found:
            error_text += "\n  * " + file
        raise FileNotFoundError(error_text)

    if not os.path.isdir(arguments.out_dir):
        logging.warning("Output directory doesn't exist. Creating a new folder.")
        os.mkdir(arguments.out_dir)


def process_zonefile_if_yaml(zonefile: str) -> Optional[Dict[str, List[int]]]:
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
                logging.error(exc)
                sys.exit(1)
        if "zranges" not in zfile:
            error_text = "The yaml zone file must be in the format:\nzranges:\
            \n    - Zone1: [1, 5]\n    - Zone2: [6, 10]\n    - Zone3: [11, 14])"
            raise InputError(error_text)
        zranges = zfile["zranges"]
        if len(zranges) > 1:
            zranges_ = zranges[0]
            for zr in zranges[1:]:
                zranges_.update(zr)
            zranges = zranges_
        return zranges
    return None


def log_input_configuration(arguments_processed: argparse.Namespace) -> None:
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
    logging.info("CCS-scripts - Containment calculations")
    logging.info("======================================")
    logging.info(f"Version             : {version}")
    logging.info(f"Date and time       : {date_time}")
    logging.info(f"User                : {getpass.getuser()}")
    logging.info(f"Host                : {socket.gethostname()}")
    logging.info(f"Platform            : {platform.system()} ({platform.release()})")
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    logging.info(f"Python version      : {py_version}")

    logging.info(f"\nCase                : {arguments_processed.case}")
    logging.info(f"Calculation type    : {arguments_processed.calc_type_input}")
    logging.info(f"Root directory      : {arguments_processed.root_dir}")
    logging.info(f"Output directory    : {arguments_processed.out_dir}")
    logging.info(f"Containment polygon : {arguments_processed.containment_polygon}")
    logging.info(f"Hazardous polygon   : {arguments_processed.hazardous_polygon}")
    logging.info(f"EGRID file          : {arguments_processed.egrid}")
    logging.info(f"UNRST file          : {arguments_processed.unrst}")
    logging.info(f"INIT file           : {arguments_processed.init}")
    logging.info(f"Zone file           : {arguments_processed.zonefile}")
    logging.info(
        f"Compact             : {'yes' if arguments_processed.compact else 'no'}\n"
    )


def log_summary_of_results(df: pd.DataFrame) -> None:
    dfs = df.sort_values("date")
    last_date = max(df["date"])
    df_subset = dfs[dfs["date"] == last_date]
    if "zone" in df_subset and any(df_subset["zone"].str.contains("all", na=False)):
        df_subset = df_subset[df_subset["zone"] == "all"]
    if "region" in df_subset and any(df_subset["region"].str.contains("all", na=False)):
        df_subset = df_subset[df_subset["region"] == "all"]
    total = df_subset["total"].iloc[-1]
    n = len(f"{total:.1f}")

    logging.info("\nSummary of results:")
    logging.info("===================")
    logging.info(f"Number of dates     : {len(dfs['date'].unique())}")
    logging.info(f"First date          : {dfs['date'].iloc[0]}")
    logging.info(f"Last date           : {dfs['date'].iloc[-1]}")
    logging.info(f"End state total     : {total:{n+1}.1f}")
    if "total_gas" in df_subset:
        value = df_subset["total_gas"].iloc[-1]
        percent = 100.0 * value / total if total > 0.0 else 0.0
        logging.info(f"End state gaseous   : {value:{n+1}.1f}  ({percent:.1f} %)")
    if "total_aqueous" in df_subset:
        value = df_subset["total_aqueous"].iloc[-1]
        percent = 100.0 * value / total if total > 0.0 else 0.0
        logging.info(f"End state aqueous   : {value:{n+1}.1f}  ({percent:.1f} %)")
    value = df_subset["total_contained"].iloc[-1]
    percent = 100.0 * value / total if total > 0.0 else 0.0
    logging.info(f"End state contained : {value:{n+1}.1f}  ({percent:.1f} %)")
    value = df_subset["total_outside"].iloc[-1]
    percent = 100.0 * value / total if total > 0.0 else 0.0
    logging.info(f"End state outside   : {value:{n+1}.1f}  ({percent:.1f} %)")
    value = df_subset["total_hazardous"].iloc[-1]
    percent = 100.0 * value / total if total > 0.0 else 0.0
    logging.info(f"End state hazardous : {value:{n+1}.1f}  ({percent:.1f} %)")
    if "zone" in dfs:
        logging.info("Split into zones?   : yes")
        unique_zones = dfs["zone"].unique()
        n_zones = (
            len(unique_zones) - 1 if "all" in dfs["zone"].values else len(unique_zones)
        )
        logging.info(f"Number of zones     : {n_zones}")
        logging.info(f"Zones               : {', '.join(unique_zones)}")
    else:
        logging.info("Split into zones?   : no")
    if "region" in dfs:
        logging.info("Split into regions? : yes")
        unique_regions = dfs["region"].unique()
        n_regions = (
            len(unique_regions) - 1
            if "all" in dfs["region"].values
            else len(unique_regions)
        )
        logging.info(f"Number of regions   : {n_regions}")
        logging.info(f"Regions             : {', '.join(unique_regions)}")
    else:
        logging.info("Split into regions? : no")


def _combine_data_frame(
    data_frame: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    zone_info: Dict[str, Any],
    region_info: Dict[str, Any],
) -> pd.DataFrame:
    """
    Combine data frames from different zones into one single data frame
    """
    if zone_info["source"] is None and region_info["int_to_region"] is None:
        assert isinstance(data_frame, pd.DataFrame)
        return data_frame

    assert isinstance(data_frame, Dict)
    zone_df = pd.DataFrame()
    region_df = pd.DataFrame()
    summed_part = pd.DataFrame()
    if zone_info["source"] is not None:
        assert isinstance(data_frame["zone"], Dict)
        assert zone_info["int_to_zone"] is not None
        zone_keys = list(data_frame["zone"].keys())
        for key in zone_info["int_to_zone"]:
            if key is not None:
                if key in zone_keys:
                    _df = data_frame["zone"][key]
                else:
                    _df = data_frame["zone"][zone_keys[0]]
                    numeric_cols = _df.select_dtypes(include=["number"]).columns
                    _df[numeric_cols] = 0
                _df["zone"] = [key] * _df.shape[0]
                zone_df = pd.concat([zone_df, _df])
        if region_info["int_to_region"] is not None:
            zone_df["region"] = ["all"] * zone_df.shape[0]
        summed_part = zone_df.groupby("date").sum(numeric_only=True).reset_index()
        summed_part["zone"] = ["all"] * summed_part.shape[0]
    if region_info["int_to_region"] is not None:
        assert isinstance(data_frame["region"], Dict)
        region_keys = list(data_frame["region"].keys())
        for key in region_info["int_to_region"]:
            if key is not None:
                if key in region_keys:
                    _df = data_frame["region"][key]
                else:
                    _df = data_frame["region"][region_keys[0]]
                    numeric_cols = _df.select_dtypes(include=["number"]).columns
                    _df[numeric_cols] = 0
                _df["region"] = [key] * _df.shape[0]
                region_df = pd.concat([region_df, _df])
        if zone_info["source"] is None:
            summed_part = region_df.groupby("date").sum(numeric_only=True).reset_index()
        else:
            region_df["zone"] = ["all"] * region_df.shape[0]
        summed_part["region"] = ["all"] * summed_part.shape[0]
    combined_df = pd.concat([summed_part, zone_df, region_df])
    return combined_df


def export_output_to_csv(
    out_dir: str,
    calc_type_input: str,
    data_frame: pd.DataFrame,
):
    """
    Exports the results to a csv file, named according to the calculation type
    (mass / cell_volume / actual_volume)
    """
    file_name = f"plume_{calc_type_input}.csv"
    logging.info(f"\nExport results to CSV file: {file_name}")
    file_path = os.path.join(out_dir, file_name)
    if os.path.isfile(file_path):
        logging.info(f"Output CSV file already exists. Overwriting: {file_path}")

    data_frame.to_csv(file_path, index=False)


def main() -> None:
    """
    Takes input arguments and calculates total co2 mass or volume at each time
    step, divided into different phases and locations. Creates a data frame,
    then exports the data frame to a csv file.
    """
    arguments_processed = process_args()
    check_input(arguments_processed)
    zone_info = {
        "source": arguments_processed.zonefile,
        "zranges": None,
        "int_to_zone": None,
    }
    region_info = {
        "source": arguments_processed.regionfile,
        "int_to_region": None,
        "property_name": arguments_processed.region_property,
    }
    if zone_info["source"] is not None:
        zone_info["zranges"] = process_zonefile_if_yaml(zone_info["source"])

    log_input_configuration(arguments_processed)

    data_frame = calculate_out_of_bounds_co2(
        arguments_processed.egrid,
        arguments_processed.unrst,
        arguments_processed.init,
        arguments_processed.compact,
        arguments_processed.calc_type_input,
        zone_info,
        region_info,
        arguments_processed.containment_polygon,
        arguments_processed.hazardous_polygon,
    )
    df_combined = _combine_data_frame(data_frame, zone_info, region_info)
    log_summary_of_results(df_combined)
    export_output_to_csv(
        arguments_processed.out_dir,
        arguments_processed.calc_type_input,
        df_combined,
    )


if __name__ == "__main__":
    main()
