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
from typing import Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import pandas as pd
import shapely.geometry
import yaml
from resdata.grid import Grid
from resdata.resfile import ResdataFile

from ccs_scripts.co2_containment.calculate import (
    ContainedCo2,
    calculate_co2_containment,
)
from ccs_scripts.co2_containment.co2_calculation import (
    CalculationType,
    Co2Data,
    RegionInfo,
    ZoneInfo,
    _fetch_properties,
    _set_calc_type_from_input_string,
    calculate_co2,
    find_active_and_gasless_cells,
)
from ccs_scripts.co2_plume_tracking.co2_plume_tracking import (
    DEFAULT_THRESHOLD_DISSOLVED,
    Configuration,
    calculate_plume_groups,
)
from ccs_scripts.co2_plume_tracking.utils import InjectionWellData

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
    calc_type_input: str,
    zone_info: ZoneInfo,
    region_info: RegionInfo,
    residual_trapping: bool,
    injection_wells: List[InjectionWellData],
    file_containment_polygon: Optional[str] = None,
    file_hazardous_polygon: Optional[str] = None,
    gas_molar_mass: Optional[float] = None,
) -> pd.DataFrame:
    """
    Calculates sum of co2 mass or volume at each time step. Use polygons
    to divide into different categories (inside / outside / hazardous). Result
    is a data frame.

    Args:
        grid_file (str): Path to EGRID-file
        unrst_file (str): Path to UNRST-file
        init_file (str): Path to INIT-file
        calc_type_input (str): Choose mass / cell_volume / actual_volume
        file_containment_polygon (str): Path to polygon defining the
            containment area
        file_hazardous_polygon (str): Path to polygon defining the
            hazardous area
        zone_info (ZoneInfo): Containing path to zone-file,
            or zranges (if the zone-file is provided as a YAML-file
            with zones defined through intervals in depth)
            as well as a list connecting zone-numbers to names
        region_info (RegionInfo): Containing path to potential region-file,
            and list connecting region-numbers to names, if available
        residual_trapping (bool): Indicate if residual trapping should be calculated
        injection_wells (List): Injection wells used for plume tracking
        gas_molar_mass (float): Hydrocarbon gas molar mass. (Applies for cases with more
            than two components)

    Returns:
        pd.DataFrame
    """
    co2_data = calculate_co2(
        grid_file,
        unrst_file,
        zone_info,
        region_info,
        residual_trapping,
        calc_type_input,
        init_file,
        gas_molar_mass,
    )

    if file_containment_polygon is not None:
        containment_polygon = _read_polygon(file_containment_polygon)
    else:
        containment_polygon = None
    if file_hazardous_polygon is not None:
        hazardous_polygon = _read_polygon(file_hazardous_polygon)
    else:
        hazardous_polygon = None

    if len(injection_wells) == 0:
        plume_groups = None
    else:
        plume_groups = _find_plume_groups(grid_file, unrst_file, injection_wells)

    return calculate_from_co2_data(
        co2_data,
        containment_polygon,
        hazardous_polygon,
        calc_type_input,
        zone_info.int_to_zone,
        region_info.int_to_region,
        residual_trapping,
        plume_groups,
    )


def _find_plume_groups(
    grid_file: str,
    unrst_file: str,
    injection_wells: List[InjectionWellData],
) -> Optional[List[List[str]]]:
    grid = Grid(grid_file)
    unrst = ResdataFile(unrst_file)
    if "AMFG" in unrst:
        dissolved_prop = "AMFG"
    elif "XMF2" in unrst:
        dissolved_prop = "XMF2"
    else:
        dissolved_prop = None

    if dissolved_prop is None:
        plume_groups = None
    else:
        plume_groups = calculate_plume_groups(
            attribute_key=dissolved_prop,
            threshold=0.1 * DEFAULT_THRESHOLD_DISSOLVED,
            unrst=unrst,
            grid=grid,
            inj_wells=injection_wells,
        )

        # NBNB-AS: Plume tracking works on active grid cells, containment script on
        #          gasless active cells. We do the conversion here, but could do a
        #          conversion earlier (in plume tracking)
        properties_to_extract = ["SGAS", dissolved_prop]
        properties, _ = _fetch_properties(unrst, properties_to_extract)
        active, gasless = find_active_and_gasless_cells(grid, properties, False)
        global_active_idx = active[~gasless]
        non_gasless = np.where(np.isin(active, global_active_idx))[0]
        plume_groups = [list(np.array(x)[non_gasless]) for x in plume_groups]
    return plume_groups


def calculate_from_co2_data(
    co2_data: Co2Data,
    containment_polygon: shapely.geometry.Polygon,
    hazardous_polygon: Union[shapely.geometry.Polygon, None],
    calc_type_input: str,
    int_to_zone: Optional[List[Optional[str]]],
    int_to_region: Optional[List[Optional[str]]],
    residual_trapping: bool = False,
    plume_groups: Optional[List[List[str]]] = None,
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
        calc_type_input (str): Choose mass / cell_volume / actual_volume
        int_to_zone (List): List of zone names
        int_to_region (List): List of region names
        residual_trapping (bool): Indicate if residual trapping should be calculated
        plume_groups (List): For each time step, list of plume group for each grid cell

    Returns:
        pd.DataFrame
    """
    calc_type = _set_calc_type_from_input_string(calc_type_input.lower())
    contained_co2 = calculate_co2_containment(
        co2_data,
        containment_polygon,
        hazardous_polygon,
        int_to_zone,
        int_to_region,
        calc_type,
        residual_trapping,
        plume_groups,
    )
    return _construct_containment_table(contained_co2)


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
    data_frame: pd.DataFrame, calc_type: CalculationType, residual_trapping: bool
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
    data_frame = data_frame.drop(
        columns=["zone", "region", "plume_group"], axis=1, errors="ignore"
    )
    locations = ["contained", "outside", "hazardous"]
    if calc_type == CalculationType.CELL_VOLUME:
        total_df = (
            data_frame[data_frame["containment"] == "total"]
            .drop(["phase", "containment"], axis=1)
            .rename(columns={"amount": "total"})
        )
        for location in locations:
            _df = (
                data_frame[data_frame["containment"] == location]
                .drop(columns=["phase", "containment"])
                .rename(columns={"amount": f"total_{location}"})
            )
            total_df = total_df.merge(_df, on="date", how="left")
    else:
        total_df = (
            data_frame[
                (data_frame["phase"] == "total")
                & (data_frame["containment"] == "total")
            ]
            .drop(["phase", "containment"], axis=1)
            .rename(columns={"amount": "total"})
        )
        phases = ["free_gas", "trapped_gas"] if residual_trapping else ["gas"]
        phases += ["dissolved"]
        # Total by phase
        for phase in phases:
            _df = (
                data_frame[
                    (data_frame["containment"] == "total")
                    & (data_frame["phase"] == phase)
                ]
                .drop(columns=["phase", "containment"])
                .rename(columns={"amount": f"total_{phase}"})
            )
            total_df = total_df.merge(_df, on="date", how="left")
        # Total by containment
        for location in locations:
            _df = (
                data_frame[
                    (data_frame["containment"] == location)
                    & (data_frame["phase"] == "total")
                ]
                .drop(columns=["phase", "containment"])
                .rename(columns={"amount": f"total_{location}"})
            )
            total_df = total_df.merge(_df, on="date", how="left")
        # Total by containment
        for location in locations:
            for phase in phases:
                _df = (
                    data_frame[
                        (data_frame["containment"] == location)
                        & (data_frame["phase"] == phase)
                    ]
                    .drop(columns=["phase", "containment"])
                    .rename(columns={"amount": f"{phase}_{location}"})
                )
                total_df = total_df.merge(_df, on="date", how="left")
    return total_df.reset_index(drop=True)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "no", "0"}:
        return False
    elif value.lower() in {"true", "yes", "1"}:
        return True
    elif value == "-1":
        return "-1"
    raise ValueError(f"{value} is not a valid boolean value")


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
        help="CO2 calculation options: mass / cell_volume / actual_volume. "
        "Mass is calculated in tons, volume in cubic metres.",
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
        "--zonefile",
        help="Path to yaml or roff file containing zone information.",
        default=None,
    )
    parser.add_argument(
        "--regionfile",
        help="Path to roff file containing region information. "
        "Use either 'regionfile' or 'region_property', not both.",
        default=None,
    )
    parser.add_argument(
        "--region_property",
        help="Property in INIT file containing integer grid of regions. "
        "Use either 'regionfile' or 'region_property', not both.",
        default=None,
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
    parser.add_argument(
        "--residual_trapping",
        help="Compute mass/volume of trapped CO2 in gas phase.",
        type=str_to_bool,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--readable_output",
        help="Generate output text-file that is easier to parse than the standard"
        " output.",
        type=str_to_bool,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--config_file_inj_wells",
        help="YML file with configurations for plume tracking calculations.",
        default="",
    )
    parser.add_argument(
        "--gas_molar_mass",
        help="Gas molar mass if working in COMP3/COMP4",
        default=None,
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
    if args.no_logging == "-1":
        args.no_logging = False
    if args.debug == "-1":
        args.debug = False
    if args.residual_trapping == "-1":
        args.residual_trapping = False
    if args.readable_output == "-1":
        args.readable_output = False
    if args.gas_molar_mass == "-1":
        args.gas_molar_mass = None


class InputError(Exception):
    """Raised for various mistakes in the provided input."""


# pylint: disable-msg=too-many-branches
def process_args() -> argparse.Namespace:
    """
    Process arguments and do some minor conversions.
    Create absolute paths if relative paths are provided.

    Returns:
        argparse.Namespace
    """
    args = get_parser().parse_args()
    args.calc_type_input = args.calc_type_input.lower()
    if args.gas_molar_mass is not None:
        try:
            args.gas_molar_mass = float(args.gas_molar_mass)
        except ValueError:
            raise ValueError("Invalid input: gas_molar_mass must be numeric")
    # NBNB: Remove this when residual trapping is added for cell_volume
    if args.residual_trapping and args.calc_type_input == "cell_volume":
        args.residual_trapping = False

    _replace_default_dummies_from_ert(args)

    if args.root_dir is None:
        p = pathlib.Path(args.case).parents
        if len(p) < 3:
            error_text = "Invalid input, <case> must have at least two parent levels \
            if <root_dir> is not provided."
            raise InputError(error_text)
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
        "hazardous_polygon",
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

    if args.debug:
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    elif args.no_logging:
        logging.basicConfig(format="%(message)s", level=logging.WARNING)
    else:
        logging.basicConfig(format="%(message)s", level=logging.INFO)

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

    if arguments.regionfile is not None and arguments.region_property is not None:
        raise InputError(
            "Both 'regionfile' and 'region_property' have been provided. "
            "Please provide only one of the two options."
        )

    if not os.path.isdir(arguments.out_dir):
        logging.warning("Output directory doesn't exist. Creating a new folder.")
        os.mkdir(arguments.out_dir)

    if not os.path.isfile(arguments.init):
        logging.info(f"The INIT-file {arguments.init} was not found\n")


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


def log_input_configuration(args: argparse.Namespace) -> None:
    """
    Log the provided input
    """
    version = "v0.9.2"
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
    logging.info(f"{'Hazardous polygon':<{col1}} : {args.hazardous_polygon}")
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
    config_file_inj_wells_str = (
        args.config_file_inj_wells if args.config_file_inj_wells != "" else "-"
    )
    logging.info(
        f"{'Plume tracking YAML-file':<{col1}} : " f"{config_file_inj_wells_str}\n"
    )


# pylint: disable = too-many-statements
def log_summary_of_results(
    df: pd.DataFrame,
    calc_type_input: str,
) -> None:
    """
    Log a rough summary of the output
    """
    cell_volume = calc_type_input == "cell_volume"
    dfs = df.sort_values("date")
    last_date = max(df["date"])
    df_subset = dfs[dfs["date"] == last_date]
    df_subset = df_subset[
        (df_subset["zone"] == "all")
        & (df_subset["region"] == "all")
        & (df_subset["plume_group"] == "all")
    ]
    total = extract_amount(df_subset, "total", "total", cell_volume)
    n = len(f"{total:.1f}")

    col1 = 24
    logging.info("\nSummary of results:")
    logging.info("===================")
    logging.info(f"{'Number of dates':<{col1}} : {len(dfs['date'].unique())}")
    logging.info(f"{'First date':<{col1}} : {dfs['date'].iloc[0]}")
    logging.info(f"{'Last date':<{col1}} : {dfs['date'].iloc[-1]}")
    logging.info(f"{'End state total':<{col1}} : {total:{n}.1f}")
    if not cell_volume:
        if "gas" in list(df_subset["phase"]):
            value = extract_amount(df_subset, "total", "gas")
            percent = 100.0 * value / total if total > 0.0 else 0.0
            logging.info(
                f"{'End state gaseous':<{col1}} : "
                f"{value:{n}.1f}  ={percent:>5.1f} %"
            )
        else:
            value = extract_amount(df_subset, "total", "free_gas")
            percent = 100.0 * value / total if total > 0.0 else 0.0
            logging.info(
                f"{'End state free gas':<{col1}} : "
                f"{value:{n}.1f}  ={percent:>5.1f} %"
            )
            value = extract_amount(df_subset, "total", "trapped_gas")
            percent = 100.0 * value / total if total > 0.0 else 0.0
            logging.info(
                f"{'End state trapped gas':<{col1}} : "
                f"{value:{n}.1f}  ={percent:>5.1f} %"
            )
        value = extract_amount(df_subset, "total", "dissolved")
        percent = 100.0 * value / total if total > 0.0 else 0.0
        logging.info(
            f"{'End state dissolved':<{col1}} : {value:{n}.1f}  ={percent:>5.1f} %"
        )
    value = extract_amount(df_subset, "contained", "total", cell_volume)
    percent = 100.0 * value / total if total > 0.0 else 0.0
    logging.info(
        f"{'End state contained':<{col1}} : {value:{n}.1f}  ={percent:>5.1f} %"
    )
    value = extract_amount(df_subset, "outside", "total", cell_volume)
    percent = 100.0 * value / total if total > 0.0 else 0.0
    logging.info(f"{'End state outside':<{col1}} : {value:{n}.1f}  ={percent:>5.1f} %")
    value = extract_amount(df_subset, "hazardous", "total", cell_volume)
    percent = 100.0 * value / total if total > 0.0 else 0.0
    logging.info(
        f"{'End state hazardous':<{col1}} : {value:{n}.1f}  ={percent:>5.1f} %"
    )
    if "zone" in dfs:
        unique_zones = set(dfs["zone"].unique())
        unique_zones.discard("all")
        if len(unique_zones) == 0:
            logging.info(f"{'Split into zones?':<{col1}} : no")
        else:
            logging.info(f"{'Split into zones?':<{col1}} : yes")
            logging.info(f"{'Number of zones':<{col1}} : {len(unique_zones)}")
            logging.info(f"{'Zones':<{col1}} : {', '.join(unique_zones)}")
    else:
        logging.info(f"{'Split into zones?':<{col1}} : no")
    if "region" in dfs:
        unique_regions = set(dfs["region"].unique())
        unique_regions.discard("all")
        if len(unique_regions) == 0:
            logging.info(f"{'Split into regions?':<{col1}} : no")
        else:
            logging.info(f"{'Split into regions?':<{col1}} : yes")
            logging.info(f"{'Number of regions':<{col1}} : {len(unique_regions)}")
            logging.info(f"{'Regions':<{col1}} : {', '.join(unique_regions)}")
    else:
        logging.info("{'Split into regions?':<{col1}} : no")
    if "plume_group" in dfs:
        unique_plumes = set(dfs["plume_group"].unique())
        unique_plumes.discard("all")
        unique_plumes.discard("undetermined")
        if len(unique_plumes) == 0:
            logging.info(f"{'Split into plume groups?':<{col1}} : no")
        else:
            logging.info(f"{'Split into plume groups?':<{col1}} : yes")
            logging.info(f"{'Number of plume groups':<{col1}} : {len(unique_plumes)}")
            logging.info(f"{'Plume groups':<{col1}} : {', '.join(unique_plumes)}")


def extract_amount(
    df: pd.DataFrame,
    c: str,
    p: str,
    cv: Optional[bool] = False,
    ind: int = -1,
) -> float:
    """
    Return the total co2 amount in grid nodes with the specified to phase and location
    at the latest recorded date (or at a specified index 'ind')
    """
    if cv:
        return df[df["containment"] == c]["amount"].iloc[ind]
    return df[(df["containment"] == c) & (df["phase"] == p)]["amount"].iloc[ind]


def sort_and_replace_nones(
    data_frame: pd.DataFrame,
):
    """
    Replaces empty zone and region fields with "all", and sorts the data frame
    """
    data_frame.replace(to_replace=[None], value="AAAAAll", inplace=True)
    data_frame.replace(to_replace=["total"], value="AAAAtotal", inplace=True)
    data_frame.sort_values(by=list(data_frame.columns[-1:1:-1]), inplace=True)
    data_frame.replace(to_replace=["AAAAtotal"], value="total", inplace=True)
    data_frame.replace(to_replace=["AAAAAll"], value="all", inplace=True)


def convert_data_frame(
    data_frame: pd.DataFrame,
    int_to_zone: Optional[List[Optional[str]]],
    int_to_region: Optional[List[Optional[str]]],
    calc_type_input: str,
    residual_trapping: bool,
) -> pd.DataFrame:
    """
    Convert output format to human-/Excel-readable state.
    """
    calc_type = _set_calc_type_from_input_string(calc_type_input)
    logging.info("\nMerge data rows for data frame")
    total_df = _merge_date_rows(
        data_frame[
            (data_frame["zone"] == "all")
            & (data_frame["region"] == "all")
            & (data_frame["plume_group"] == "all")
        ],
        calc_type,
        residual_trapping,
    )
    total_df["zone"] = ["all"] * total_df.shape[0]
    total_df["region"] = ["all"] * total_df.shape[0]
    total_df["plume_group"] = ["all"] * total_df.shape[0]

    zone_df = pd.DataFrame()
    if int_to_zone is not None:
        zones = [z for z in int_to_zone if z is not None]
        for z in zones:
            _df = _merge_date_rows(
                data_frame[
                    (data_frame["zone"] == z) & (data_frame["plume_group"] == "all")
                ],
                calc_type,
                residual_trapping,
            )
            _df["zone"] = [z] * _df.shape[0]
            zone_df = pd.concat([zone_df, _df])
        zone_df["region"] = ["all"] * zone_df.shape[0]
        zone_df["plume_group"] = ["all"] * zone_df.shape[0]

    region_df = pd.DataFrame()
    if int_to_region is not None:
        regions = [r for r in int_to_region if r is not None]
        for r in regions:
            _df = _merge_date_rows(
                data_frame[
                    (data_frame["region"] == r) & (data_frame["plume_group"] == "all")
                ],
                calc_type,
                residual_trapping,
            )
            _df["region"] = [r] * _df.shape[0]
            region_df = pd.concat([region_df, _df])
        region_df["zone"] = ["all"] * region_df.shape[0]
        region_df["plume_group"] = ["all"] * region_df.shape[0]

    plume_groups_df = pd.DataFrame()
    plume_groups = list(pd.unique(data_frame["plume_group"]))
    plume_groups = [name for name in plume_groups if name not in ["all"]]
    if len(plume_groups) > 0:
        for p in plume_groups:
            _df = _merge_date_rows(
                data_frame[
                    (data_frame["plume_group"] == p)
                    & (data_frame["zone"] == "all")
                    & (data_frame["region"] == "all")
                ],
                calc_type,
                residual_trapping,
            )
            _df["plume_group"] = [p] * _df.shape[0]
            plume_groups_df = pd.concat([plume_groups_df, _df])
        plume_groups_df["zone"] = ["all"] * plume_groups_df.shape[0]
        plume_groups_df["region"] = ["all"] * plume_groups_df.shape[0]

    combined_df = pd.concat([total_df, zone_df, region_df, plume_groups_df])
    return combined_df


def export_output_to_csv(
    out_dir: str,
    calc_type_input: str,
    data_frame: pd.DataFrame,
):
    """
    Exports the results to a csv file, named according to the calculation type
    (mass / cell_volume / actual_volume).
    """
    file_name = f"plume_{calc_type_input}.csv"
    logging.info("\nExport results to CSV file")
    logging.info(f"    - File name     : {file_name}")
    file_path = os.path.join(out_dir, file_name)
    logging.info(f"    - Path          : {file_path}")
    if not os.path.isabs(file_path):
        logging.info(f"    - Absolute path : {os.path.abspath(file_path)}")
    if os.path.isfile(file_path):
        logging.info("Output CSV file already exists => Will overwrite existing file")

    data_frame.to_csv(file_path, index=False)


def export_readable_output(
    df: pd.DataFrame,
    int_to_zone: Optional[List[Optional[str]]],
    int_to_region: Optional[List[Optional[str]]],
    out_dir: str,
    calc_type_input: str,
    residual_trapping: bool,
) -> None:
    """
    Exports the results to a more readable csv file than the standard output,
    both directly in a text editor and when loaded into Excel.
    Named according to the calculation type (mass / cell_volume / actual_volume)
    """
    file_name = f"plume_{calc_type_input}_summary_format.csv"
    logging.info(f"\nExport results to readable text file: {file_name}")
    file_path = os.path.join(out_dir, file_name)
    if os.path.isfile(file_path):
        logging.info(f"Output text file already exists. Overwriting: {file_path}")
    df, details = prepare_writing_details(df, calc_type_input, residual_trapping)

    zones = []
    regions = []
    plume_groups = []
    if int_to_zone is not None:
        zones += [zone for zone in int_to_zone if zone is not None]
    if int_to_region is not None:
        regions += [region for region in int_to_region if region is not None]

    all_plume_groups = list(pd.unique(df["plume_group"]))
    all_plume_groups = [name for name in all_plume_groups if name not in ["all"]]
    if len(all_plume_groups) > 0:
        plume_groups += all_plume_groups
    if "undetermined" in plume_groups:
        # To report undetermined last in the CSV-file:
        plume_groups.remove("undetermined")
        plume_groups.append("undetermined")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(details["type"])
        file.write(details["unit"])
        file.write(details["empty"])
        write_lines(file, df, "all", "all", "all", details)
        if len(zones) > 0:
            file.write(
                f"\n{'Filtered by zone:,':<{11 + details['width']}}"
                + details["blank"] * (details["num_cols"] - 2)
            )
            for zone in zones:
                write_lines(file, df, zone, "all", "all", details)
        if len(regions) > 0:
            file.write(
                f"\n{'Filtered by region:,':<{11 + details['width']}}"
                + details["blank"] * (details["num_cols"] - 2)
            )
            for region in regions:
                write_lines(file, df, "all", region, "all", details)
        if len(plume_groups) > 0:
            file.write(
                f"\n{'Filtered by plume gr.:,':<{11 + details['width']}}"
                + details["blank"] * (details["num_cols"] - 2)
            )
            for plume_group in plume_groups:
                write_lines(file, df, "all", "all", plume_group, details)


def find_width(num_decimals: int, max_value: Union[int, float]) -> int:
    """
    Use wider columns in the summary format if the numbers are large.
    """
    return int(max((12, num_decimals + 3 + np.floor(np.log(max_value) / np.log(10)))))


def prepare_writing_details(
    df: pd.DataFrame,
    calc_type: str,
    residual_trapping: bool,
) -> Tuple[pd.DataFrame, dict]:
    """
    Prepare headers and other information to be written in the summary file.
    """
    details: Dict = {
        "numeric": [
            c for c in df.columns if c not in ["date", "zone", "region", "plume_group"]
        ],
        "num_decimals": (
            3 if calc_type == "mass" else 6 if calc_type == "actual_volume" else 2
        ),
    }
    for column in details["numeric"]:
        df[column] /= 1e6
    width = find_width(details["num_decimals"], np.nanmax(df[details["numeric"]]))
    phase = (
        f",{'Free gas':>{width}},{'Trapped gas':>{width}},{'Dissolved':>{width}}"
        if residual_trapping
        else f",{'Gas':>{width}},{'Dissolved':>{width}}"
    )
    n_phase = 0 if calc_type == "cell_volume" else 3 if residual_trapping else 2

    details["num_phase"] = n_phase
    details["num_cols"] = 5 + 4 * n_phase
    details["blank"] = "," + " " * width

    dat = "\n      Date"
    tot = f",{'Total':>{width}}"
    con = f",{'Contained':>{width}}"
    out = f",{'Outside':>{width}}"
    haz = f",{'Hazardous':>{width}}"
    if calc_type == "cell_volume":
        details["over_header"] = details["blank"] * (details["num_cols"] - 2)
        details["header"] = dat + tot + con + out + haz
    else:
        details["over_header"] = (
            tot * (n_phase + 3) + con * n_phase + out * n_phase + haz * n_phase
        )
        details["header"] = dat + tot + phase + con + out + haz + phase * 3
    if calc_type == "mass":
        c_type = f" Calc type,{'Mass':>{width}}"
        unit = f"\n      Unit,{'Megatons':>{width}}," + " " * width
    elif calc_type == "actual_volume":
        c_type = f" Calc type,{'Volume':>{width}}"
        unit = f"\n      Unit,{'Cubic kilometers':>{max((17, width))}},"
        unit += " " * (width + min((0, width - 17)))
    else:
        c_type = f" Calc type,{'Cell volume':>{width}}"
        unit = f"\n      Unit,{'#cells (millions)':>{max((18, width))}},"
        unit += " " * (width + min((0, width - 18)))
    details["type"] = c_type + details["blank"] * (details["num_cols"] - 2)
    details["unit"] = unit + details["blank"] * (details["num_cols"] - 3)
    details["empty"] = "\n          " + details["blank"] * (details["num_cols"] - 1)
    details["width"] = width
    return df, details


def write_lines(
    file: TextIO,
    data_frame: pd.DataFrame,
    zone: str,
    region: str,
    plume_group: str,
    details: dict,
) -> None:
    """
    Write lines for the section of the containment output corresponding to the area
    defined by the specified region or zone or plume_group (or the total across all).
    """
    df = data_frame[
        (data_frame["zone"] == zone)
        & (data_frame["region"] == region)
        & (data_frame["plume_group"] == plume_group)
    ]
    max_name_length = 10 + details["width"]
    if zone == "all" and region == "all" and plume_group == "all":
        over_header = "\n          ," + " " * details["width"]
    elif region != "all":
        if len(region) > max_name_length:
            logging.warning(
                "Region name is long and will be cut off in the summary format!"
            )
            region = region[:max_name_length]
        over_header = f"\n{region:>10}," + " " * (
            details["width"] + min((0, 10 - len(region)))
        )
    elif zone != "all":
        if len(zone) > max_name_length:
            logging.warning(
                "Zone name is long and will be cut off in the summary format!"
            )
            zone = zone[:max_name_length]
        over_header = f"\n{zone:>10}," + " " * (
            details["width"] + min((0, 10 - len(zone)))
        )
    else:  # plume_group != "all"
        if len(plume_group) > max_name_length:
            logging.warning(
                "Plume group name is long and will be cut off in the summary format!"
            )
            plume_group = plume_group[:max_name_length]
        over_header = f"\n{plume_group:>10}," + " " * (
            details["width"] + min((0, 10 - len(plume_group)))
        )

    file.write(over_header + details["over_header"])
    file.write(details["header"])
    for lines_done in range(df.shape[0]):
        line = f"\n{df['date'].values[lines_done]}"
        values = df[details["numeric"]].values[lines_done]
        for value in values:
            line += f",{value:>{details['width']}.{details['num_decimals']}f}"
        file.write(line)
    file.write(details["empty"])


def main() -> None:
    """
    Takes input arguments and calculates total co2 mass or volume at each time
    step, divided into different phases and locations. Creates a data frame,
    then exports the data frame to a csv file.
    """
    arguments_processed = process_args()
    check_input(arguments_processed)
    zone_info = ZoneInfo(
        source=arguments_processed.zonefile,
        zranges=None,
        int_to_zone=None,
    )
    region_info = RegionInfo(
        source=arguments_processed.regionfile,
        int_to_region=None,  # set during calculation if source or property is given
        property_name=arguments_processed.region_property,
    )
    if zone_info.source is not None:
        zone_info.zranges = process_zonefile_if_yaml(zone_info.source)

    log_input_configuration(arguments_processed)

    if arguments_processed.config_file_inj_wells == "":
        injection_wells = []
    else:
        config = Configuration(arguments_processed.config_file_inj_wells)
        injection_wells = config.injection_wells

    data_frame = calculate_out_of_bounds_co2(
        arguments_processed.egrid,
        arguments_processed.unrst,
        arguments_processed.init,
        arguments_processed.calc_type_input,
        zone_info,
        region_info,
        arguments_processed.residual_trapping,
        injection_wells,
        arguments_processed.containment_polygon,
        arguments_processed.hazardous_polygon,
        arguments_processed.gas_molar_mass,
    )
    sort_and_replace_nones(data_frame)
    log_summary_of_results(data_frame, arguments_processed.calc_type_input)
    export_output_to_csv(
        arguments_processed.out_dir,
        arguments_processed.calc_type_input,
        data_frame,
    )
    if arguments_processed.readable_output:
        df_old_output = convert_data_frame(
            data_frame,
            zone_info.int_to_zone,
            region_info.int_to_region,
            arguments_processed.calc_type_input,
            arguments_processed.residual_trapping,
        )
        export_readable_output(
            df_old_output,
            zone_info.int_to_zone,
            region_info.int_to_region,
            arguments_processed.out_dir,
            arguments_processed.calc_type_input,
            arguments_processed.residual_trapping,
        )


if __name__ == "__main__":
    main()
