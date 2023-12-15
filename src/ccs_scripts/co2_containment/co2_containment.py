#!/usr/bin/env python
"""
Calculates the amount of CO2 inside and outside a given perimeter,
and separates the result per formation and phase (gas/dissolved).
Output is a table in CSV format.
"""
import argparse
import dataclasses
import os
import pathlib
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


# pylint: disable=too-many-arguments
def calculate_out_of_bounds_co2(
    grid_file: str,
    unrst_file: str,
    init_file: str,
    compact: bool,
    calc_type_input: str,
    file_containment_polygon: Optional[str] = None,
    file_hazardous_polygon: Optional[str] = None,
    zone_info: Optional[Dict] = None,
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
        zone_info (Dict): Dictionary containing path to zone-file and
            potentially zranges (if the zone-file is provided as a YAML-file
            with zones defined through intervals in depth)

    Returns:
        pd.DataFrame
    """
    co2_data = calculate_co2(
        grid_file, unrst_file, calc_type_input, init_file, zone_info
    )
    print("Done calculating CO2 data for all active grid cells")
    if file_containment_polygon is not None:
        containment_polygon = _read_polygon(file_containment_polygon)
    else:
        containment_polygon = None
    if file_hazardous_polygon is not None:
        hazardous_polygon = _read_polygon(file_hazardous_polygon)
    else:
        hazardous_polygon = None
    return calculate_from_co2_data(
        co2_data, containment_polygon, hazardous_polygon, compact, calc_type_input
    )


def calculate_from_co2_data(
    co2_data: Co2Data,
    containment_polygon: shapely.geometry.Polygon,
    hazardous_polygon: Union[shapely.geometry.Polygon, None],
    compact: bool,
    calc_type_input: str,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
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

    Returns:
        pd.DataFrame
    """
    calc_type = _set_calc_type_from_input_string(calc_type_input.lower())
    print("Calculate contained CO2 using input polygons")
    contained_co2 = calculate_co2_containment(
        co2_data, containment_polygon, hazardous_polygon, calc_type=calc_type
    )
    data_frame = _construct_containment_table(contained_co2)
    if compact:
        return data_frame
    if co2_data.zone is None:
        return _merge_date_rows(data_frame, calc_type)
    return {z: _merge_date_rows(g, calc_type) for z, g in data_frame.groupby("zone")}


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
    print("Merging data rows for data frame")
    data_frame = data_frame.drop("zone", axis=1)
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
        "--compact",
        help="Write the output to a single file as compact as possible.",
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
    paths = [
        "case",
        "out_dir",
        "egrid",
        "unrst",
        "init",
        "zonefile",
        "containment_polygon",
        "hazardous_polygon",
    ]

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


def process_zonefile_if_yaml(zone_info: Dict) -> Optional[Dict[str, List[int]]]:
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
    if zone_info["source"].split(".")[-1] in ["yml", "yaml"]:
        with open(zone_info["source"], "r", encoding="utf8") as stream:
            try:
                zfile = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
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
    else:
        return None


def export_output_to_csv(
    out_dir: str,
    calc_type_input: str,
    data_frame: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    zone_info: Optional[Dict[str, Any]] = None,
):
    """
    Exports the results to a csv file, named according to the calculation type
    (mass / cell_volume / actual_volume)
    """
    # pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_name = f"plume_{calc_type_input}"
    if isinstance(data_frame, dict):
        assert zone_info is not None
        keys = (
            data_frame.keys()
            if zone_info["zranges"] is None
            else list(zone_info["zranges"].keys())
        )
        combined_df = pd.DataFrame()
        for key, _df in zip(keys, data_frame.values()):
            _df["zone"] = [key] * _df.shape[0]
            combined_df = pd.concat([combined_df, _df])
        summed_part = combined_df.groupby("date").sum(numeric_only=True).reset_index()
        summed_part["zone"] = ["all"] * summed_part.shape[0]
        combined_df = pd.concat([summed_part, combined_df])
        combined_df.to_csv(os.path.join(out_dir, f"{out_name}.csv"), index=False)
    else:
        data_frame.to_csv(os.path.join(out_dir, f"{out_name}.csv"), index=False)


def main() -> None:
    """
    Takes input arguments and calculates total co2 mass or volume at each time
    step, divided into different phases and locations. Creates a data frame,
    then exports the data frame to a csv file.
    """
    arguments_processed = process_args()
    check_input(arguments_processed)
    if arguments_processed.zonefile is not None:
        zone_info = dict({"source": arguments_processed.zonefile, "zranges": None})
        zone_info["zranges"] = process_zonefile_if_yaml(zone_info)
    else:
        zone_info = None
    data_frame = calculate_out_of_bounds_co2(
        arguments_processed.egrid,
        arguments_processed.unrst,
        arguments_processed.init,
        arguments_processed.compact,
        arguments_processed.calc_type_input,
        arguments_processed.containment_polygon,
        arguments_processed.hazardous_polygon,
        zone_info,
    )
    export_output_to_csv(
        arguments_processed.out_dir,
        arguments_processed.calc_type_input,
        data_frame,
        zone_info,
    )


if __name__ == "__main__":
    main()
