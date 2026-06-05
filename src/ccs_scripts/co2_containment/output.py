"""Output formatting and export for CO2 containment calculations."""

import logging
import os
from typing import Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import pandas as pd

from ccs_scripts.co2_containment.input import (
    CalculationType,
    _set_calc_type_from_input_string,
)
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import format_warning


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
    locations = ["contained", "outside", "nogo"]
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
        df_phases = list(pd.unique(data_frame["phase"]))
        df_phases = [name for name in df_phases if name not in ["all"]]
        phases = ["free_gas", "trapped_gas"] if residual_trapping else ["gas"]
        phases += ["dissolved_water"]
        phases += ["dissolved_oil"] if "dissolved_oil" in df_phases else []
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

    col1 = 30
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
        value = extract_amount(df_subset, "total", "dissolved_water")
        percent = 100.0 * value / total if total > 0.0 else 0.0
        logging.info(
            f"{'End state dissolved in water':<{col1}} : "
            f"{value:{n}.1f}  ={percent:>5.1f} %"
        )
        if "dissolved_oil" in list(df_subset["phase"]):
            value = extract_amount(df_subset, "total", "dissolved_oil")
            percent = 100.0 * value / total if total > 0.0 else 0.0
            logging.info(
                f"{'End state dissolved in oil':<{col1}} : "
                f"{value:{n}.1f}  ={percent:>5.1f} %"
            )
    value = extract_amount(df_subset, "contained", "total", cell_volume)
    percent = 100.0 * value / total if total > 0.0 else 0.0
    logging.info(
        f"{'End state contained':<{col1}} : {value:{n}.1f}  ={percent:>5.1f} %"
    )
    value = extract_amount(df_subset, "outside", "total", cell_volume)
    percent = 100.0 * value / total if total > 0.0 else 0.0
    logging.info(f"{'End state outside':<{col1}} : {value:{n}.1f}  ={percent:>5.1f} %")
    value = extract_amount(df_subset, "nogo", "total", cell_volume)
    percent = 100.0 * value / total if total > 0.0 else 0.0
    logging.info(f"{'End state no-go':<{col1}} : {value:{n}.1f}  ={percent:>5.1f} %")
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
    # Keep length of column names below <= 11 to be sure of no alignment issues
    phase_names = ["Free gas", "Trapped gas"] if residual_trapping else ["Gas"]
    phase_names += ["Dis. water"]
    phase_names += (
        ["Dis. oil"] if any("dissolved_oil" in col for col in df.columns) else []
    )
    phase = "," + ",".join(f"{name:>{width}}" for name in phase_names)
    n_phase = 0 if calc_type == "cell_volume" else len(phase_names)
    details["num_phase"] = n_phase
    details["num_cols"] = 5 + 4 * n_phase
    details["blank"] = "," + " " * width

    dat = "\n      Date"
    tot = f",{'Total':>{width}}"
    con = f",{'Contained':>{width}}"
    out = f",{'Outside':>{width}}"
    nog = f",{'No-go':>{width}}"
    if calc_type == "cell_volume":
        details["over_header"] = details["blank"] * (details["num_cols"] - 2)
        details["header"] = dat + tot + con + out + nog
    else:
        details["over_header"] = (
            tot * (n_phase + 3) + con * n_phase + out * n_phase + nog * n_phase
        )
        details["header"] = dat + tot + phase + con + out + nog + phase * 3
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
            warning_text = (
                "Region name is long and will be cut off in the summary format!"
            )
            logging.warning(format_warning(warning_text))
            region = region[:max_name_length]
        over_header = f"\n{region:>10}," + " " * (
            details["width"] + min((0, 10 - len(region)))
        )
    elif zone != "all":
        if len(zone) > max_name_length:
            warning_text = (
                "Zone name is long and will be cut off in the summary format!"
            )
            logging.warning(format_warning(warning_text))
            zone = zone[:max_name_length]
        over_header = f"\n{zone:>10}," + " " * (
            details["width"] + min((0, 10 - len(zone)))
        )
    else:  # plume_group != "all"
        if len(plume_group) > max_name_length:
            warning_text = (
                "Plume group name is long and will be cut off in the summary format!"
            )
            logging.warning(format_warning(warning_text))
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


def export_results(
    containment_data: pd.DataFrame,
    calc_type_input: str,
    out_dir: str,
    int_to_zone: Optional[List[Optional[str]]],
    int_to_region: Optional[List[Optional[str]]],
    residual_trapping: bool,
    readable_output: bool,
) -> None:
    """
    Exports the results to a csv file, named according to the calculation type
    (mass / cell_volume / actual_volume), and also in a more readable format.
    """
    timer = Timer()
    sort_and_replace_nones(containment_data)
    log_summary_of_results(containment_data, calc_type_input)
    timer.start("export_results")
    export_output_to_csv(
        out_dir,
        calc_type_input,
        containment_data,
    )
    if readable_output:
        df_old_output = convert_data_frame(
            containment_data,
            int_to_zone,
            int_to_region,
            calc_type_input,
            residual_trapping,
        )
        export_readable_output(
            df_old_output,
            int_to_zone,
            int_to_region,
            out_dir,
            calc_type_input,
            residual_trapping,
        )
    timer.stop("export_results")
