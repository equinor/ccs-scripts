"""Output assembly and reporting for plume extent calculations."""

import logging
import string
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ccs_scripts.co2_plume_extent.config import Calculation, CalculationType, Configuration
from ccs_scripts.co2_plume_tracking.utils import InjectionWellData, sort_well_names


def find_output_file(output: str, case: str):
    if output is None:
        p = Path(case).parents[2]
        p2 = p / "share" / "results" / "tables" / "plume_extent.csv"
        return str(p2)
    else:
        return output


def log_results(df: pd.DataFrame) -> None:
    dfs = df.sort_values("date")
    col_width = 1 + max(31, max([len(c) for c in df]))
    logging.info("\nSummary of results:")
    logging.info("===================")
    logging.info(
        f"Number of dates {' ' * (col_width - 5)}: {len(dfs['date'].unique()):>11}"
    )
    logging.info(f"First date      {' ' * (col_width - 5)}: {dfs['date'].iloc[0]:>11}")
    logging.info(f"Last date       {' ' * (col_width - 5)}: {dfs['date'].iloc[-1]:>11}")

    for col in df.drop("date", axis=1).columns:
        logging.info(f"End state {col:<{col_width}} : {dfs[col].iloc[-1]:>11.1f}")


def log_results_detailed(df: pd.DataFrame):
    dist_cols = [col for col in df.columns if col != "date"]
    letter_names = list(string.ascii_uppercase)[: len(dist_cols)]
    col_mapping = dict(zip(dist_cols, letter_names))
    col_mapping["date"] = "date"
    df = df.rename(columns=col_mapping)
    pd.options.display.float_format = "{:.1f}".format
    for col in df.columns:
        if col != "date":
            df[col] = df[col].round(1)

    logging.info("\nDetailed summary of results:")
    logging.info("============================")
    logging.info("Columns:")
    for key, value in col_mapping.items():
        if key != "date":
            logging.info(f"  {value}: {key}")

    def custom_format(x):
        if x == 0.0:
            return "-"
        else:
            return f"{x:.1f}"

    formatters = {
        col: custom_format if col != "date" else "{: >10}".format for col in df.columns
    }

    logging.info("\nResults:")
    logging.info(df.to_string(index=False, formatters=formatters))


def _find_dates(all_results: List[Tuple[dict, Optional[dict], Optional[str]]]):
    one_dict = all_results[0][0][next(iter(all_results[0][0]))]
    one_array = one_dict[next(iter(one_dict))]
    dates = [[date] for (date, _) in one_array]
    return dates


def _find_column_name(
    single_config: Calculation,
    n_calculations: int,
    calculation_number: int,
):
    if single_config.type == CalculationType.PLUME_EXTENT:
        col = "MAX_"
    elif single_config.type in (CalculationType.POINT, CalculationType.LINE):
        col = "MIN_"
    else:
        col = "?"

    if single_config.column_name != "":
        col = col + single_config.column_name
    else:
        calc_number = "" if n_calculations == 1 else str(calculation_number)
        col = col + f"{single_config.type.name.upper()}{calc_number}"

    return col


def collect_results_into_dataframe(
    all_results: List[Tuple[dict, Optional[dict], Optional[str]]],
    config: Configuration,
    injection_wells: Optional[List[InjectionWellData]] = None,
) -> pd.DataFrame:
    dates = _find_dates(all_results)
    df = pd.DataFrame.from_records(dates, columns=["date"])
    for i, (result, single_config) in enumerate(
        zip(all_results, config.distance_calculations), 1
    ):
        gas_results, dissolved_results, _ = result

        col = _find_column_name(single_config, len(config.distance_calculations), i)

        if injection_wells is not None and config.do_plume_tracking:
            gas_results = sort_well_names(gas_results, injection_wells)

        for group_str, results in gas_results.items():
            for well_name, result2 in results.items():
                full_col_name = col + "_GAS"
                if group_str != "ALL":
                    full_col_name += "_PLUME_" + group_str
                if well_name != "ALL" and well_name != "WELL":
                    full_col_name += "_FROM_INJ_" + well_name
                gas_df = pd.DataFrame.from_records(
                    result2, columns=["date", full_col_name]
                )
                df = pd.merge(df, gas_df, on="date")
        if dissolved_results is not None:
            if injection_wells is not None and config.do_plume_tracking:
                dissolved_results_sorted = sort_well_names(
                    dissolved_results, injection_wells
                )
            else:
                dissolved_results_sorted = dissolved_results
            for group_str, results in dissolved_results_sorted.items():
                for well_name, result2 in results.items():
                    if result2 is not None:
                        full_col_name = col + "_DISSOLVED"
                        if group_str != "ALL":
                            full_col_name += "_PLUME_" + group_str
                        if well_name != "ALL" and well_name != "WELL":
                            full_col_name += "_FROM_INJ_" + well_name
                        dissolved_df = pd.DataFrame.from_records(
                            result2, columns=["date", full_col_name]
                        )
                        df = pd.merge(df, dissolved_df, on="date")
    return df
