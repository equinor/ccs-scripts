"""Output assembly and reporting for plume tracking calculations."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ccs_scripts.co2_plume_tracking.utils import (
    InjectionWellData,
    assemble_plume_groups_into_dict,
    sort_well_names,
)


def _log_results(
    df: pd.DataFrame,
) -> None:
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


def _find_output_file(output: Optional[str], case: str):
    if output is None:
        p = Path(case).parents[2]
        p2 = p / "share" / "results" / "tables" / "plume_tracking.csv"
        return str(p2)
    return output


def _collect_results_into_dataframe(
    report_dates: List[datetime],
    pg_prop_gas: List[List[str]],
    pg_prop_dissolved: Optional[List[List[str]]],
    dissolved_prop_key: Optional[str],
    injection_wells: List[InjectionWellData],
) -> pd.DataFrame:
    dates = [[d.strftime("%Y-%m-%d")] for d in report_dates]
    df = pd.DataFrame.from_records(dates, columns=["date"])

    for prop_key, pg_prop in zip(
        ["SGAS", dissolved_prop_key], [pg_prop_gas, pg_prop_dissolved]
    ):
        if pg_prop is None or prop_key is None:
            continue
        results = {}
        for i, p in enumerate(pg_prop):
            pg_dict = assemble_plume_groups_into_dict(p)
            for group_name, indices in pg_dict.items():
                if group_name not in results:
                    results[group_name] = np.zeros(
                        shape=(len(dates)),
                        dtype=int,
                    )
                results[group_name][i] = len(indices)
        results_sorted = sort_well_names(results, injection_wells)
        results_sorted = {
            prop_key + "_" + key: value for key, value in results_sorted.items()
        }

        prop_df = pd.DataFrame(results_sorted)
        df = pd.concat([df, prop_df], axis=1)

    return df


def export_results(
    output_csv: Optional[str],
    case: str,
    dates: List[datetime],
    pg_prop_gas: List[List[str]],
    pg_prop_dissolved: Optional[List[List[str]]],
    dissolved_prop_key: Optional[str],
    injection_wells: List[InjectionWellData],
):
    output_file = _find_output_file(output_csv, case)

    df = _collect_results_into_dataframe(
        dates,
        pg_prop_gas,
        pg_prop_dissolved,
        dissolved_prop_key,
        injection_wells,
    )
    _log_results(df)

    logging.info("\nExport results to CSV file")
    logging.info(f"    - File path: {output_file}")
    if os.path.isfile(output_file):
        logging.info("Output CSV file already exists => Will overwrite existing file")
    df.to_csv(output_file, index=False)
