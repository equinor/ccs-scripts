#!/usr/bin/env python
"""
Script calculating the area extent of the plume depending on which map / date
are present in the share/results/maps folder
"""

################################################################################
#
# Created by : Jorge Sicacha (NR), Oct 2022
# Modified by: Floriane Mortier (fmmo), Nov 2022 - To fit FMU workflow
#
################################################################################

import argparse
import glob
import logging
import os
import pathlib
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xtgeo

from ccs_scripts.utils.utils import (
    format_error,
    format_warning,
    log_input_banner,
    replace_default_ert_dummies,
    setup_log_configuration,
    str_to_bool,
)

xtgeo_logger = logging.getLogger("xtgeo")
xtgeo_logger.setLevel(logging.WARNING)

GAS_PHASE = "gas_phase"
AQUEOUS_PHASE = "dissolved_phase"
OIL_PHASE = "oil_phase"


def _phase_of_property(prop: str) -> Optional[str]:
    """
    The phase CO2 is in for a property (lower case), or None if it is not one of
    the properties for CO2.

    XMF is taken as the aqueous phase, which is what it is when there is no oil.
    With oil, XMF is the oil phase, and AMF is the aqueous phase: see
    _warn_about_xmf().
    """
    if re.fullmatch(r"sgas|ymfs|ymf\d+", prop):
        return GAS_PHASE
    if re.fullmatch(r"amfg|amfs|amf\d+|xmf\d+", prop):
        return AQUEOUS_PHASE
    if prop == "xmfs":  # Cirrus: always the oil phase
        return OIL_PHASE
    return None


def _make_parser():
    parser = argparse.ArgumentParser(description="Calculate plume area")
    parser.add_argument(
        "input",
        help="Path to maps created through grid3d_aggregate_map",
        metavar="<INPUT>",
    )
    parser.add_argument(
        "--output_csv",
        help="Path to output CSV file",
        default=None,
        metavar="<OUTPUT_CSV>",
    )
    parser.add_argument(
        "--no_logging",
        help="Skip print of detailed information during execution of script",
        type=str_to_bool,
        nargs="?",
        const=True,
        metavar="<NO_LOGGING>",
    )
    parser.add_argument(
        "--debug",
        help="Enable print of debugging data during execution of script. "
        "Normally not necessary for most users.",
        type=str_to_bool,
        nargs="?",
        const=True,
        metavar="<DEBUG>",
    )

    return parser


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
    Finds plume area for each formation and year for a given rskey, for instance
    SGAS (gas phase) or AMFG/XMF2 (dissolved phase). The plume areas are found
    using data from surface files (.gri).
    """
    logging.info(f"Calculating plume area for           : {rskey}")

    if path[-1] != "/":
        path = path + "/"
    out = _find_formations(path, rskey)
    if not out:
        return None

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
            node_indices = np.ma.nonzero(mysurf.values)  # Indexes of the existing nodes
            use_nodes = set(list(tuple(zip(node_indices[0], node_indices[1]))))
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
    replace_default_ert_dummies(args, false_list=["no_logging", "debug"], none_list=[])
    setup_log_configuration(args)

    input_path = args.input
    output_path = args.output_csv

    if not os.path.isdir(input_path):
        text = f"Input surface directory not found: {input_path}"
        raise FileNotFoundError(format_error(text))

    return input_path, output_path


def _log_input_configuration(input_path: str, output_path: str) -> None:
    log_input_banner(
        __file__,
        "Plume area calculations",
        is_dev_version=True,
    )

    logging.info(f"\nInput path  : {input_path}")
    logging.info(f"Output path : {output_path}\n")


def _find_map_properties(search_path: str) -> Dict[str, str]:
    """
    The properties for CO2 that have max_<property> maps in the folder, with the
    phase each stands for. Gas phase first, then aqueous phase, then oil phase.
    """
    found = set()
    for file in glob.glob(search_path + "*--max_*--*.gri"):
        parts = pathlib.Path(file).stem.split("--")
        if len(parts) >= 3 and parts[1].lower().startswith("max_"):
            found.add(parts[1][len("max_") :].lower())
    order = {GAS_PHASE: 0, AQUEOUS_PHASE: 1, OIL_PHASE: 2}
    phases = {prop: _phase_of_property(prop) for prop in found}
    kept = {prop: phase for prop, phase in phases.items() if phase is not None}
    return dict(sorted(kept.items(), key=lambda item: (order[item[1]], item[0])))


def _warn_about_xmf(properties: Dict[str, str]) -> None:
    """
    Plume area cannot tell whether there is oil, so XMF is taken as the aqueous
    phase. This is always said, as it is wrong when there is oil.
    """
    xmf = [p for p in properties if re.fullmatch(r"xmf\d+", p)]
    if not xmf:
        return
    names = ", ".join(p.upper() for p in xmf)
    amf = [p for p in properties if re.fullmatch(r"amf\d+", p)]
    text = (
        f"WARNING: {names} is taken as CO2 in the aqueous phase "
        f"({AQUEOUS_PHASE}_{xmf[0]}). If the case has oil, XMF is the oil phase "
        "and that column is the area of CO2 in the oil phase. "
    )
    if amf:
        text += (
            f"Maps for {', '.join(p.upper() for p in amf)} exist: "
            "use those for the aqueous phase."
        )
    else:
        text += (
            "The aqueous phase is then given by AMF, which has to be written to "
            "the restart file (RPTRST)."
        )
    logging.warning(format_warning(text))


def _warn_about_sgas(properties: Dict[str, str]) -> None:
    if "sgas" in properties and any(
        p != "sgas" and phase == GAS_PHASE for p, phase in properties.items()
    ):
        text = (
            "WARNING: Maps for SGAS and for the mole fraction of CO2 in the gas "
            "phase both exist. In depleted fields SGAS also counts hydrocarbon gas, "
            "so its area is not the area of CO2."
        )
        logging.warning(format_warning(text))


def _warn_about_amfg(properties: Dict[str, str]) -> None:
    if "amfg" in properties and "amfs" in properties:
        text = (
            "WARNING: Maps for AMFG and AMFS both exist. In Cirrus cases with "
            "native gas, AMFG is the native gas dissolved in water, and AMFS is "
            "the CO2 dissolved in water: the area of CO2 is given by AMFS."
        )
        logging.warning(format_warning(text))


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
    dfs = df.sort_values("date")
    logging.info("\nSummary of results:")
    logging.info("===================")
    logging.info(f"Number of dates : {len(dfs['date'].unique()):>11}")
    logging.info(f"First date      : {dfs['date'].iloc[0]:>11}")
    logging.info(f"Last date       : {dfs['date'].iloc[-1]:>11}")

    columns = [c for c in dfs if c != "date"]
    n1 = max(len(c) for c in columns) if len(columns) > 0 else 5
    df_subset = dfs.drop("date", axis=1)
    n2 = len(f"{df_subset.max().max():.1f}") if len(columns) > 0 else 5
    logging.info("End state plume area:")
    for c in columns:
        logging.info(f"    * {c:<{n1 + 1}}: {dfs[c].iloc[-1]:>{n2 + 1}.1f}")


def main():
    """
    Reads directory of input surface files (.gri) and calculates plume area per formatin and
    year for SGAS/YMF (gas phase) and AMFG/XMF2 (dissolved in water phase) and XMFS (dissolved
    in oil phase. The columns are named <formation>_<phase>_<property>.Collects the results
    into a CSV file.
    """
    input_path, output_path = _read_args()

    if output_path is None:
        p = pathlib.Path("share") / "results" / "tables" / "plume_area.csv"
        output_path = str(p)
    _log_input_configuration(input_path, output_path)
    if not input_path.endswith("/"):
        input_path += "/"

    properties = _find_map_properties(input_path)
    if not properties:
        text = (
            f"No max_<property> maps for CO2 found in {input_path}. Looking for "
            "SGAS, YMF*, AMF* and XMF*, made with grid3d_aggregate_map."
        )
        logging.error(format_error(text))
        return 1
    logging.info(
        "Properties found: "
        + ", ".join(f"{p.upper()} ({phase})" for p, phase in properties.items())
        + "\n"
    )
    _warn_about_xmf(properties)
    _warn_about_sgas(properties)
    _warn_about_amfg(properties)

    df = None
    for prop, phase in properties.items():
        results = calculate_plume_area(input_path, prop)
        if not results:
            continue
        logging.info(f"\nDone calculating plume areas for {prop.upper()} ({phase}).")
        df_prop = _convert_to_data_frame(results, f"{phase}_{prop}")
        df = df_prop if df is None else pd.merge(df, df_prop)

    if df is None:
        logging.error(format_error("No plume areas could be calculated."))
        return 1

    _log_results(df)

    logging.info("\nExport results to CSV file")
    logging.info(f"    - File path: {output_path}")
    if os.path.isfile(output_path):
        logging.info("Output CSV file already exists => Will overwrite existing file")
    df.to_csv(output_path, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
