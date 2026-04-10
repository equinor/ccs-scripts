import logging
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from resdata.grid import Grid
from resdata.resfile import ResdataFile

THRESHOLD_GAS = 1e-16
THRESHOLD_DISSOLVED = 1e-16  # Used also in co2_calculation to avoid numerical issues


def format_warning(txt: Union[str, Exception]) -> str:
    return f"\x1b[37;45m\x1b[1m{txt}\x1b[0m"


def format_error(txt: Union[str, Exception]) -> str:
    return f"\x1b[37;41m\x1b[1m{txt}\x1b[0m"


def try_prop(unrst: ResdataFile, prop_name: str):
    """
    Function to determine if a property (prop_name) is part of a ResdataFile (unrst)

    Args:
      unrst (ResdataFile): ResdataFile to fetch property names from
      prop_name (str): The property name to be searched in unrst

    Returns:
      str if prop_names exists in unrst, None otherwise

    """
    try:
        prop = unrst[prop_name]
    except KeyError:
        prop = None
    return prop


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


def test_for_soil(props: dict):
    if "SGAS" not in props or "SWAT" not in props:
        return None
    tol = 1e-6
    sgas = props["SGAS"]
    swat = props["SWAT"]
    soil = {}
    max_val = float("-inf")
    for date in sgas:
        soil[date] = np.maximum(0.0, 1.0 - sgas[date] - swat[date])
        max_soil_date = soil[date].max()
        if max_soil_date > max_val:
            max_val = max_soil_date
    return soil if max_val > tol else None


def _read_props(
    unrst: ResdataFile,
    prop_names: List,
) -> dict:
    """
    Reads the properties in prop_names from a ResdataFile named unrst

    Args:
      unrst (ResdataFile): ResdataFile to read prop_names from
      prop_names (List): List with property names to be read

    Returns:
      dict
    """
    active_props = {}
    for p in prop_names:
        result = try_prop(unrst, p)
        if result is not None:
            active_props.update({p: result})
    return active_props


def fetch_properties(
    unrst: ResdataFile, props_to_extract: List
) -> Tuple[Dict[str, Dict[str, List[np.ndarray]]], List[str]]:
    """
    Fetches the properties in props_to_extract from a ResdataFile
    named unrst

    Args:
      unrst (ResdataFile): ResdataFile to fetch props_to_extract from
      props_to_extract: List with property names to be fetched

    Returns:
      Tuple

    """
    dates = [d.strftime("%Y%m%d") for d in unrst.report_dates]
    props = _read_props(unrst, props_to_extract)
    test_unrst_consistency(dates, props)
    props = {
        p: {d[1]: props[p][d[0]].numpy_copy() for d in enumerate(dates)} for p in props
    }
    if "SOIL" not in props:
        soil = test_for_soil(props)
        if soil is not None:
            props["SOIL"] = soil
            logging.info(
                "Oil Saturation (SOIL) not found as property"
                "\nHowever, as SGAS + SWAT is not 1 everywhere"
                "\nThe remaining saturation is assumed to be SOIL."
                "\nThis propery has been computed"
            )
        else:
            logging.info(
                "Oil Saturation is zero everywhere."
                "\n Therefore, two-phase scenario is assumed."
            )
    logging.info(
        "Done reading properties from file"
        "\nRelevant properties extracted:"
        f"\n    {', '.join(list(props.keys()))}\n"
    )
    return props, dates


def test_unrst_consistency(dates: List, props: dict) -> None:
    """
    Checks consistency between UNRST dates and properties.

    Args:
        dates (list): List of dates
        props (list): List with lists of properties at each date

    """

    lengths = {name: len(values) for name, values in props.items()}

    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            format_error(
                "Inconsistent UNRST properties lengths: "
                + ", ".join(f"{k}={v}" for k, v in lengths.items())
            )
        )

    n_time_steps = unique_lengths.pop()
    n_dates = len(dates)

    if n_time_steps != n_dates:
        raise ValueError(
            format_error(
                f"Mismatch between number of dates ({n_dates}) "
                f"and number of timesteps for properties({n_time_steps})"
            )
        )


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
    gas_less = np.logical_and.reduce([np.abs(sgas[s]) < THRESHOLD_GAS for s in sgas])
    if dissolved_prop is not None:
        gas_less &= np.logical_and.reduce(
            [np.abs(dissolved_prop[a]) < THRESHOLD_DISSOLVED for a in dissolved_prop]
        )
    return gas_less


def reduce_properties(
    properties: Dict[str, Dict[str, List[np.ndarray]]], keep_idx: np.ndarray
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


def find_active_and_gasless_cells(
    grid: Grid, properties, do_logging: bool = False, ignore_dissolved: bool = False
):
    act_num = grid.export_actnum().numpy_copy()
    active = np.where(act_num > 0)[0]

    if ignore_dissolved:
        gasless = identify_gas_less_cells(properties["SGAS"])
    else:
        dissolved_prop = None
        if is_subset(["SGAS", "AMFS"], list(properties.keys())):
            dissolved_prop = "AMFS"
        elif is_subset(["SGAS", "AMFG"], list(properties.keys())):
            dissolved_prop = "AMFG"
        elif is_subset(["SGAS", "XMF2"], list(properties.keys())):
            dissolved_prop = "XMF2"

        if dissolved_prop is not None:
            gasless = identify_gas_less_cells(
                properties["SGAS"], properties[dissolved_prop]
            )
        else:
            error_text = (
                "CO2 containment calculation failed. Cannot find required properties "
            )
            error_text += "SGAS+AMFG, SGAS+XMF2 or SGAS+AMFS"
            raise RuntimeError(format_error(error_text))

    if do_logging:
        logging.info(f"Number of grid cells                    : {len(act_num):>10}")
        logging.info(f"Number of active grid cells             : {len(active):>10}")
        logging.info(
            f"Number of active non-gasless grid cells : {len(active[~gasless]):>10}"
        )

    return active, gasless


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
