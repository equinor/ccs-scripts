import logging
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from resdata.grid import Grid
from resdata.resfile import ResdataFile

TRESHOLD_GAS = 1e-16
TRESHOLD_DISSOLVED = 1e-16


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
    props_att = {p: try_prop(unrst, p) for p in prop_names}
    act_prop_names = [k for k in prop_names if props_att[k] is not None]
    act_props = {k: props_att[k] for k in act_prop_names}
    return act_props


def fetch_properties(
    unrst: ResdataFile, properties_to_extract: List
) -> Tuple[Dict[str, Dict[str, List[np.ndarray]]], List[str]]:
    """
    Fetches the properties in properties_to_extract from a ResdataFile
    named unrst

    Args:
      unrst (ResdataFile): ResdataFile to fetch properties_to_extract from
      properties_to_extract: List with property names to be fetched

    Returns:
      Tuple

    """
    dates = [d.strftime("%Y%m%d") for d in unrst.report_dates]
    properties = _read_props(unrst, properties_to_extract)
    properties = {
        p: {d[1]: properties[p][d[0]].numpy_copy() for d in enumerate(dates)}
        for p in properties
    }
    return properties, dates


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
    gas_less = np.logical_and.reduce([np.abs(sgas[s]) < TRESHOLD_GAS for s in sgas])
    if dissolved_prop is not None:
        gas_less &= np.logical_and.reduce(
            [np.abs(dissolved_prop[a]) < TRESHOLD_DISSOLVED for a in dissolved_prop]
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
            raise RuntimeError(error_text)

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
            logging.error(exc)
            sys.exit(1)
