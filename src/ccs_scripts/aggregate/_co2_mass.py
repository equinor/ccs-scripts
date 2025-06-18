import copy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import xtgeo
from resdata.resfile import ResdataFile
from resfo._unformatted.write import unformatted_write
from xtgeo.io._file import FileWrapper

from ccs_scripts.aggregate._config import CO2MassSettings
from ccs_scripts.co2_containment.co2_calculation import (
    Co2Data,
    Co2DataAtTimeStep,
    Scenario,
)
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import fetch_properties, identify_gas_less_cells, is_subset

CO2_MASS_PNAME = "CO2Mass"

# pylint: disable=invalid-name,too-many-instance-attributes


class MapName(Enum):
    MASS_TOT = "co2_mass_total"
    MASSDISW = "co2_mass_dissolved_water_phase"
    MASSDISO = "co2_mass_dissolved_oil_phase"
    MASS_GAS = "co2_mass_gas_phase"
    MASSTGAS = "co2_mass_trapped_gas_phase"
    MASSFGAS = "co2_mass_free_gas_phase"


class PropertyGridOutput(TypedDict):
    data: np.ndarray
    unrst_path: str
    egrid_path: str


def _get_gasless(properties: Dict[str, Dict[str, List[np.ndarray]]]) -> np.ndarray:
    """
    Identifies global index for grid cells without CO2 based on Gas Saturation (SGAS)
    and Mole Fraction of Gas in dissolved phase (AMFG/XMF2)

    Args:
        properties (Dict) : Properties that will be used to compute CO2 mass

    Returns:
        np.ndarray
    """
    if is_subset(["SGAS", "AMFS"], list(properties.keys())):
        gasless = identify_gas_less_cells(properties["SGAS"], properties["AMFS"])
    elif is_subset(["SGAS", "AMFG"], list(properties.keys())):
        gasless = identify_gas_less_cells(properties["SGAS"], properties["AMFG"])
    elif is_subset(["SGAS", "XMF2"], list(properties.keys())):
        gasless = identify_gas_less_cells(properties["SGAS"], properties["XMF2"])
    else:
        error_text = (
            "CO2 containment calculation failed. "
            "Cannot find required properties SGAS+AMFG, SGAS+XMF2 or SGAS+AMFS"
        )
        raise RuntimeError(error_text)
    return gasless


def translate_co2data_to_property(
    co2_data: Co2Data,
    grid_file: str,
    co2_mass_settings: CO2MassSettings,
    grid_out_dir: str,
    properties_to_extract: List[str],
    dates_idx: List[int],
) -> List[Optional[str]]:
    """
    Convert CO2 data into 3D GridProperty

    Args:
        co2_data (Co2Data): Information of the amount of CO2 at each cell in
                            each time step
        grid_file (str): Path to EGRID-file
        co2_mass_settings (CO2MassSettings): Settings from config file for calculation
                                             of CO2 mass maps.
        grid_out_dir (str): Path to store the produced 3D GridProperties.
        properties_to_extract (List): Names of the properties to be extracted

    Returns:
        List[List[xtgeo.GridProperty]]

    """
    timer = Timer()
    timer.start("translate_co2data_to_property")
    gas_idxs = _get_gas_idxs(co2_mass_settings.unrst_source, properties_to_extract)
    maps = co2_mass_settings.maps
    if maps is None:
        maps = []
    elif isinstance(maps, str):
        maps = [maps]
    maps = [map_name.lower() for map_name in maps]

    mass_data_template: Dict[str, List[Any]] = {
        "unrst_path": [],
        "unrst_kw": [],
        "egrid_path": [],
        "egrid_kw": [],
    }
    total_mass_data = copy.deepcopy(mass_data_template)
    dissolved_water_mass_data = copy.deepcopy(mass_data_template)
    dissolved_oil_mass_data = copy.deepcopy(mass_data_template)
    free_mass_data = copy.deepcopy(mass_data_template)
    free_gas_mass_data = copy.deepcopy(mass_data_template)
    trapped_gas_mass_data = copy.deepcopy(mass_data_template)

    unrst_data = ResdataFile(co2_mass_settings.unrst_source)
    grid_data = ResdataFile(grid_file)
    grid_pf = xtgeo.grid_from_file(grid_file)
    n_act_cells = len(grid_pf.actnum_indices)
    store_all = "all" in maps or len(maps) == 0

    custom_egrid = _create_custom_egrid_kw(grid_data)

    for date_idx, co2_at_date in zip(dates_idx, co2_data.data_list):
        mass_as_grid = _convert_to_grid(
            co2_at_date, gas_idxs, n_act_cells, grid_out_dir
        )
        logihead_array = np.array([x for x in unrst_data["LOGIHEAD"][date_idx]])
        if store_all or "total_co2" in maps:
            total_mass_data["unrst_kw"].extend(
                [
                    ("SEQNUM  ", [date_idx]),
                    ("INTEHEAD", unrst_data["INTEHEAD"][date_idx].numpyView()),
                    ("LOGIHEAD", logihead_array),
                    ("MASS_TOT", mass_as_grid["MASS_TOT"]["data"]),
                ]
            )
            if (
                mass_as_grid["MASS_TOT"]["unrst_path"]
                not in total_mass_data["unrst_path"]
            ):
                total_mass_data["unrst_path"].append(
                    mass_as_grid["MASS_TOT"]["unrst_path"]
                )
                total_mass_data["egrid_path"].append(
                    mass_as_grid["MASS_TOT"]["egrid_path"]
                )
                total_mass_data["egrid_kw"].extend(custom_egrid)
        if store_all or "dissolved_water_co2" in maps:
            dissolved_water_mass_data["unrst_kw"].extend(
                [
                    ("SEQNUM  ", [date_idx]),
                    ("INTEHEAD", unrst_data["INTEHEAD"][date_idx].numpyView()),
                    ("LOGIHEAD", logihead_array),
                    ("MASSDISW", mass_as_grid["MASSDISW"]["data"]),
                ]
            )
            if (
                mass_as_grid["MASSDISW"]["unrst_path"]
                not in dissolved_water_mass_data["unrst_path"]
            ):
                dissolved_water_mass_data["unrst_path"].append(
                    mass_as_grid["MASSDISW"]["unrst_path"]
                )
                dissolved_water_mass_data["egrid_path"].append(
                    mass_as_grid["MASSDISW"]["egrid_path"]
                )
                dissolved_water_mass_data["egrid_kw"].extend(custom_egrid)
        if (
            store_all or "dissolved_oil_co2" in maps
        ) and co2_data.scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
            dissolved_oil_mass_data["unrst_kw"].extend(
                [
                    ("SEQNUM  ", [date_idx]),
                    ("INTEHEAD", unrst_data["INTEHEAD"][date_idx].numpyView()),
                    ("LOGIHEAD", logihead_array),
                    ("MASSDISO", mass_as_grid["MASSDISO"]["data"]),
                ]
            )
            if (
                mass_as_grid["MASSDISO"]["unrst_path"]
                not in dissolved_oil_mass_data["unrst_path"]
            ):
                dissolved_oil_mass_data["unrst_path"].append(
                    mass_as_grid["MASSDISO"]["unrst_path"]
                )
                dissolved_oil_mass_data["egrid_path"].append(
                    mass_as_grid["MASSDISO"]["egrid_path"]
                )
                dissolved_oil_mass_data["egrid_kw"].extend(custom_egrid)
        if (
            store_all or "free_co2" in maps
        ) and not co2_mass_settings.residual_trapping:
            free_mass_data["unrst_kw"].extend(
                [
                    ("SEQNUM  ", [date_idx]),
                    ("INTEHEAD", unrst_data["INTEHEAD"][date_idx].numpyView()),
                    ("LOGIHEAD", logihead_array),
                    ("MASS_GAS", mass_as_grid["MASS_GAS"]["data"]),
                ]
            )
            if (
                mass_as_grid["MASS_GAS"]["unrst_path"]
                not in free_mass_data["unrst_path"]
            ):
                free_mass_data["unrst_path"].append(
                    mass_as_grid["MASS_GAS"]["unrst_path"]
                )
                free_mass_data["egrid_path"].append(
                    mass_as_grid["MASS_GAS"]["egrid_path"]
                )
                free_mass_data["egrid_kw"].extend(custom_egrid)
        if (store_all or "free_co2" in maps) and co2_mass_settings.residual_trapping:
            free_gas_mass_data["unrst_kw"].extend(
                [
                    ("SEQNUM  ", [date_idx]),
                    ("INTEHEAD", unrst_data["INTEHEAD"][date_idx].numpyView()),
                    ("LOGIHEAD", logihead_array),
                    ("MASSFGAS", mass_as_grid["MASSFGAS"]["data"]),
                ]
            )
            if (
                mass_as_grid["MASSFGAS"]["unrst_path"]
                not in free_gas_mass_data["unrst_path"]
            ):
                free_gas_mass_data["unrst_path"].append(
                    mass_as_grid["MASSFGAS"]["unrst_path"]
                )
                free_gas_mass_data["egrid_path"].append(
                    mass_as_grid["MASSFGAS"]["egrid_path"]
                )
                free_gas_mass_data["egrid_kw"].extend(custom_egrid)
            trapped_gas_mass_data["unrst_kw"].extend(
                [
                    ("SEQNUM  ", [date_idx]),
                    ("INTEHEAD", unrst_data["INTEHEAD"][date_idx].numpyView()),
                    ("LOGIHEAD", logihead_array),
                    ("MASSTGAS", mass_as_grid["MASSTGAS"]["data"]),
                ]
            )
            if (
                mass_as_grid["MASSTGAS"]["unrst_path"]
                not in trapped_gas_mass_data["unrst_path"]
            ):
                trapped_gas_mass_data["unrst_path"].append(
                    mass_as_grid["MASSTGAS"]["unrst_path"]
                )
                trapped_gas_mass_data["egrid_path"].append(
                    mass_as_grid["MASSTGAS"]["egrid_path"]
                )
                trapped_gas_mass_data["egrid_kw"].extend(custom_egrid)
    out = [
        _export_unrst_and_kw_data(free_mass_data),
        _export_unrst_and_kw_data(dissolved_water_mass_data),
        _export_unrst_and_kw_data(dissolved_oil_mass_data),
        _export_unrst_and_kw_data(total_mass_data),
        _export_unrst_and_kw_data(free_gas_mass_data),
        _export_unrst_and_kw_data(trapped_gas_mass_data),
    ]
    timer.stop("translate_co2data_to_property")
    return out


def _create_custom_egrid_kw(
    grid_data: ResdataFile,
) -> List[Tuple[str, Union[List[int], np.ndarray]]]:
    """
    Create the custom list of keywords to export the EGRID file for
    each co2_mass property
    """
    kw_sequence = [
        "FILEHEAD",
        "GRIDUNIT",
        "GDORIENT",
        "GRIDHEAD",
        "COORD   ",
        "ZCORN   ",
        "ACTNUM  ",
        "ENDGRID ",
        "NNCHEAD ",
        "NNC1    ",
        "NNC2    ",
    ]
    mandatory_kws = [
        "FILEHEAD",
        "GRIDUNIT",
        "GRIDHEAD",
        "COORD   ",
        "ZCORN   ",
        "ENDGRID ",
    ]
    custom_egrid = []
    for kw in kw_sequence:
        try:
            val = grid_data[kw.rstrip()][0].numpyView()
            custom_egrid.append((kw, val))
        except (AttributeError, ValueError, KeyError):
            try:
                val = grid_data[kw.rstrip()][0]
                custom_egrid.append((kw, val))
            except KeyError as err:
                if kw in mandatory_kws:
                    raise KeyError(
                        f"Mandatory key '{kw}' is missing in grid_data"
                    ) from err
                pass
    return custom_egrid


def _export_unrst_and_kw_data(mass_data: Dict[str, List[Any]]) -> Optional[str]:
    """
    Exports the grid with the property at different time steps as well as
    the path where the file is located

    Args:
        mass_data (Dict[str,List[Any]]): A dict with
        the information that feeds the 3d grid properties

        Returns:
             Optional[str]
    """
    if len(mass_data["unrst_path"]) > 0:
        outfile_wrapper = FileWrapper(mass_data["unrst_path"][0], mode="rb")
        with open(outfile_wrapper.file, "wb") as stream:
            unformatted_write(stream, mass_data["unrst_kw"])
        grid_outfile_wrapper = FileWrapper(mass_data["egrid_path"][0], mode="rb")
        with open(grid_outfile_wrapper.file, "wb") as stream:
            unformatted_write(stream, mass_data["egrid_kw"])
        return mass_data["unrst_path"][0]
    else:
        return None


def _get_gas_idxs(
    unrst_file: str,
    properties_to_extract: List[str],
) -> np.ndarray:
    """
    Gets the global index of cells with CO2

    Args:
        unrst_file (str): Path to UNRST-file
        properties_to_extract (List): Names of the properties to be extracted

    Returns:
        np.ndarray

    """
    unrst = ResdataFile(unrst_file)
    properties, _ = fetch_properties(unrst, properties_to_extract)
    gasless = _get_gasless(properties)
    gas_idxs = np.array([index for index, value in enumerate(gasless) if not value])
    return gas_idxs


def _convert_to_grid(
    co2_at_date: Co2DataAtTimeStep,
    gas_idxs: np.ndarray,
    n_act_cells: int,
    grid_out_dir: str,
) -> Dict[str, PropertyGridOutput]:
    """
    Store CO2DataAtTimeStep for a property in a 3DGridProperties object

    Args:
        co2_at_date (Co2DataAtTimeStep):       Amount of CO2 per phase at each cell
                                               at each time step
        gas_idxs (np.ndarray):                 Global index of cells with CO2
        n_act_cells (int):                     Number of active cells in EGRID
        grid_out_dir (str):                    Path to store the produced
                                               3D GridProperties

    Returns:
        Dict[str, xtgeo.GridProperty]
    """
    mass_grid_output = {}
    for mass, name in zip(
        [
            co2_at_date.total_mass(),
            co2_at_date.dis_water_phase,
            co2_at_date.dis_oil_phase,
            co2_at_date.gas_phase,
            co2_at_date.trapped_gas_phase,
            co2_at_date.free_gas_phase,
        ],
        [
            "MASS_TOT",
            "MASSDISW",
            "MASSDISO",
            "MASS_GAS",
            "MASSTGAS",
            "MASSFGAS",
        ],
    ):
        mass_array = np.zeros(n_act_cells, dtype=mass.dtype)
        mass_array[gas_idxs] = mass
        prop_grid_output: PropertyGridOutput = {
            "data": mass_array,
            "unrst_path": grid_out_dir + "/" + str(MapName[name].value) + ".UNRST",
            "egrid_path": grid_out_dir + "/" + str(MapName[name].value) + ".EGRID",
        }
        mass_grid_output[name] = prop_grid_output
    return mass_grid_output
