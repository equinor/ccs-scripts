import os
import tempfile
from typing import Dict, List, Tuple, Optional

import numpy as np
import xtgeo
from resdata.resfile import ResdataFile
from resfo._unformatted.write import unformatted_write
from xtgeo.io._file import FileWrapper
from enum import Enum

from ccs_scripts.aggregate._config import CO2MassSettings
from ccs_scripts.co2_containment.co2_calculation import (
    Co2Data,
    Co2DataAtTimeStep,
    _fetch_properties,
    _identify_gas_less_cells,
    _is_subset,
)

CO2_MASS_PNAME = "CO2Mass"

# pylint: disable=invalid-name,too-many-instance-attributes

class MapName(Enum):
    MASS_TOT = "co2_mass_total"
    MASS_AQU = "co2_mass_aqu_phase"
    MASS_GAS = "co2_mass_gas_phase"
    MASSTGAS = "co2_mass_trapped_gas_phase"
    MASSFGAS = "co2_mass_free_gas_phase"


def _get_gasless(properties: Dict[str, Dict[str, List[np.ndarray]]]) -> np.ndarray:
    """
    Identifies global index for grid cells without CO2 based on Gas Saturation (SGAS)
    and Mole Fraction of Gas in aqueous phase (AMFG/XMF2)

    Args:
        properties (Dict) : Properties that will be used to compute CO2 mass

    Returns:
        np.ndarray
    """
    if _is_subset(["SGAS", "AMFG"], list(properties.keys())):
        gasless = _identify_gas_less_cells(properties["SGAS"], properties["AMFG"])
    elif _is_subset(["SGAS", "XMF2"], list(properties.keys())):
        gasless = _identify_gas_less_cells(properties["SGAS"], properties["XMF2"])
    else:
        error_text = (
            "CO2 containment calculation failed. "
            "Cannot find required properties SGAS+AMFG or SGAS+XMF2."
        )
        raise RuntimeError(error_text)
    return gasless


def translate_co2data_to_property(
    co2_data: Co2Data,
    grid_file: str,
    co2_mass_settings: CO2MassSettings,
    properties_to_extract: List[str],
    grid_out_dir: Optional[str] = None,
) -> List[List[xtgeo.GridProperty]]:
    """
    Convert CO2 data into 3D GridProperty

    Args:
        co2_data (Co2Data): Information of the amount of CO2 at each cell in
                            each time step
        grid_file (str): Path to EGRID-file
        co2_mass_settings (CO2MassSettings): Settings from config file for calculation
                                             of CO2 mass maps.
        properties_to_extract (List): Names of the properties to be extracted
        grid_out_dir (str): Path to store the produced 3D GridProperties.

    Returns:
        List[List[xtgeo.GridProperty]]

    """
    idxs = _get_dimensions_and_triplets(
        co2_mass_settings.unrst_source, properties_to_extract
    )

    # Setting up the grid folder to store the gridproperties
    if grid_out_dir:
        if not os.path.exists(grid_out_dir):
            os.makedirs(grid_out_dir)
    else:
        grid_out_dir = tempfile.mkdtemp()
    maps = co2_mass_settings.maps
    if maps is None:
        maps = []
    elif isinstance(maps, str):
        maps = [maps]
    maps = [map_name.lower() for map_name in maps]

    total_mass_list = []
    dissolved_mass_list = []
    free_mass_list = []
    free_gas_mass_list = []
    trapped_gas_mass_list = []
    total_mass_kw_list = []
    dissolved_mass_kw_list = []
    free_mass_kw_list = []
    free_gas_mass_kw_list = []
    trapped_gas_mass_kw_list = []

    unrst_data = ResdataFile(co2_mass_settings.unrst_source)

    store_all = "all" in maps or len(maps) == 0

    for i , co2_at_date in enumerate(co2_data.data_list):
        mass_as_grids = _convert_to_grid(co2_at_date, idxs, grid_file, grid_out_dir)
        logihead_array = np.array([x for x in unrst_data["LOGIHEAD"][i]])
        if store_all or "total_co2" in maps:
            total_mass_kw_list.extend([
                ("SEQNUM  ", [i]),
                ("INTEHEAD", unrst_data["INTEHEAD"][i].numpyView()),
                ("LOGIHEAD", logihead_array),
                ("MASS_TOT", mass_as_grids["MASS_TOT"]["data"])
            ])
            total_mass_list.append(mass_as_grids["MASS_TOT"]["path"])
        if store_all or "dissolved_co2" in maps:
            dissolved_mass_kw_list.extend([
                ("SEQNUM  ", [i]),
                ("INTEHEAD", unrst_data["INTEHEAD"][i].numpyView()),
                ("LOGIHEAD", logihead_array),
                ("MASS_AQU", mass_as_grids["MASS_AQU"]["data"])
            ])
            dissolved_mass_list.append(mass_as_grids["MASS_AQU"]["path"])
        if (
            store_all or "free_co2" in maps
        ) and not co2_mass_settings.residual_trapping:
            free_mass_kw_list.extend([
                ("SEQNUM  ", [i]),
                ("INTEHEAD", unrst_data["INTEHEAD"][i].numpyView()),
                ("LOGIHEAD", logihead_array),
                ("MASS_GAS", mass_as_grids["MASS_GAS"]["data"])
            ])
            free_mass_list.append(mass_as_grids["MASS_GAS"]["path"])
        if (store_all or "free_co2" in maps) and co2_mass_settings.residual_trapping:
            free_gas_mass_kw_list.extend([
                ("SEQNUM  ", [i]),
                ("INTEHEAD", unrst_data["INTEHEAD"][i].numpyView()),
                ("LOGIHEAD", logihead_array),
                ("MASSFGAS", mass_as_grids["MASSFGAS"]["data"])
            ])
            free_gas_mass_list.append(mass_as_grids["MASSFGAS"]["path"])
            trapped_gas_mass_kw_list.extend([
                ("SEQNUM  ", [i]),
                ("INTEHEAD", unrst_data["INTEHEAD"][i].numpyView()),
                ("LOGIHEAD", logihead_array),
                ("MASSTGAS", mass_as_grids["MASSTGAS"]["data"])
            ])
            trapped_gas_mass_list.append(mass_as_grids["MASSTGAS"]["path"])

    return [
        _export_and_simplify_kw_list(free_mass_kw_list,free_mass_list),
        _export_and_simplify_kw_list(dissolved_mass_kw_list,dissolved_mass_list),
        _export_and_simplify_kw_list(total_mass_kw_list,total_mass_list),
        _export_and_simplify_kw_list(free_gas_mass_kw_list,free_gas_mass_list),
        _export_and_simplify_kw_list(trapped_gas_mass_kw_list,trapped_gas_mass_list),
    ]


def _export_and_simplify_kw_list(
        kwlist,
        outputlist
):
    if len(outputlist)>0:
        outfile_wrapper = FileWrapper(outputlist[0], mode="rb")
        with open(outfile_wrapper.file, "wb") as stream:
            unformatted_write(stream,kwlist)
        return outputlist[0]
    else:
        return []

def _get_dimensions_and_triplets(
    unrst_file: str,
    properties_to_extract: List[str],
) -> Tuple[Tuple[int, int, int], List[Tuple[int, int, int]]]:
    """
    Gets the size of the 3D grid and (X,Y,Z) position of cells with CO2

    Args:
        grid_file (str): Path to EGRID-file
        unrst_file (str): Path to UNRST-file
        properties_to_extract (List): Names of the properties to be extracted

    Returns:
        Tuple[Tuple[int, int, int], List[Tuple[int, int, int]]]:

    """
    unrst = ResdataFile(unrst_file)
    properties, _ = _fetch_properties(unrst, properties_to_extract)
    gasless = _get_gasless(properties)
    idxs = np.array([index for index, value in enumerate(gasless) if not value])
    return idxs


def _convert_to_grid(
    co2_at_date: Co2DataAtTimeStep,
    idxs: List[int],
    grid_file: str,
    grid_out_dir: str
) -> Dict[str, xtgeo.GridProperty]:
    """
    Store the CO2 mass arrays in 3D GridProperties

    Args:
        co2_at_date (Co2DataAtTimeStep):       Amount of CO2 per phase at each cell
                                               at each time step
        dimensions (Tuple[int,int,int]):       Size of the 3D grid
        triplets (List[Tuple[int, int, int]]): List of triplets with the (X,Y,Z)
                                               position of cells with CO2

    Returns:
        Dict[str, xtgeo.GridProperty]
    """
    mass_arrays = {}
    for mass, name in zip(
        [
            co2_at_date.total_mass(),
            co2_at_date.aqu_phase,
            co2_at_date.gas_phase,
            co2_at_date.trapped_gas_phase,
            co2_at_date.free_gas_phase,
        ],
        [
            "MASS_TOT",
            "MASS_AQU",
            "MASS_GAS",
            "MASSTGAS",
            "MASSFGAS",
        ],
    ):
        grid_pf = xtgeo.grid_from_file(grid_file)
        act_cells =len(grid_pf.actnum_indices)
        mass_array = np.zeros(act_cells, dtype=mass.dtype)
        mass_array[idxs] = mass
        mass_arrays[name] = {'data':mass_array, 'path':  grid_out_dir + "/" + str(MapName[name].value) + ".UNRST"}
    return mass_arrays
