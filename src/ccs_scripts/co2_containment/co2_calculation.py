# pylint: disable-msg=too-many-lines
"""Methods for CO2 containment calculations"""
import copy
import logging
from dataclasses import dataclass, fields, make_dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import xtgeo
from resdata.grid import Grid
from resdata.resfile import ResdataFile

from ccs_scripts.utils.utils import Timer

DEFAULT_CO2_MOLAR_MASS = 44.0
DEFAULT_WATER_MOLAR_MASS = 18.0
TRESHOLD_GAS = 1e-16
TRESHOLD_DISSOLVED = 1e-16
PROPERTIES_NEEDED_PFLOTRAN = ["DGAS", "DWAT", "AMFG", "YMFG"]
PROPERTIES_NEEDED_ECLIPSE = ["BGAS", "BWAT", "XMF2", "YMF2"]

RELEVANT_PROPERTIES = [
    "RPORV",
    "PORV",
    "SGAS",
    "DGAS",
    "BGAS",
    "DWAT",
    "BWAT",
    "AMFG",
    "YMFG",
    "XMFG",
    "AMFS",
    "YMFS",
    "XMFS",
    "AMFW",
    "YMFW",
    "XMFW",
]

source_data_: List[Tuple[str, Any, None]] = [
    ("x_coord", np.ndarray, None),
    ("y_coord", np.ndarray, None),
    ("DATES", List[str], None),
    ("VOL", Optional[Dict[str, np.ndarray]], None),
    ("SWAT", Optional[Dict[str, np.ndarray]], None),
    ("SGAS", Optional[Dict[str, np.ndarray]], None),
    ("SGSTRAND", Optional[Dict[str, np.ndarray]], None),
    ("SGTRH", Optional[Dict[str, np.ndarray]], None),
    ("RPORV", Optional[Dict[str, np.ndarray]], None),
    ("PORV", Optional[Dict[str, np.ndarray]], None),
    ("AMFG", Optional[Dict[str, np.ndarray]], None),
    ("YMFG", Optional[Dict[str, np.ndarray]], None),
    ("XMFG", Optional[Dict[str, np.ndarray]], None),
    ("DWAT", Optional[Dict[str, np.ndarray]], None),
    ("DGAS", Optional[Dict[str, np.ndarray]], None),
    ("DOIL", Optional[Dict[str, np.ndarray]], None),
    ("BWAT", Optional[Dict[str, np.ndarray]], None),
    ("BGAS", Optional[Dict[str, np.ndarray]], None),
    ("AMFS", Optional[Dict[str, np.ndarray]], None),
    ("YMFS", Optional[Dict[str, np.ndarray]], None),
    ("XMFS", Optional[Dict[str, np.ndarray]], None),
    ("AMFW", Optional[Dict[str, np.ndarray]], None),
    ("YMFW", Optional[Dict[str, np.ndarray]], None),
    ("XMFW", Optional[Dict[str, np.ndarray]], None),
    ("zone", Optional[np.ndarray], None),
    ("region", Optional[np.ndarray], None),
]


class CalculationType(Enum):
    """
    Which type of CO2 calculation is made
    """

    MASS = 0
    CELL_VOLUME = 1
    ACTUAL_VOLUME = 2

    @classmethod
    def check_for_key(cls, key: str):
        """
        Check if key in enum
        """
        if key not in cls.__members__:
            error_text = "Illegal calculation type: " + key
            error_text += "\nValid options:"
            for calc_type in CalculationType:
                error_text += "\n  * " + calc_type.name.lower()
            error_text += "\nExiting"
            raise ValueError(error_text)


class Scenario(Enum):
    """
    Which scenario is CO2 amounts calculated in
    """

    AQUIFER = 0
    DEPLETED_GAS_FIELD = 1
    DEPLETED_OIL_GAS_FIELD = 2


@dataclass
class Co2DataAtTimeStep:
    """
    Dataclass with amount of co2 for each phase (dissolved/gas/undefined)
    at a given time step.

    Args:
      date (str): The time step
      dis_water_phase (np.ndarray): The amount of CO2 in dissolved phase
      gas_phase (np.ndarray): The amount of CO2 in gaseous phase
      dis_oil_phase (np.ndarray): The amount of CO2 in oil phase
      volume_coverage (np.ndarray): The volume of a cell (specific of
                                    calc_type_input = volume_extent)
      trapped_gas_phase (np.ndarray): The amount of CO2 in trapped/stranded gas phase
      free_gas_phase (np.ndarray): The amount of CO2 in free gas phase
    """

    date: str
    dis_water_phase: np.ndarray
    gas_phase: np.ndarray
    dis_oil_phase: np.ndarray
    volume_coverage: np.ndarray
    trapped_gas_phase: np.ndarray
    free_gas_phase: np.ndarray

    def total_mass(self) -> np.ndarray:
        """
        Computes total mass as the sum of gas in dissolved and gas
        phase.
        """
        return self.dis_water_phase + self.gas_phase + self.dis_oil_phase


@dataclass
class Co2Data:
    """
    Dataclass with amount of CO2 at (x,y) coordinates

    Args:
      x_coord (np.ndarray): x coordinates
      y_coord (np.ndarray): y coordinates
      data_list (List): List with CO2 amounts calculated
                        at multiple time steps
      units (Literal): Units of the calculated amount of CO2
      zone (np.ndarray): Zone information
      region (np.ndarray): Region information

    """

    x_coord: np.ndarray
    y_coord: np.ndarray
    data_list: List[Co2DataAtTimeStep]
    units: Literal["kg", "tons", "m3"]
    scenario: Scenario
    zone: Optional[np.ndarray] = None
    region: Optional[np.ndarray] = None


@dataclass
class ZoneInfo:
    source: Optional[str]
    zranges: Optional[Dict[str, List[int]]]
    int_to_zone: Optional[List[Optional[str]]]


@dataclass
class RegionInfo:
    source: Optional[str]
    int_to_region: Optional[List[Optional[str]]]
    property_name: Optional[str]


def _detect_eclipse_mole_fraction_props(
    unrst_file: str,
    properties_to_extract: List,
    current_source_data: List[Tuple[str, Any, None]],
):
    """
    Detects which and how many components are there in Eclipse data

    Args:
        unrst_file (str): Path to UNSRT file
        properties_to_extract (List): List of current properties to extract
        current_source_data (List): List with properties to edit
    """
    unrst = ResdataFile(unrst_file)
    suffix_count = 1
    while True and suffix_count < 50:
        tmp_x = _try_prop(unrst, "XMF" + str(suffix_count))
        tmp_y = _try_prop(unrst, "YMF" + str(suffix_count))
        if tmp_x is None and tmp_y is None:
            break
        elif (tmp_x is None) != (tmp_y is None):
            raise ValueError(
                "Error: Number of components with XMF property differ from "
                "the number of components with YMF"
            )
        else:
            current_source_data.extend(
                [
                    (name + str(suffix_count), Optional[Dict[str, np.ndarray]], None)
                    for name in ["XMF", "YMF"]
                ]
            )
            properties_to_extract.extend(
                [name + str(suffix_count) for name in ["XMF", "YMF"]]
            )
        suffix_count += 1
    return current_source_data, properties_to_extract


def _n_components(active_props: List):
    """
    Detects how many components are there in vapor phase

    Args:
        active_props (List): List of active properties

    Returns
        int with the number of components
    """
    xmf_suffixes = [int(item[3:]) for item in active_props if item.startswith("XMF")]
    # Find the max suffix
    max_xmf_suffix = max(xmf_suffixes)

    ymf_suffixes = [int(item[3:]) for item in active_props if item.startswith("YMF")]
    # Find the max suffix
    max_ymf_suffix = max(ymf_suffixes)

    if max_xmf_suffix != max_ymf_suffix:
        raise ValueError(
            "Error: Number of components with XMF property differ from "
            "the number of components with YMF"
        )
    return max_xmf_suffix


def _try_prop(unrst: ResdataFile, prop_name: str):
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
    props_att = {p: _try_prop(unrst, p) for p in prop_names}
    act_prop_names = [k for k in prop_names if props_att[k] is not None]
    act_props = {k: props_att[k] for k in act_prop_names}
    return act_props


def _fetch_properties(
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


def _identify_gas_less_cells(sgas: dict, dissolved_prop: dict) -> np.ndarray:
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
    gas_less &= np.logical_and.reduce(
        [np.abs(dissolved_prop[a]) < TRESHOLD_DISSOLVED for a in dissolved_prop]
    )
    return gas_less


def _reduce_properties(
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


def _is_subset(first: List[str], second: List[str]) -> bool:
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


# NBNB-AS: Move this ?
def find_active_and_gasless_cells(grid: Grid, properties, do_logging: bool = False):
    act_num = grid.export_actnum().numpy_copy()
    active = np.where(act_num > 0)[0]
    if _is_subset(["SGAS", "AMFS"], list(properties.keys())):
        gasless = _identify_gas_less_cells(properties["SGAS"], properties["AMFS"])
    elif _is_subset(["SGAS", "AMFG"], list(properties.keys())):
        gasless = _identify_gas_less_cells(properties["SGAS"], properties["AMFG"])
    elif _is_subset(["SGAS", "XMF2"], list(properties.keys())):
        gasless = _identify_gas_less_cells(properties["SGAS"], properties["XMF2"])
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


# pylint: disable=too-many-arguments
def _extract_source_data(
    grid_file: str,
    unrst_file: str,
    source_data_: List[Tuple[str, Any, None]],
    properties_to_extract: List[str],
    zone_info: ZoneInfo,
    region_info: RegionInfo,
    init_file: Optional[str] = None,
):
    # pylint: disable=too-many-locals, too-many-statements
    """Extracts the properties in properties_to_extract from Grid files

    Args:
      grid_file (str): Path to EGRID-file
      unrst_file (str): Path to UNRST-file
      properties_to_extract (List): Names of the properties to be extracted
      init_file (str): Path to INIT-file
      zone_info (ZoneInfo): Zone information
      region_info (Dict): Region information

    Returns:
      SourceData

    """
    logging.info("Start extracting source data")
    grid = Grid(grid_file)
    unrst = ResdataFile(unrst_file)
    try:
        init = ResdataFile(init_file)
    except Exception:
        init = None
        logging.info("No INIT-file loaded")
    properties, dates = _fetch_properties(unrst, properties_to_extract)
    logging.info("Done fetching properties")

    active, gasless = find_active_and_gasless_cells(grid, properties, True)
    global_active_idx = active[~gasless]

    properties_reduced = _reduce_properties(properties, ~gasless)
    # Tuple with (x,y,z) for each cell:
    xyz = [grid.get_xyz(global_index=a) for a in global_active_idx]
    cells_x = np.array([coord[0] for coord in xyz])
    cells_y = np.array([coord[1] for coord in xyz])

    zone = _process_zones(zone_info, grid, grid_file, global_active_idx)
    region = _process_regions(region_info, grid, grid_file, init, active, gasless)
    vol0 = [grid.cell_volume(global_index=x) for x in global_active_idx]
    properties_reduced["VOL"] = {d: vol0 for d in dates}
    if init is not None:
        try:
            porv = init["PORV"]
            properties_reduced["PORV"] = {
                d: porv[0].numpy_copy()[global_active_idx] for d in dates
            }
        except KeyError:
            pass
    SourceData = make_dataclass("SourceData", source_data_)
    source_data = SourceData(
        cells_x,
        cells_y,
        dates,
        **dict(properties_reduced.items()),
        zone=zone,
        region=region,
    )
    logging.info("Done extracting source data\n")
    return source_data


def _check_grid_dimensions(
    roff_file: str,
    grid_file: str,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    grid_shape = (nx, ny, nz)
    roff_grid = xtgeo.gridproperty_from_file(roff_file)
    roff_shape = roff_grid.values.shape
    if roff_shape != grid_shape:
        err = f"Inconsistent grid dimensions {roff_shape} from file {roff_file}"
        err += f" and {grid_shape} from file {grid_file}."
        raise ValueError(err)


def _process_zones(
    zone_info: ZoneInfo,
    grid: Grid,
    grid_file: str,
    global_active_idx: np.ndarray,
) -> Optional[np.ndarray]:
    zone = None
    if zone_info.source is None:
        logging.info("No zone info specified")
    else:
        logging.info("Using zone info")
        if zone_info.zranges is not None:
            zone_array = np.zeros(
                (grid.get_nx(), grid.get_ny(), grid.get_nz()), dtype=int
            )
            zonevals = [int(x) for x in range(len(zone_info.zranges))]
            zone_info.int_to_zone = [f"Zone_{x}" for x in range(len(zonevals))]
            for zv, zr, zn in zip(
                zonevals,
                list(zone_info.zranges.values()),
                zone_info.zranges.keys(),
            ):
                zone_array[:, :, zr[0] - 1 : zr[1]] = zv
                zone_info.int_to_zone[zv] = zn
            zone = zone_array.flatten(order="F")[global_active_idx]
        else:
            xtg_grid = xtgeo.grid_from_file(grid_file)
            _check_grid_dimensions(
                zone_info.source,
                grid_file,
                xtg_grid.ncol,
                xtg_grid.nrow,
                xtg_grid.nlay,
            )
            zone = xtgeo.gridproperty_from_file(zone_info.source, grid=xtg_grid)
            try:
                zone_name_dict = zone.codes
                zone_values = list(zone_name_dict.keys())
            except AttributeError:
                zone_name_dict = {}
                zone_values = []
            zone = zone.values.data.flatten(order="F")
            zonevals = np.unique(zone)
            intvals = np.array(zonevals, dtype=int)
            if np.sum(intvals == zonevals) != len(zonevals):
                logging.info(
                    "Warning: Grid provided in zone file contains non-integer values. "
                    "This might cause problems with the calculations for "
                    "containment in different zones."
                )
            zone_info.int_to_zone = [None] * (np.max(intvals) + 1)
            for zv in intvals:
                if zv >= 0:
                    if zv in zone_values:
                        zone_info.int_to_zone[zv] = zone_name_dict[zv]
                    else:
                        zone_info.int_to_zone[zv] = f"Zone_{zv}"
                        logging.info(
                            f"Value {zv} in roff-grid not found in Codes."
                            f" Using generic zone name Zone_{zv}."
                        )
                else:
                    logging.info("Ignoring negative value in grid from zone file.")
            zone = np.array(zone[global_active_idx], dtype=int)
    return zone


def _process_regions(
    region_info: RegionInfo,
    grid: Grid,
    grid_file: str,
    init: Optional[ResdataFile],
    active: np.ndarray,
    gasless: np.ndarray,
) -> Optional[np.ndarray]:
    region = None
    if region_info.source is not None:
        logging.info("Using regions info")
        xtg_grid = xtgeo.grid_from_file(grid_file)
        _check_grid_dimensions(
            region_info.source,
            grid_file,
            xtg_grid.ncol,
            xtg_grid.nrow,
            xtg_grid.nlay,
        )
        region = xtgeo.gridproperty_from_file(region_info.source, grid=xtg_grid)
        try:
            region_name_dict = region.codes
            region_values = list(region_name_dict.keys())
        except AttributeError:
            region_name_dict = {}
            region_values = []
        region = region.values.data.flatten(order="F")
        regvals = np.unique(region)
        intvals = np.array(regvals, dtype=int)
        if np.sum(intvals == regvals) != len(regvals):
            logging.info(
                "Warning: Grid provided in region file contains non-integer values. "
                "This might cause problems with the calculations for "
                "containment in different regions."
            )
        region_info.int_to_region = [None] * (np.max(intvals) + 1)
        for rv in intvals:
            if rv >= 0:
                if rv in region_values:
                    region_info.int_to_region[rv] = region_name_dict[rv]
                else:
                    region_info.int_to_region[rv] = f"Region_{rv}"
                    logging.info(
                        f"Value {rv} in roff-grid not found in Codes."
                        f" Using generic region name Region_{rv}."
                    )
            else:
                logging.info("Ignoring negative value in grid from region file.")
        region = np.array(region[active[~gasless]], dtype=int)
    elif region_info.property_name is not None:
        if init is None:
            logging.info("No INIT-file to use for region information.")
            region = None
            region_info.int_to_region = None
        else:
            try:
                logging.info(
                    f"Try reading region information ({region_info.property_name}"
                    f" property) from INIT-file."
                )
                region = np.array(init[region_info.property_name][0], dtype=int)
                if region.shape[0] == grid.get_nx() * grid.get_ny() * grid.get_nz():
                    region = region[active]
                regvals = np.unique(region)
                region_info.int_to_region = [None] * (np.max(regvals) + 1)
                for rv in regvals:
                    if rv >= 0:
                        region_info.int_to_region[rv] = f"Region_{rv}"
                    else:
                        logging.info(
                            f"Ignoring negative value in {region_info.property_name}."
                        )
                logging.info("Region information successfully read from INIT-file")
                region = region[~gasless]
            except KeyError:
                logging.info("Region information not found in INIT-file.")
                region = None
                region_info.int_to_region = None
    return region


def _mole_to_mass_fraction(
    co2_mf_prop: np.ndarray,
    gas_mf_prop: np.ndarray,
    water_mf_prop: np.ndarray,
    m_co2: float,
    m_h20: float,
    m_gas: Optional[float],
    m_oil: Optional[float],
) -> np.ndarray:
    """
    Converts from mole fraction to mass fraction

    Args:
      co2_mf_prop (np.ndarray): Property with mole fractions of CO2 in a given phase
      gas_mf_prop (np.ndarray): Property with mole fractions of hydrocarbon gas
                                in a given phase.For more than two components
      h20_mf_prop (np.ndarray): Property with mole fractions of H2O in a given phase
      m_co2 (float): Molar mass of CO2
      m_h20 (float): Molar mass of H2O
      m_gas (float): Molar mass of hydrocarbon gas
      m_oil (float): Molar mass of oil

    Returns:
      np.ndarray

    """

    m_gas = m_gas if m_gas is not None else 0.0
    m_oil = m_oil if m_oil is not None else 0.0
    return (
        co2_mf_prop
        * m_co2
        / (
            co2_mf_prop * m_co2
            + gas_mf_prop * m_gas
            + water_mf_prop * m_h20
            + (1 - co2_mf_prop - gas_mf_prop - water_mf_prop) * m_oil
        )
    )


def _set_calc_type_from_input_string(calc_type_input: str) -> CalculationType:
    """
    Creates a CalculationType object from an input string

    Args:
      calc_type_input (str): Input string with calculation type to perform

    Returns:
      CalculationType

    """
    calc_type_input = calc_type_input.upper()
    CalculationType.check_for_key(calc_type_input)
    return CalculationType[calc_type_input]


def _pflotran_co2mass(
    source_data,
    scenario: Scenario,
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
    gas_molar_mass: Optional[float] = None,
    oil_molar_mass: Optional[float] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Calculates CO2 mass based on the existing properties in PFlotran

    Args:
      source_data (SourceData): Data with the information of the necessary properties
                                for the calculation of CO2 mass
      scenario (Scenario): Which scenario co2 mass is computed for
      co2_molar_mass (float): CO2 molar mass - Default is 44 g/mol
      water_molar_mass (float): Water molar mass - Default is 18 g/mol
      gas_molar_mass (float): Gas molar mass - Default is 0 g/mol,
                              input required if more than 2 components
      oil_molar_mass (float): Oil molar mass - Default is 0 g/mol
                              input required if more than 3 components

    Returns:
      Dict

    """
    dates = source_data.DATES
    dwat = source_data.DWAT
    dgas = source_data.DGAS
    doil = source_data.DOIL
    amfg = source_data.AMFG
    ymfg = source_data.YMFG
    xmfg = source_data.XMFG
    amfw = source_data.AMFW
    ymfw = source_data.YMFW
    xmfw = source_data.XMFW
    amfs = source_data.AMFS
    ymfs = source_data.YMFS
    xmfs = source_data.XMFS
    sgas = source_data.SGAS
    swat = source_data.SWAT
    if swat is None and scenario != Scenario.DEPLETED_OIL_GAS_FIELD:
        swat = {key: 1 - sgas[key] for key in sgas}
    sgstrand = source_data.SGSTRAND
    eff_vols = source_data.PORV
    mole_fraction_dic = {
        "Aqueous": {
            "CO2": amfg if scenario == Scenario.AQUIFER else amfs,
            "Water": (
                amfw
                if amfw is not None
                else (
                    {key: 1 - amfg[key] for key in amfg}
                    if scenario == Scenario.AQUIFER
                    else None
                )
            ),
            "Gas": (
                {key: np.zeros_like(value) for key, value in amfg.items()}
                if scenario == Scenario.AQUIFER
                else amfg
            ),
        },
        "Gas": {
            "CO2": ymfg if scenario == Scenario.AQUIFER else ymfs,
            "Water": (
                ymfw
                if ymfw is not None
                else (
                    {key: 1 - ymfg[key] for key in ymfg}
                    if scenario == Scenario.AQUIFER
                    else None
                )
            ),
            "Gas": (
                {key: np.zeros_like(value) for key, value in ymfg.items()}
                if scenario == Scenario.AQUIFER
                else ymfg
            ),
        },
        "Oil": {
            "CO2": (
                xmfs
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
                else {key: np.zeros_like(value) for key, value in ymfg.items()}
            ),
            "Water": (
                xmfw
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
                else {key: np.zeros_like(value) for key, value in ymfg.items()}
            ),
            "Gas": (
                xmfg
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
                else {key: np.zeros_like(value) for key, value in ymfg.items()}
            ),
        },
    }
    co2_mass = {}
    for date in dates:
        co2_mass[date] = [
            eff_vols[date]
            * swat[date]
            * dwat[date]
            * _mole_to_mass_fraction(
                mole_fraction_dic["Aqueous"]["CO2"][date],
                mole_fraction_dic["Aqueous"]["Gas"][date],
                mole_fraction_dic["Aqueous"]["Water"][date],
                co2_molar_mass,
                water_molar_mass,
                gas_molar_mass,
                oil_molar_mass,
            ),
            eff_vols[date]
            * sgas[date]
            * dgas[date]
            * _mole_to_mass_fraction(
                mole_fraction_dic["Gas"]["CO2"][date],
                mole_fraction_dic["Gas"]["Gas"][date],
                mole_fraction_dic["Gas"]["Water"][date],
                co2_molar_mass,
                water_molar_mass,
                gas_molar_mass,
                oil_molar_mass,
            ),
        ]
        if scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
            co2_mass[date].extend(
                [
                    eff_vols[date]
                    * (1 - sgas[date] - swat[date])
                    * doil[date]
                    * _mole_to_mass_fraction(
                        mole_fraction_dic["Oil"]["CO2"][date],
                        mole_fraction_dic["Oil"]["Gas"][date],
                        mole_fraction_dic["Oil"]["Water"][date],
                        co2_molar_mass,
                        water_molar_mass,
                        gas_molar_mass,
                        oil_molar_mass,
                    ),
                ]
            )
        else:
            co2_mass[date].extend([np.zeros_like(co2_mass[date][0])])

        if sgstrand:
            co2_mass[date].extend(
                [
                    eff_vols[date]
                    * sgstrand[date]
                    * dgas[date]
                    * _mole_to_mass_fraction(
                        mole_fraction_dic["Gas"]["CO2"][date],
                        mole_fraction_dic["Gas"]["Gas"][date],
                        mole_fraction_dic["Gas"]["Water"][date],
                        co2_molar_mass,
                        water_molar_mass,
                        gas_molar_mass,
                        oil_molar_mass,
                    ),
                    eff_vols[date]
                    * (sgas[date] - sgstrand[date])
                    * dgas[date]
                    * _mole_to_mass_fraction(
                        mole_fraction_dic["Gas"]["CO2"][date],
                        mole_fraction_dic["Gas"]["Gas"][date],
                        mole_fraction_dic["Gas"]["Water"][date],
                        co2_molar_mass,
                        water_molar_mass,
                        gas_molar_mass,
                        oil_molar_mass,
                    ),
                ]
            )
    return co2_mass


def _eclipse_co2mass(
    source_data,
    scenario: Scenario,
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
) -> Dict[str, List[np.ndarray]]:
    """
    Calculates CO2 mass based on the existing properties in Eclipse

    Args:
      source_data (SourceData): Data with the information of the necessary properties
                                for the calculation of CO2 mass
      scenario (Scenario): Which scenario co2 mass is computed for
      co2_molar_mass (float): CO2 molar mass - Default is 44 g/mol

    Returns:
      Dict

    """
    dates = source_data.DATES
    bgas = source_data.BGAS
    bwat = source_data.BWAT
    xmf2 = source_data.XMF2
    ymf2 = source_data.YMF2
    sgas = source_data.SGAS
    swat = source_data.SWAT
    sgtrh = source_data.SGTRH
    eff_vols = source_data.RPORV
    conv_fact = co2_molar_mass
    co2_mass = {}
    for date in dates:
        co2_mass[date] = [
            (
                conv_fact * bwat[date] * xmf2[date] * swat[date] * eff_vols[date]
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
                else conv_fact
                * bwat[date]
                * xmf2[date]
                * (1 - sgas[date])
                * eff_vols[date]
            ),
            conv_fact * bgas[date] * ymf2[date] * sgas[date] * eff_vols[date],
        ]
        co2_mass[date].extend([np.zeros_like(co2_mass[date][0])])

        if sgtrh:
            co2_mass[date].extend(
                [
                    conv_fact * bgas[date] * ymf2[date] * sgtrh[date] * eff_vols[date],
                    conv_fact
                    * bgas[date]
                    * ymf2[date]
                    * (sgas[date] - sgtrh[date])
                    * eff_vols[date],
                ]
            )
    return co2_mass


def _pflotran_co2_molar_volume(
    source_data,
    scenario: Scenario,
    water_density: np.ndarray,
    gas_density=np.ndarray,
    oil_density=Optional[np.ndarray],
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
    gas_molar_mass: Optional[float] = None,
    oil_molar_mass: Optional[float] = None,
) -> Dict:
    """
    Calculates CO2 molar volume (mol/m3) based on the existing properties in PFlotran

    Args:
      source_data (SourceData): Data with the information of the necessary properties
                                for the calculation of CO2 molar volume
      scenario (Scenario): Scenario under which CO2 is calculated
      water_density (float): Water density - Default is 1000 kg/m3
      co2_molar_mass (float): CO2 molar mass - Default is 44 g/mol
      water_molar_mass (float): Water molar mass - Default is 18 g/mol

    Returns:
      Dict

    """
    dates = source_data.DATES
    dgas = source_data.DGAS
    dwat = source_data.DWAT
    doil = source_data.DOIL
    ymfg = source_data.YMFG
    amfg = source_data.AMFG
    xmfg = source_data.XMFG
    amfw = source_data.AMFW
    ymfw = source_data.YMFW
    xmfw = source_data.XMFW
    amfs = source_data.AMFS
    ymfs = source_data.YMFS
    xmfs = source_data.XMFS

    gas_molar_mass = gas_molar_mass if gas_molar_mass is not None else 0.0
    oil_molar_mass = oil_molar_mass if oil_molar_mass is not None else 0.0

    mole_fraction_dic = {
        "Aqueous": {
            "CO2": amfg if scenario == Scenario.AQUIFER else amfs,
            "Water": (
                amfw
                if amfw is not None
                else (
                    {key: 1 - amfg[key] for key in amfg}
                    if scenario == Scenario.AQUIFER
                    else None
                )
            ),
            "Gas": (
                {key: np.zeros_like(value) for key, value in amfg.items()}
                if scenario == Scenario.AQUIFER
                else amfg
            ),
        },
        "Gas": {
            "CO2": ymfg if scenario == Scenario.AQUIFER else ymfs,
            "Water": (
                ymfw
                if ymfw is not None
                else (
                    {key: 1 - ymfg[key] for key in ymfg}
                    if scenario == Scenario.AQUIFER
                    else None
                )
            ),
            "Gas": (
                {key: np.zeros_like(value) for key, value in ymfg.items()}
                if scenario == Scenario.AQUIFER
                else ymfg
            ),
        },
        "Oil": {
            "CO2": (
                xmfs
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
                else {key: np.zeros_like(value) for key, value in ymfg.items()}
            ),
            "Water": (
                xmfw
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
                else {key: np.zeros_like(value) for key, value in ymfg.items()}
            ),
            "Gas": (
                xmfg
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
                else {key: np.zeros_like(value) for key, value in ymfg.items()}
            ),
        },
    }

    co2_molar_vol = {}
    for date in dates:
        co2_molar_vol[date] = [
            [
                (
                    (1 / mole_fraction_dic["Aqueous"]["CO2"][date][x])
                    * (
                        -water_molar_mass
                        * (mole_fraction_dic["Aqueous"]["Water"][date][x])
                        / (1000 * water_density[x])
                        + (
                            co2_molar_mass
                            * mole_fraction_dic["Aqueous"]["CO2"][date][x]
                            + water_molar_mass
                            * (mole_fraction_dic["Aqueous"]["Water"][date][x])
                        )
                        / (1000 * dwat[date][x])
                    )
                    if not mole_fraction_dic["Aqueous"]["CO2"][date][x] == 0
                    else 0
                )
                for x in range(len(mole_fraction_dic["Aqueous"]["CO2"][date]))
            ],
            [
                (
                    (1 / mole_fraction_dic["Gas"]["CO2"][date][x])
                    * (
                        -water_molar_mass
                        * mole_fraction_dic["Gas"]["Water"][date][x]
                        / (1000 * water_density[x])
                        - gas_molar_mass
                        * mole_fraction_dic["Gas"]["Gas"][date][x]
                        / (1000 * gas_density[x])
                        - oil_molar_mass
                        * (
                            1
                            - mole_fraction_dic["Gas"]["CO2"][date][x]
                            - mole_fraction_dic["Gas"]["Water"][date][x]
                            - mole_fraction_dic["Gas"]["Gas"][date][x]
                        )
                        / (1000 * oil_density[x])
                        + (
                            co2_molar_mass * mole_fraction_dic["Gas"]["CO2"][date][x]
                            + water_molar_mass
                            * mole_fraction_dic["Gas"]["Water"][date][x]
                            + gas_molar_mass * mole_fraction_dic["Gas"]["Gas"][date][x]
                            + oil_molar_mass
                            * (
                                1
                                - mole_fraction_dic["Gas"]["CO2"][date][x]
                                - mole_fraction_dic["Gas"]["Water"][date][x]
                                - mole_fraction_dic["Gas"]["Gas"][date][x]
                            )
                        )
                        / (1000 * dgas[date][x])
                    )
                    if not mole_fraction_dic["Gas"]["CO2"][date][x] == 0
                    else 0
                )
                for x in range(len(mole_fraction_dic["Gas"]["CO2"][date]))
            ],
        ]
        if scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
            co2_molar_vol[date].extend(
                [
                    [
                        (
                            (1 / mole_fraction_dic["Oil"]["CO2"][date][x])
                            * (
                                -water_molar_mass
                                * mole_fraction_dic["Oil"]["Water"][date][x]
                                / (1000 * water_density[x])
                                - gas_molar_mass
                                * mole_fraction_dic["Oil"]["Gas"][date][x]
                                / (1000 * gas_density[x])
                                - oil_molar_mass
                                * (
                                    1
                                    - mole_fraction_dic["Oil"]["CO2"][date][x]
                                    - mole_fraction_dic["Oil"]["Water"][date][x]
                                    - mole_fraction_dic["Oil"]["Gas"][date][x]
                                )
                                / (1000 * oil_density[x])
                                + (
                                    co2_molar_mass
                                    * mole_fraction_dic["Oil"]["CO2"][date][x]
                                    + water_molar_mass
                                    * mole_fraction_dic["Oil"]["Water"][date][x]
                                    + gas_molar_mass
                                    * mole_fraction_dic["Oil"]["Gas"][date][x]
                                    + oil_molar_mass
                                    * (
                                        1
                                        - mole_fraction_dic["Oil"]["CO2"][date][x]
                                        - mole_fraction_dic["Oil"]["Water"][date][x]
                                        - mole_fraction_dic["Oil"]["Gas"][date][x]
                                    )
                                )
                                / (1000 * doil[date][x])
                            )
                            if not mole_fraction_dic["Oil"]["CO2"][date][x] == 0
                            else 0
                        )
                        for x in range(len(mole_fraction_dic["Oil"]["CO2"][date]))
                    ]
                ],
            )
        else:
            co2_molar_vol[date].extend([list(np.zeros_like(co2_molar_vol[date][0]))])
        co2_molar_vol[date][0] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(
                co2_molar_vol[date][0], mole_fraction_dic["Aqueous"]["CO2"][date]
            )
        ]
        co2_molar_vol[date][1] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(
                co2_molar_vol[date][1], mole_fraction_dic["Gas"]["CO2"][date]
            )
        ]
        co2_molar_vol[date][2] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(
                co2_molar_vol[date][2], mole_fraction_dic["Oil"]["CO2"][date]
            )
        ]
        if source_data.SGSTRAND is not None:
            co2_molar_vol[date].extend([co2_molar_vol[date][1], co2_molar_vol[date][1]])
    return co2_molar_vol


def _eclipse_co2_molar_volume(
    source_data,
    water_density: np.ndarray,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
) -> Dict:
    """
    Calculates CO2 molar volume (mol/m3) based on the existing properties in Eclipse

    Args:
      source_data (SourceData): Data with the information of the necessary properties
                                for the calculation of CO2 molar volume
      water_density (float): Water density - Default is 1000 kg/m3
      water_molar_mass (float): Water molar mass - Default is 18 g/mol

    Returns:
      Dict

    """
    dates = source_data.DATES
    bgas = source_data.BGAS
    bwat = source_data.BWAT
    xmf2 = source_data.XMF2
    ymf2 = source_data.YMF2
    co2_molar_vol = {}
    for date in dates:
        co2_molar_vol[date] = [
            [
                (
                    (1 / xmf2[date][x])
                    * (
                        -water_molar_mass
                        * (1 - xmf2[date][x])
                        / (1000 * water_density[x])
                        + 1 / (1000 * bwat[date][x])
                    )
                    if not xmf2[date][x] == 0
                    else 0
                )
                for x in range(len(xmf2[date]))
            ],
            [
                (
                    (1 / ymf2[date][x])
                    * (
                        -water_molar_mass
                        * (1 - ymf2[date][x])
                        / (1000 * water_density[x])
                        + 1 / (1000 * bgas[date][x])
                    )
                    if not ymf2[date][x] == 0
                    else 0
                )
                for x in range(len(ymf2[date]))
            ],
        ]
        co2_molar_vol[date].extend([list(np.zeros_like(co2_molar_vol[date][0]))])
        co2_molar_vol[date][0] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(co2_molar_vol[date][0], xmf2[date])
        ]
        co2_molar_vol[date][1] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(co2_molar_vol[date][1], ymf2[date])
        ]
        if source_data.SGTRH is not None:
            co2_molar_vol[date].extend([co2_molar_vol[date][1], co2_molar_vol[date][1]])
    return co2_molar_vol


def _calculate_co2_data_from_source_data(
    source_data,
    calc_type: CalculationType,
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
    residual_trapping: bool = False,
    gas_molar_mass: Optional[float] = None,
    oil_molar_mass: Optional[float] = None,
) -> Co2Data:
    """
    Calculates a given calc_type (mass/cell_volume/actual_volume)
    from properties in source_data.

    Args:
        source_data (SourceData): Data with the information of the necessary properties
                                  for the calculation of calc_type
        calc_type (CalculationType): Which amount is calculated (mass / cell_volume /
                                     actual_volume)
        co2_molar_mass (float): CO2 molar mass - Default is 44 g/mol
        water_molar_mass (float): Water molar mass - Default is 18 g/mol
        gas_molar_mass (float): Hydrocarbon gas molar mass - Default is 0 g/mol,
                                should by provided by user
        oil_molar_mass (float) = Oil molar mass - Default is 0 g/mol, not there yet
        residual_trapping (bool): Indicate if residual trapping should be calculated

    Returns:
      Co2Data
    """
    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=too-many-branches
    # pylint: disable-msg=too-many-statements
    logging.info(f"Start calculating CO2 {calc_type.name.lower()} from source data")
    properties_needed_pflotran = PROPERTIES_NEEDED_PFLOTRAN.copy()
    properties_needed_eclipse = PROPERTIES_NEEDED_ECLIPSE.copy()
    if residual_trapping:
        properties_needed_pflotran.append("SGSTRAND")
        properties_needed_eclipse.append("SGTRH")
    props_check = [
        x.name
        for x in fields(source_data)
        if x.name not in ["x_coord", "y_coord", "DATES", "zone", "VOL"]
    ]
    active_props_idx = np.where(
        [getattr(source_data, x) is not None for x in props_check]
    )[0]
    active_props = [props_check[i] for i in active_props_idx]
    scenario = Scenario.AQUIFER
    porv_prop = None
    if _is_subset(["SGAS"], active_props):
        if _is_subset(["PORV", "RPORV"], active_props):
            porv_prop = "RPORV"
            active_props.remove("PORV")
            active_props.remove("RPORV")
            logging.info("Using attribute RPORV instead of PORV")
        elif _is_subset(["PORV"], active_props):
            active_props.remove("PORV")
            porv_prop = "PORV"
            logging.info("Using attribute PORV")
        elif _is_subset(["RPORV"], active_props):
            active_props.remove("RPORV")
            porv_prop = "RPORV"
            logging.info("Using attribute RPORV")
        else:
            error_text = "No pore volume provided"
            error_text += "\nNeed either PORV or RPORV"
            raise ValueError(error_text)
        if _is_subset(properties_needed_pflotran, active_props):
            source = "PFlotran"
            if _is_subset(["AMFS", "SOIL"], active_props):
                scenario = Scenario.DEPLETED_OIL_GAS_FIELD
            elif _is_subset(["AMFS"], active_props):
                scenario = Scenario.DEPLETED_GAS_FIELD
        elif _is_subset(properties_needed_eclipse, active_props):
            source = "Eclipse"
            if _is_subset(["XMF2", "SOIL"], active_props):
                scenario = Scenario.DEPLETED_OIL_GAS_FIELD
            # NBNB: X/YMF properties ending in 2 are assumed to correspond to CO2
            elif _n_components(active_props) > 3:
                scenario = Scenario.DEPLETED_GAS_FIELD
                active_props = [
                    prop
                    for prop in active_props
                    if not (prop.startswith("XMF") or prop.startswith("YMF"))
                    or prop.endswith("2")
                ]
        elif any(prop in properties_needed_pflotran for prop in active_props):
            missing_props = [
                x for x in properties_needed_pflotran if x not in active_props
            ]
            error_text = "Lacking some required properties to compute CO2 mass/volume."
            error_text += "\nAssumed source: PFlotran"
            error_text += "\nMissing properties: "
            error_text += ", ".join(missing_props)
            raise ValueError(error_text)
        elif any(prop in properties_needed_eclipse for prop in active_props):
            missing_props = [
                x for x in properties_needed_eclipse if x not in active_props
            ]
            error_text = "Lacking some required properties to compute CO2 mass/volume."
            error_text += "\nAssumed source: Eclipse"
            error_text += "\nMissing properties: "
            error_text += ", ".join(missing_props)
            raise ValueError(error_text)
        else:
            error_text = "Lacking all required properties to compute CO2 mass/volume."
            error_text += "\nNeed either:"
            error_text += f"\n  PFlotran: \
                {', '.join(properties_needed_pflotran)}"
            error_text += f"\n  Eclipse : \
                {', '.join(properties_needed_eclipse)}"
            raise ValueError(error_text)
    else:
        error_text = "Lacking required property SGAS to compute CO2 mass/volume."
        raise ValueError(error_text)

    active_props.extend([porv_prop])
    if scenario != Scenario.AQUIFER and gas_molar_mass is None:
        error_text = f"\nScenario: {scenario.name}."
        error_text += (
            "\nTo compute mass or actual volume in this scenario "
            "hydrocarbon gas molar mass must be provided"
        )
        raise ValueError(error_text)
    elif scenario == Scenario.AQUIFER:
        gas_molar_mass = None
        oil_molar_mass = None
    logging.info("Found valid properties")
    logging.info(f"Data source: {source}")
    logging.info(f"Scenario: {scenario.name}")
    logging.info(f"Properties used in the calculations: {', '.join(active_props)}")

    if calc_type in (CalculationType.ACTUAL_VOLUME, CalculationType.MASS):
        if source == "PFlotran":
            co2_mass_cell = _pflotran_co2mass(
                source_data,
                scenario,
                co2_molar_mass,
                water_molar_mass,
                gas_molar_mass,
                oil_molar_mass,
            )
        else:
            co2_mass_cell = _eclipse_co2mass(source_data, scenario, co2_molar_mass)
        co2_mass_output = Co2Data(
            source_data.x_coord,
            source_data.y_coord,
            [
                Co2DataAtTimeStep(
                    key,
                    value[0],
                    value[1],
                    value[2],
                    np.zeros_like(value[0]),
                    (
                        np.zeros_like(value[0])
                        if source_data.SGSTRAND is None and source_data.SGTRH is None
                        else value[3]
                    ),
                    (
                        np.zeros_like(value[0])
                        if source_data.SGSTRAND is None and source_data.SGTRH is None
                        else value[4]
                    ),
                )
                for key, value in co2_mass_cell.items()
            ],
            "kg",
            scenario,
            source_data.zone,
            source_data.region,
        )
        if calc_type != CalculationType.MASS:
            if source == "PFlotran":
                y_prop = (
                    source_data.AMFG
                    if scenario == Scenario.AQUIFER
                    else source_data.AMFS
                )
                y = y_prop[source_data.DATES[0]]
                min_y = np.min(y)
                where_min_amf_co2 = np.where(np.isclose(y, min_y))[0]
                # Where amfg is 0, or the closest approximation available
                dwat = source_data.DWAT[source_data.DATES[0]]
                water_density = np.array(
                    [
                        (
                            x[1]
                            if np.isclose((y[x[0]]), 0)
                            else np.mean(dwat[where_min_amf_co2])
                        )
                        for x in enumerate(dwat)
                    ]
                )
                y = source_data.YMFG[source_data.DATES[0]]
                max_y = np.max(y)
                where_max_ymfg = np.where(np.isclose(y, max_y))[0]
                dgas = source_data.DGAS[source_data.DATES[0]]
                gas_density = np.array(
                    [
                        (
                            x[1]
                            if np.isclose((y[x[0]]), 1)
                            else np.mean(dgas[where_max_ymfg])
                        )
                        for x in enumerate(dgas)
                    ]
                )
                oil_density = np.ones_like(water_density)
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
                    y = source_data.YMFO[source_data.DATES[0]]
                    max_y = np.max(y)
                    where_max_xmfo = np.where(np.isclose(y, max_y))[0]
                    doil = source_data.DOIL[source_data.DATES[0]]
                    oil_density = np.array(
                        [
                            (
                                x[1]
                                if np.isclose((y[x[0]]), 1)
                                else np.mean(doil[where_max_xmfo])
                            )
                            for x in enumerate(doil)
                        ]
                    )
                molar_vols_co2 = _pflotran_co2_molar_volume(
                    source_data,
                    scenario,
                    water_density,
                    gas_density,
                    oil_density,
                    co2_molar_mass,
                    water_molar_mass,
                    gas_molar_mass,
                    oil_molar_mass,
                )
            else:
                y = source_data.XMF2[source_data.DATES[0]]
                min_y = np.min(y)
                where_min_xmf2 = np.where(np.isclose(y, min_y))[0]
                # Where xmf2 is 0, or the closest approximation available
                bwat = source_data.BWAT[source_data.DATES[0]]
                water_density = np.array(
                    [
                        (
                            water_molar_mass * x[1]
                            if np.isclose((y[x[0]]), 0)
                            else water_molar_mass * np.mean(bwat[where_min_xmf2])
                        )
                        for x in enumerate(bwat)
                    ]
                )
                molar_vols_co2 = _eclipse_co2_molar_volume(
                    source_data,
                    water_density,
                    water_molar_mass,
                )
            co2_mass = {
                co2_mass_output.data_list[t].date: (
                    [
                        co2_mass_output.data_list[t].dis_water_phase,
                        co2_mass_output.data_list[t].gas_phase,
                        co2_mass_output.data_list[t].dis_oil_phase,
                    ]
                    if (source_data.SGSTRAND is None and source_data.SGTRH is None)
                    else [
                        co2_mass_output.data_list[t].dis_water_phase,
                        co2_mass_output.data_list[t].gas_phase,
                        co2_mass_output.data_list[t].dis_oil_phase,
                        co2_mass_output.data_list[t].trapped_gas_phase,
                        co2_mass_output.data_list[t].free_gas_phase,
                    ]
                )
                for t in range(0, len(co2_mass_output.data_list))
            }
            vols_co2 = {
                t: [
                    a * b / (co2_molar_mass / 1000)
                    for a, b in zip(molar_vols_co2[t], co2_mass[t])
                ]
                for t in co2_mass
            }
            co2_amount = Co2Data(
                source_data.x_coord,
                source_data.y_coord,
                [
                    Co2DataAtTimeStep(
                        t,
                        np.array(vols_co2[t][0]),
                        np.array(vols_co2[t][1]),
                        np.array(vols_co2[t][2]),
                        np.zeros_like(np.array(vols_co2[t][0])),
                        (
                            np.zeros_like(np.array(vols_co2[t][0]))
                            if source_data.SGSTRAND is None
                            and source_data.SGTRH is None
                            else np.array(vols_co2[t][3])
                        ),
                        (
                            np.zeros_like(np.array(vols_co2[t][0]))
                            if source_data.SGSTRAND is None
                            and source_data.SGTRH is None
                            else np.array(vols_co2[t][4])
                        ),
                    )
                    for t in vols_co2
                ],
                "m3",
                scenario,
                source_data.zone,
                source_data.region,
            )
        else:
            _convert_from_kg_to_tons(co2_mass_output)
            co2_amount = co2_mass_output
    elif calc_type == CalculationType.CELL_VOLUME:
        props_idx = np.where(
            [getattr(source_data, x) is not None for x in props_check]
        )[0]
        props_names = [props_check[i] for i in props_idx]
        plume_props_names = [x for x in props_names if x in ["SGAS", "AMFG", "XMF2"]]
        if scenario != Scenario.AQUIFER:
            plume_props_names[plume_props_names.index("AMFG")] = "AMFS"
        properties = {x: getattr(source_data, x) for x in plume_props_names}
        inactive_gas_cells = {
            x: _identify_gas_less_cells(
                {x: properties[plume_props_names[0]][x]},
                {x: properties[plume_props_names[1]][x]},
            )
            for x in source_data.DATES
        }
        vols_ext = {
            t: np.array([0] * len(source_data.VOL[t])) for t in source_data.DATES
        }
        for date in source_data.DATES:
            vols_ext[date][~inactive_gas_cells[date]] = np.array(source_data.VOL[date])[
                ~inactive_gas_cells[date]
            ]
        co2_amount = Co2Data(
            source_data.x_coord,
            source_data.y_coord,
            [
                Co2DataAtTimeStep(
                    t,
                    np.zeros_like(np.array(vols_ext[t])),
                    np.zeros_like(np.array(vols_ext[t])),
                    np.zeros_like(np.array(vols_ext[t])),
                    np.array(vols_ext[t]),
                    np.zeros_like(np.array(vols_ext[t])),
                    np.zeros_like(np.array(vols_ext[t])),
                )
                for t in vols_ext
            ],
            "m3",
            scenario,
            source_data.zone,
            source_data.region,
        )
    else:
        error_text = "Illegal calculation type: " + calc_type.name
        error_text += "\nValid options:"
        for calculation_type in CalculationType:
            error_text += "\n  * " + calculation_type.name
        error_text += "\nExiting"
        raise ValueError(error_text)

    logging.info(f"Done calculating CO2 {calc_type.name.lower()} from source data\n")
    return co2_amount


def _convert_from_kg_to_tons(co2_mass_output: Co2Data):
    co2_mass_output.units = "tons"
    for values in co2_mass_output.data_list:
        for x in [
            values.dis_water_phase,
            values.gas_phase,
            values.dis_oil_phase,
            values.trapped_gas_phase,
            values.free_gas_phase,
        ]:
            x *= 0.001


def calculate_co2(
    grid_file: str,
    unrst_file: str,
    zone_info: ZoneInfo,
    region_info: RegionInfo,
    residual_trapping: bool = False,
    calc_type_input: str = "mass",
    init_file: Optional[str] = None,
    gas_molar_mass: Optional[float] = None,
    oil_molar_mass: Optional[float] = None,
) -> Co2Data:
    """
    Calculates the desired amount (calc_type_input) of CO2

    Args:
      grid_file (str): Path to EGRID-file
      unrst_file (str): Path to UNRST-file
      calc_type_input (str): Input string with calculation type to perform
      init_file (str): Path to INIT-file
      zone_info (ZoneInfo): Zone information
      region_info (Dict): Region information
      residual_trapping (bool): Calculate residual trapping or not
      gas_molar_mass (float): Hydrocarbon gas molar mass (Applies for cases with more
            than two components)
      oil_molar_mass (float): Oil molar mass (Applies for cases with more than two
            components)

    Returns:
      CO2Data

    """
    timer = Timer()
    properties_to_extract = copy.deepcopy(RELEVANT_PROPERTIES)
    current_source_data = copy.deepcopy(source_data_)
    properties_to_add, properties_to_extract = _detect_eclipse_mole_fraction_props(
        unrst_file, properties_to_extract, current_source_data
    )
    if residual_trapping:
        properties_to_extract.extend(["SGSTRAND", "SGTRH"])
    timer.start("extract_source_data")
    source_data = _extract_source_data(
        grid_file,
        unrst_file,
        properties_to_add,
        properties_to_extract,
        zone_info,
        region_info,
        init_file,
    )
    timer.stop("extract_source_data")
    calc_type = _set_calc_type_from_input_string(calc_type_input)

    timer.start("calculate_co2")
    co2_data = _calculate_co2_data_from_source_data(
        source_data,
        calc_type=calc_type,
        residual_trapping=residual_trapping,
        gas_molar_mass=gas_molar_mass,
        oil_molar_mass=oil_molar_mass,
    )
    timer.stop("calculate_co2")
    return co2_data


if __name__ == "__main__":
    pass
