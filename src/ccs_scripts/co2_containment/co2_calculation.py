# pylint: disable-msg=too-many-lines
"""Methods for CO2 containment calculations"""

import copy
import logging
from dataclasses import dataclass, fields, make_dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xtgeo
from resdata.grid import Grid
from resdata.resfile import ResdataFile

from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import (
    THRESHOLD_DISSOLVED,
    fetch_properties,
    find_active_and_gasless_cells,
    format_error,
    format_warning,
    identify_gas_less_cells,
    is_subset,
    reduce_properties,
    try_prop,
)

DEFAULT_CO2_MOLAR_MASS = 44.0
DEFAULT_WATER_MOLAR_MASS = 18.0
PROPERTIES_NEEDED_PFLOTRAN = ["SGAS", "DGAS", "DWAT"]
PROPERTIES_NEEDED_ECLIPSE = ["SGAS", "BGAS", "BWAT", "XMF2", "YMF2"]

RELEVANT_PROPERTIES = [
    "RPORV",
    "PORV",
    "SGAS",
    "DGAS",
    "BGAS",
    "SWAT",
    "DWAT",
    "BWAT",
    "SOIL",
    "DOIL",
    "BOIL",
    "AMFG",
    "YMFG",
    "XMFG",
    "AMFS",
    "YMFS",
    "XMFS",
    "AMFW",
    "YMFW",
    "XMFW",
    "XMFO",
    "YMFO",
]

source_data_: List[Tuple[str, Any, None]] = [
    ("x_coord", np.ndarray, None),
    ("y_coord", np.ndarray, None),
    ("DATES", List[str], None),
    ("VOL", Optional[Dict[str, np.ndarray]], None),
    ("SOIL", Optional[Dict[str, np.ndarray]], None),
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
    ("BOIL", Optional[Dict[str, np.ndarray]], None),
    ("AMFS", Optional[Dict[str, np.ndarray]], None),
    ("YMFS", Optional[Dict[str, np.ndarray]], None),
    ("XMFS", Optional[Dict[str, np.ndarray]], None),
    ("AMFW", Optional[Dict[str, np.ndarray]], None),
    ("YMFW", Optional[Dict[str, np.ndarray]], None),
    ("XMFW", Optional[Dict[str, np.ndarray]], None),
    ("XMFO", Optional[Dict[str, np.ndarray]], None),
    ("YMFO", Optional[Dict[str, np.ndarray]], None),
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
            raise ValueError(format_error(error_text))


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


def _extract_mnemonic_value(info_data, mnemonic: str) -> Optional[float]:
    """Return value for mnemonic if present and valid, else None."""
    if mnemonic not in info_data["Mnemonic"].values:
        return None
    subset = info_data.loc[info_data["Mnemonic"] == mnemonic, "Value"]
    if subset.empty:
        return None
    val = subset.iloc[0]
    if pd.isna(val) or (isinstance(val, str) and not val.strip()):
        return None
    return float(val)


def _extract_molar_masses(
    scenario: Scenario,
    cirrus_info_file: Optional[str] = None,
):
    """
    Extract gas and oil molar masses from a CSV file.

    Args:
        cirrus_info_file (str): Path to the Cirrus info CSV file.
        scenario (Scenario): Which scenario co2 mass is computed for
    Returns:
        tuple[float | None, float | None]: (gas_molar_mass, oil_molar_mass)
    """
    if scenario == Scenario.AQUIFER:
        return None, None
    info_data = pd.read_csv(cirrus_info_file)
    info_data.columns = info_data.columns.str.strip()
    info_data["Mnemonic"] = info_data["Mnemonic"].str.strip()
    gas_molar_mass = _extract_mnemonic_value(info_data, "MWG")
    oil_molar_mass = (
        _extract_mnemonic_value(info_data, "MWO")
        if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
        else None
    )
    if gas_molar_mass is None:
        error_text = f"\nScenario: {scenario.name}."
        error_text += (
            "\nTo compute mass or actual volume in this scenario "
            "hydrocarbon gas molar mass must be provided"
        )
        raise ValueError(format_error(error_text))
    if scenario == Scenario.DEPLETED_OIL_GAS_FIELD and oil_molar_mass is None:
        error_text = f"\nScenario: {scenario.name}."
        error_text += (
            "\nTo compute mass or actual volume in this scenario "
            "oil molar mass must be provided"
        )
        raise ValueError(format_error(error_text))
    return gas_molar_mass, oil_molar_mass


def _extract_comp_molar_masses(
    cirrus_info_file: str,
):
    info_data = pd.read_csv(cirrus_info_file)
    info_data.columns = info_data.columns.str.strip()
    info_data["Mnemonic"] = info_data["Mnemonic"].str.strip()
    mw_df = (
        info_data.loc[
            info_data["Mnemonic"].str.startswith("MW_", na=False),
            ["Mnemonic", "Value"],
        ]
        .assign(
            Value=lambda df: df["Value"].astype(float),
            Component=lambda df: df["Mnemonic"].str.replace("MW_", "", regex=False),
        )
        .reset_index(drop=True)
    )
    molar_weights = {
        row["Component"]: (i + 1, row["Value"]) for i, row in mw_df.iterrows()
    }
    if "CO2" not in molar_weights:
        raise ValueError("CO2 molar mass not found in cirrus info file")
    return molar_weights


def _detect_eclipse_mole_fraction_props(
    unrst_file: str,
    props_to_extract: List,
    current_source_data: List[Tuple[str, Any, None]],
):
    """
    Detects which and how many components are there in Eclipse data

    Args:
        unrst_file (str): Path to UNSRT file
        props_to_extract (List): List of current properties to extract
        current_source_data (List): List with properties to edit
    """
    unrst = ResdataFile(unrst_file)
    suffix_count = 1
    review_z = True
    while suffix_count < 50:
        tmp_x = try_prop(unrst, "XMF" + str(suffix_count))
        tmp_y = try_prop(unrst, "YMF" + str(suffix_count))
        tmp_z = try_prop(unrst, "ZMF" + str(suffix_count))
        if suffix_count == 1 and tmp_z is None:
            review_z = False
        if tmp_x is None and tmp_y is None:
            break
        if review_z:
            if (tmp_x is None) != (tmp_y is None) or (tmp_z is None) != (tmp_y is None):
                error_text = (
                    "Error: Number of components with XMF property differ from "
                    "the number of components with YMF"
                )
                raise ValueError(format_error(error_text))
            else:
                current_source_data.extend(
                    [
                        (
                            name + str(suffix_count),
                            Optional[Dict[str, np.ndarray]],
                            None,
                        )
                        for name in ["XMF", "YMF", "ZMF"]
                    ]
                )
                props_to_extract.extend(
                    [name + str(suffix_count) for name in ["XMF", "YMF", "ZMF"]]
                )
        else:
            if (tmp_x is None) != (tmp_y is None):
                error_text = (
                    "Error: Number of components with XMF property differ from "
                    "the number of components with YMF"
                )
                raise ValueError(format_error(error_text))
            else:
                current_source_data.extend(
                    [
                        (
                            name + str(suffix_count),
                            Optional[Dict[str, np.ndarray]],
                            None,
                        )
                        for name in ["XMF", "YMF"]
                    ]
                )
                props_to_extract.extend(
                    [name + str(suffix_count) for name in ["XMF", "YMF"]]
                )
        suffix_count += 1
    return current_source_data, props_to_extract


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
        error_text = (
            "Error: Number of components with XMF property differ from "
            "the number of components with YMF"
        )
        raise ValueError(format_error(error_text))
    return max_xmf_suffix


def _compute_phases_avg_mol_weight(
    source_data,
    comp_molar_masses: Optional[Dict[str, Tuple[int, float]]],
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
):
    if comp_molar_masses is None:
        raise ValueError(
            "comp_molar_masses cannot be None when computing phase average molar "
            "mass weight"
        )
    dates = source_data.DATES
    gas_avg_mol_weight = {}
    oil_avg_mol_weight = {}
    water_avg_mol_weight = {}
    for date in dates:
        water_avg_mol_weight_at_date = {}
        gas_avg_mol_weight_at_date = {}
        oil_avg_mol_weight_at_date = {}
        for idx, molar_mass in comp_molar_masses.values():
            ymf_tmp_date = getattr(source_data, f"YMF{idx}")[date]
            xmf_tmp_date = getattr(source_data, f"XMF{idx}")[date]
            gas_avg_mol_weight_at_date[idx] = molar_mass * ymf_tmp_date
            oil_avg_mol_weight_at_date[idx] = (
                molar_mass * xmf_tmp_date if Scenario.DEPLETED_OIL_GAS_FIELD else None
            )
            water_avg_mol_weight_at_date[idx] = (
                molar_mass * xmf_tmp_date
                if not Scenario.DEPLETED_OIL_GAS_FIELD
                else (water_molar_mass / len(comp_molar_masses))
                * np.ones_like(xmf_tmp_date)
            )
        gas_avg_mol_weight[date] = np.sum(
            list(gas_avg_mol_weight_at_date.values()), axis=0
        )
        oil_avg_mol_weight[date] = np.sum(
            list(oil_avg_mol_weight_at_date.values()), axis=0
        )
        water_avg_mol_weight[date] = np.sum(
            list(water_avg_mol_weight_at_date.values()), axis=0
        )
    return water_avg_mol_weight, gas_avg_mol_weight, oil_avg_mol_weight


def _convert_phase_density_from_mass_to_mole(
    source_data,
    comp_molar_masses: Optional[Dict[str, Tuple[int, float]]],
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
):
    water_avg_mol_weight, gas_avg_mol_weight, oil_avg_mol_weight = (
        _compute_phases_avg_mol_weight(source_data, comp_molar_masses, water_molar_mass)
    )
    dates = source_data.DATES
    dwat = source_data.DWAT
    dgas = source_data.DGAS
    doil = source_data.DOIL
    bwat = {}
    bgas = {}
    boil = {}
    for date in dates:
        bwat[date] = dwat[date] / water_avg_mol_weight[date]
        bgas[date] = dgas[date] / gas_avg_mol_weight[date]
        boil[date] = (
            doil[date] / oil_avg_mol_weight[date]
            if Scenario.DEPLETED_OIL_GAS_FIELD
            else np.zeros_like(bgas[date])
        )
    return bwat, bgas, boil


def _find_props_to_extract(unrst_file: str, residual_trapping: bool):
    props_to_extract = copy.deepcopy(RELEVANT_PROPERTIES)
    current_source_data = copy.deepcopy(source_data_)
    source_data_updated, props_to_extract = _detect_eclipse_mole_fraction_props(
        unrst_file, props_to_extract, current_source_data
    )
    if residual_trapping:
        props_to_extract.extend(["SGSTRAND", "SGTRH"])

    return source_data_updated, props_to_extract


# pylint: disable=too-many-arguments
def _extract_source_data(
    grid_file: str,
    unrst_file: str,
    source_data_updated: Iterable[Union[str, Tuple[str, type], Tuple[str, type, Any]]],
    props_to_extract: List[str],
    zone_info: ZoneInfo,
    region_info: RegionInfo,
    init_file: Optional[str] = None,
):
    # pylint: disable=too-many-locals, too-many-statements
    """Extracts the properties in props_to_extract from Grid files

    Args:
      grid_file (str): Path to EGRID-file
      unrst_file (str): Path to UNRST-file
      source_data_updated: Source data with properties to be extracted
      props_to_extract (List): Names of the properties to be extracted
      init_file (str): Path to INIT-file
      zone_info (ZoneInfo): Zone information
      region_info (Dict): Region information

    Returns:
      SourceData

    """
    logging.info("Start extracting source data\n")
    grid = Grid(grid_file)
    unrst = ResdataFile(unrst_file)
    try:
        init = ResdataFile(init_file)
    except Exception:
        init = None
        logging.info(format_warning("No INIT-file loaded"))
    properties, dates = fetch_properties(unrst, props_to_extract)

    active, gasless = find_active_and_gasless_cells(grid, properties, True)
    global_active_idx = active[~gasless]

    props_reduced = reduce_properties(properties, ~gasless)
    # Tuple with (x,y,z) for each cell:
    xyz = [grid.get_xyz(global_index=a) for a in global_active_idx]
    cells_x = np.array([coord[0] for coord in xyz])
    cells_y = np.array([coord[1] for coord in xyz])

    zone = _process_zones(zone_info, grid, grid_file, global_active_idx)
    region = _process_regions(region_info, grid, grid_file, init, active, gasless)
    vol0 = [grid.cell_volume(global_index=x) for x in global_active_idx]
    props_reduced["VOL"] = {d: vol0 for d in dates}
    if init is not None:
        try:
            porv = init["PORV"]
            props_reduced["PORV"] = {
                d: porv[0].numpy_copy()[global_active_idx] for d in dates
            }
        except KeyError:
            pass
    SourceData = make_dataclass("SourceData", source_data_updated)
    source_data = SourceData(
        cells_x,
        cells_y,
        dates,
        **dict(props_reduced.items()),
        zone=zone,
        region=region,
    )
    logging.info("\nDone extracting source data\n")
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
        raise ValueError(format_error(err))


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
            zonevals = list(np.unique(zone))
            intvals = np.array(zonevals, dtype=int)
            if np.sum(intvals == zonevals) != len(zonevals):
                warning_text = (
                    "Warning: Grid provided in zone file contains non-integer values. "
                    "This might cause problems with the calculations for "
                    "containment in different zones."
                )
                logging.info(format_warning(warning_text))
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
            warning_text = (
                "Warning: Grid provided in region file contains non-integer values. "
                "This might cause problems with the calculations for "
                "containment in different regions."
            )
            logging.info(warning_text)
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
                logging.info(
                    format_warning("Region information not found in INIT-file.")
                )
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
    pore_volume_prop: str,
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
      pore_volume_prop (str): Which pore volume property to use (RPORV vs PORV)
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
    xmfo = source_data.XMFO
    if swat is None and scenario != Scenario.DEPLETED_OIL_GAS_FIELD:
        # Only gas (co2 or hydrocarbon gas) and water => sgas + swat = 1
        swat = {key: 1 - sgas[key] for key in sgas}
    if xmfw is None and scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
        # Assume g = hydrocarbon gas, s = co2, o = oil
        # => The remainder must be the mole fraction for water
        xmfw = {key: 1 - xmfg[key] - xmfs[key] - xmfo[key] for key in xmfg}
    sgstrand = source_data.SGSTRAND
    eff_vols = source_data.RPORV if pore_volume_prop == "RPORV" else source_data.PORV

    mole_fractions = _construct_mole_fractions(
        scenario, amfg, amfs, amfw, ymfg, ymfs, ymfw, xmfs, xmfw, xmfg
    )

    co2_mass = {}
    for date in dates:
        co2_mass[date] = [
            eff_vols[date]
            * swat[date]
            * dwat[date]
            * _mole_to_mass_fraction(
                mole_fractions["Aqueous"]["CO2"][date],
                mole_fractions["Aqueous"]["Gas"][date],
                mole_fractions["Aqueous"]["Water"][date],
                co2_molar_mass,
                water_molar_mass,
                gas_molar_mass,
                oil_molar_mass,
            ),
            eff_vols[date]
            * sgas[date]
            * dgas[date]
            * _mole_to_mass_fraction(
                mole_fractions["Gas"]["CO2"][date],
                mole_fractions["Gas"]["Gas"][date],
                mole_fractions["Gas"]["Water"][date],
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
                        mole_fractions["Oil"]["CO2"][date],
                        mole_fractions["Oil"]["Gas"][date],
                        mole_fractions["Oil"]["Water"][date],
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
                        mole_fractions["Gas"]["CO2"][date],
                        mole_fractions["Gas"]["Gas"][date],
                        mole_fractions["Gas"]["Water"][date],
                        co2_molar_mass,
                        water_molar_mass,
                        gas_molar_mass,
                        oil_molar_mass,
                    ),
                    eff_vols[date]
                    * (sgas[date] - sgstrand[date])
                    * dgas[date]
                    * _mole_to_mass_fraction(
                        mole_fractions["Gas"]["CO2"][date],
                        mole_fractions["Gas"]["Gas"][date],
                        mole_fractions["Gas"]["Water"][date],
                        co2_molar_mass,
                        water_molar_mass,
                        gas_molar_mass,
                        oil_molar_mass,
                    ),
                ]
            )
    return co2_mass


def _compositional_co2mass(
    source_data,
    scenario: Scenario,
    source: str,
    pore_volume_prop: str,
    co2_molar_mass: Optional[float] = None,
    co2_position: Optional[float] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Calculates CO2 mass based on molar weight and mole fraction of the components

    Args:
      source_data (SourceData): Data with the information of the necessary properties
                                for the calculation of CO2 mass
      scenario (Scenario): Which scenario co2 mass is computed for
      pore_volume_prop (str): Which pore volume property to use (RPORV vs PORV)
      co2_molar_mass (float): CO2 molar mass - Default is 44 g/mol

    Returns:
      Dict

    """
    dates = source_data.DATES
    bgas = source_data.BGAS
    bwat = source_data.BWAT
    boil = source_data.BOIL
    sgas = source_data.SGAS
    swat = source_data.SWAT
    sgtrh = source_data.SGTRH
    sgstrand = source_data.SGSTRAND
    soil = source_data.SOIL
    eff_vols = source_data.RPORV if pore_volume_prop == "RPORV" else source_data.PORV
    conv_fact = co2_molar_mass
    if co2_position is not None and source != "PFlotran COMP":
        xmf_co2 = getattr(source_data, f"XMF{co2_position}")
        ymf_co2 = getattr(source_data, f"YMF{co2_position}")
    else:
        xmf_co2 = source_data.XMF2
        ymf_co2 = source_data.YMF2
    phase_moles = {}
    co2_mass = {}
    for date in dates:
        phase_moles[date] = [
            (
                bwat[date] * swat[date] * eff_vols[date]
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
                else bwat[date] * (1 - sgas[date]) * eff_vols[date]
            ),
            bgas[date] * sgas[date] * eff_vols[date],
        ]
        if scenario != Scenario.DEPLETED_OIL_GAS_FIELD:
            phase_moles[date].extend([np.zeros_like(phase_moles[date][0])])
            co2_mass[date] = [
                conv_fact * phase_moles[date][0] * xmf_co2[date],
                conv_fact * phase_moles[date][1] * ymf_co2[date],
                phase_moles[date][2],
            ]
        else:
            zmf_co2 = (
                getattr(source_data, f"ZMF{co2_position}")
                if co2_position is not None and source != "PFlotran COMP"
                else source_data.ZMF2
            )
            phase_moles[date].extend([boil[date] * soil[date] * eff_vols[date]])
            total_moles = (
                phase_moles[date][0] + phase_moles[date][1] + phase_moles[date][2]
            )
            total_co2_mass = total_moles * zmf_co2[date] * conv_fact
            co2_mass[date] = [
                phase_moles[date][1] * ymf_co2[date] * conv_fact,
                phase_moles[date][2] * xmf_co2[date] * conv_fact,
            ]
            co2_mass[date].insert(
                0, total_co2_mass - co2_mass[date][0] - co2_mass[date][1]
            )
        if any(x is not None for x in (sgstrand, sgtrh)):
            co2_mass[date].extend(
                [
                    co2_mass[date][0] * sgtrh[date] / sgas[date],
                    co2_mass[date][0] * (sgas[date] - sgtrh[date]) / sgas[date],
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

    mole_fractions = _construct_mole_fractions(
        scenario, amfg, amfs, amfw, ymfg, ymfs, ymfw, xmfs, xmfw, xmfg
    )

    co2_molar_vol = {}
    for date in dates:
        co2_molar_vol[date] = [
            [
                (
                    (1 / mole_fractions["Aqueous"]["CO2"][date][x])
                    * (
                        -water_molar_mass
                        * (mole_fractions["Aqueous"]["Water"][date][x])
                        / (1000 * water_density[x])
                        + (
                            co2_molar_mass * mole_fractions["Aqueous"]["CO2"][date][x]
                            + water_molar_mass
                            * (mole_fractions["Aqueous"]["Water"][date][x])
                        )
                        / (1000 * dwat[date][x])
                    )
                    if mole_fractions["Aqueous"]["CO2"][date][x] >= THRESHOLD_DISSOLVED
                    else 0
                )
                for x in range(len(mole_fractions["Aqueous"]["CO2"][date]))
            ],
            [
                (
                    (1 / mole_fractions["Gas"]["CO2"][date][x])
                    * (
                        -water_molar_mass
                        * mole_fractions["Gas"]["Water"][date][x]
                        / (1000 * water_density[x])
                        - gas_molar_mass
                        * mole_fractions["Gas"]["Gas"][date][x]
                        / (1000 * gas_density[x])
                        - oil_molar_mass
                        * (
                            1
                            - mole_fractions["Gas"]["CO2"][date][x]
                            - mole_fractions["Gas"]["Water"][date][x]
                            - mole_fractions["Gas"]["Gas"][date][x]
                        )
                        / (1000 * oil_density[x])
                        + (
                            co2_molar_mass * mole_fractions["Gas"]["CO2"][date][x]
                            + water_molar_mass * mole_fractions["Gas"]["Water"][date][x]
                            + gas_molar_mass * mole_fractions["Gas"]["Gas"][date][x]
                            + oil_molar_mass
                            * (
                                1
                                - mole_fractions["Gas"]["CO2"][date][x]
                                - mole_fractions["Gas"]["Water"][date][x]
                                - mole_fractions["Gas"]["Gas"][date][x]
                            )
                        )
                        / (1000 * dgas[date][x])
                    )
                    if not mole_fractions["Gas"]["CO2"][date][x] == 0
                    else 0
                )
                for x in range(len(mole_fractions["Gas"]["CO2"][date]))
            ],
        ]
        if scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
            co2_molar_vol[date].extend(
                [
                    [
                        (
                            (1 / mole_fractions["Oil"]["CO2"][date][x])
                            * (
                                -water_molar_mass
                                * mole_fractions["Oil"]["Water"][date][x]
                                / (1000 * water_density[x])
                                - gas_molar_mass
                                * mole_fractions["Oil"]["Gas"][date][x]
                                / (1000 * gas_density[x])
                                - oil_molar_mass
                                * (
                                    1
                                    - mole_fractions["Oil"]["CO2"][date][x]
                                    - mole_fractions["Oil"]["Water"][date][x]
                                    - mole_fractions["Oil"]["Gas"][date][x]
                                )
                                / (1000 * oil_density[x])
                                + (
                                    co2_molar_mass
                                    * mole_fractions["Oil"]["CO2"][date][x]
                                    + water_molar_mass
                                    * mole_fractions["Oil"]["Water"][date][x]
                                    + gas_molar_mass
                                    * mole_fractions["Oil"]["Gas"][date][x]
                                    + oil_molar_mass
                                    * (
                                        1
                                        - mole_fractions["Oil"]["CO2"][date][x]
                                        - mole_fractions["Oil"]["Water"][date][x]
                                        - mole_fractions["Oil"]["Gas"][date][x]
                                    )
                                )
                                / (1000 * doil[date][x])
                            )
                            if not mole_fractions["Oil"]["CO2"][date][x] == 0
                            else 0
                        )
                        for x in range(len(mole_fractions["Oil"]["CO2"][date]))
                    ]
                ],
            )
        else:
            co2_molar_vol[date].extend([list(np.zeros_like(co2_molar_vol[date][0]))])
        co2_molar_vol[date][0] = [
            0 if x < 0 or y < THRESHOLD_DISSOLVED else x
            for x, y in zip(
                co2_molar_vol[date][0], mole_fractions["Aqueous"]["CO2"][date]
            )
        ]
        co2_molar_vol[date][1] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(co2_molar_vol[date][1], mole_fractions["Gas"]["CO2"][date])
        ]
        co2_molar_vol[date][2] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(co2_molar_vol[date][2], mole_fractions["Oil"]["CO2"][date])
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
                    if xmf2[date][x] >= THRESHOLD_DISSOLVED
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
            0 if x < 0 or y < THRESHOLD_DISSOLVED else x
            for x, y in zip(co2_molar_vol[date][0], xmf2[date])
        ]
        co2_molar_vol[date][1] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(co2_molar_vol[date][1], ymf2[date])
        ]
        if source_data.SGTRH is not None:
            co2_molar_vol[date].extend([co2_molar_vol[date][1], co2_molar_vol[date][1]])
    return co2_molar_vol


def _construct_mole_fractions(
    scenario: Scenario,
    amfg,
    amfs,
    amfw,
    ymfg,
    ymfs,
    ymfw,
    xmfs,
    xmfw,
    xmfg,
):
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
    return mole_fraction_dic


def _calculate_co2_data_from_source_data(
    source_data,
    calc_type: CalculationType,
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
    residual_trapping: bool = False,
    cirrus_info_file: Optional[str] = None,
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
        residual_trapping (bool): Indicate if residual trapping should be calculated
        cirrus_info_file (Optional[str]): Path to cirrus info file

    Returns:
      Co2Data
    """
    logging.info(f"Start calculating CO2 {calc_type.name.lower()} from source data")
    props_check = [
        x.name
        for x in fields(source_data)
        if x.name not in ["x_coord", "y_coord", "DATES", "zone", "region", "VOL"]
    ]

    active_props = [p for p in props_check if getattr(source_data, p) is not None]
    if not is_subset(["SGAS"], active_props):
        error_text = "Lacking required property SGAS to compute CO2 mass/volume."
        raise ValueError(format_error(error_text))

    pore_volume_prop = _find_pore_volume_prop(active_props)
    source, scenario = _find_source_and_scenario(residual_trapping, active_props)
    gas_molar_mass = None
    oil_molar_mass = None
    comp_molar_masses = None
    if source == "PFlotran COMP":
        if cirrus_info_file is None:
            error_text = "Source: PFlotran COMP"
            error_text += f"\nScenario: {scenario.name}."
            error_text += (
                "\nTo compute mass or actual volume in this scenario "
                "path to cirrus INFO file must be provided."
            )
            raise ValueError(format_error(error_text))
        comp_molar_masses = _extract_comp_molar_masses(cirrus_info_file)
    elif source == "PFlotran":
        gas_molar_mass, oil_molar_mass = _extract_molar_masses(
            scenario, cirrus_info_file
        )
    logging.info("Found valid properties")
    logging.info(f"Data source : {source}")
    logging.info(f"Scenario    : {scenario.name}")
    logging.info("Properties used in the calculations:")
    logging.info(f"    {', '.join(active_props)}")

    if calc_type in (CalculationType.ACTUAL_VOLUME, CalculationType.MASS):
        co2_amount = _calc_co2_amount(
            source,
            scenario,
            calc_type,
            residual_trapping,
            source_data,
            pore_volume_prop,
            co2_molar_mass,
            water_molar_mass,
            gas_molar_mass,
            oil_molar_mass,
            comp_molar_masses,
        )
    elif calc_type == CalculationType.CELL_VOLUME:
        co2_amount = _calc_co2_amount_cell_volume(scenario, source_data, props_check)
    else:
        error_text = "Illegal calculation type: " + calc_type.name
        error_text += "\nValid options:"
        for calculation_type in CalculationType:
            error_text += "\n  * " + calculation_type.name
        error_text += "\nExiting"
        raise ValueError(format_error(error_text))

    logging.info(f"Done calculating CO2 {calc_type.name.lower()} from source data\n")
    return co2_amount


def _find_pore_volume_prop(active_props: List[str]) -> str:
    pore_volume_prop = None
    if is_subset(["PORV", "RPORV"], active_props):
        pore_volume_prop = "RPORV"
        active_props.remove("PORV")
        logging.info("Using attribute RPORV instead of PORV")
    elif is_subset(["PORV"], active_props):
        pore_volume_prop = "PORV"
        logging.info("Using attribute PORV")
    elif is_subset(["RPORV"], active_props):
        pore_volume_prop = "RPORV"
        logging.info("Using attribute RPORV")
    else:
        error_text = "No pore volume provided"
        error_text += "\nNeed either PORV or RPORV"
        raise ValueError(format_error(error_text))

    return pore_volume_prop


def _find_source_and_scenario(
    residual_trapping: bool, active_props: List[str]
) -> Tuple[str, Scenario]:
    props_needed_pflotran = PROPERTIES_NEEDED_PFLOTRAN.copy()
    props_needed_eclipse = PROPERTIES_NEEDED_ECLIPSE.copy()
    if residual_trapping:
        props_needed_pflotran.append("SGSTRAND")
        props_needed_eclipse.append("SGTRH")
    if is_subset(props_needed_pflotran, active_props):
        source = "PFlotran"
        if is_subset(["AMFS", "YMFO"], active_props):
            scenario = Scenario.DEPLETED_OIL_GAS_FIELD
        elif is_subset(["AMFS"], active_props):
            scenario = Scenario.DEPLETED_GAS_FIELD
        elif is_subset(["AMFG", "YMFG"], active_props):
            scenario = Scenario.AQUIFER
        elif is_subset(["XMF2"], active_props):
            source = "PFlotran COMP"
            if _n_components(active_props) <= 3:
                scenario = Scenario.AQUIFER
            elif is_subset(["SOIL"], active_props):
                scenario = Scenario.DEPLETED_OIL_GAS_FIELD
            else:
                scenario = Scenario.DEPLETED_GAS_FIELD
        else:
            error_text = (
                "Need to provide either AMFS, AMFG or XMF2 to perform the calculations"
            )
            raise ValueError(format_error(error_text))
    elif is_subset(props_needed_eclipse, active_props):
        source = "Eclipse"
        if _n_components(active_props) <= 3:
            scenario = Scenario.AQUIFER
        elif is_subset(["SOIL"], active_props):
            scenario = Scenario.DEPLETED_OIL_GAS_FIELD
        else:
            scenario = Scenario.DEPLETED_GAS_FIELD
    else:
        _raise_missing_props_error(
            active_props, props_needed_pflotran, props_needed_eclipse
        )
    return source, scenario


def _calc_co2_amount(
    source: str,
    scenario: Scenario,
    calc_type: CalculationType,
    residual_trapping: bool,
    source_data,
    pore_volume_prop: str,
    co2_molar_mass: float,
    water_molar_mass: float,
    gas_molar_mass: Optional[float],
    oil_molar_mass: Optional[float],
    comp_molar_masses: Optional[Dict[str, Tuple[int, float]]],
) -> Co2Data:
    if source == "PFlotran":
        co2_mass_cell = _pflotran_co2mass(
            source_data,
            scenario,
            pore_volume_prop,
            co2_molar_mass,
            water_molar_mass,
            gas_molar_mass,
            oil_molar_mass,
        )
    else:
        co2_position = None
        if source == "PFlotran COMP" and comp_molar_masses is not None:
            bwat, bgas, boil = _convert_phase_density_from_mass_to_mole(
                source_data,
                comp_molar_masses,
                water_molar_mass,
            )
            source_data.BWAT = bwat
            source_data.BGAS = bgas
            source_data.BOIL = boil
            co2_position = comp_molar_masses["CO2"][0]

        co2_mass_cell = _compositional_co2mass(
            source_data,
            scenario,
            source,
            pore_volume_prop,
            co2_molar_mass,
            co2_position,
        )
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
                (value[3] if residual_trapping else np.zeros_like(value[0])),
                (value[4] if residual_trapping else np.zeros_like(value[0])),
            )
            for key, value in co2_mass_cell.items()
        ],
        "kg",
        scenario,
        source_data.zone,
        source_data.region,
    )
    if calc_type == CalculationType.MASS:
        _convert_from_kg_to_tons(co2_mass_output)
        co2_amount = co2_mass_output
    else:
        molar_vols_co2 = _calculate_molar_vols_co2(
            source,
            scenario,
            source_data,
            co2_molar_mass,
            water_molar_mass,
            gas_molar_mass,
            oil_molar_mass,
        )
        co2_mass = {
            co2_mass_output.data_list[t].date: (
                [
                    co2_mass_output.data_list[t].dis_water_phase,
                    co2_mass_output.data_list[t].gas_phase,
                    co2_mass_output.data_list[t].dis_oil_phase,
                ]
                if not residual_trapping
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
                        np.array(vols_co2[t][3])
                        if residual_trapping
                        else np.zeros_like(np.array(vols_co2[t][0]))
                    ),
                    (
                        np.array(vols_co2[t][4])
                        if residual_trapping
                        else np.zeros_like(np.array(vols_co2[t][0]))
                    ),
                )
                for t in vols_co2
            ],
            "m3",
            scenario,
            source_data.zone,
            source_data.region,
        )
    return co2_amount


def _calculate_molar_vols_co2(
    source: str,
    scenario: Scenario,
    source_data,
    co2_molar_mass: float,
    water_molar_mass: float,
    gas_molar_mass: Optional[float],
    oil_molar_mass: Optional[float],
):
    if source == "PFlotran":
        y_prop = source_data.AMFG if scenario == Scenario.AQUIFER else source_data.AMFS
        y = y_prop[source_data.DATES[0]]
        where_min_amf_co2 = np.where(y < THRESHOLD_DISSOLVED)[0]
        if len(where_min_amf_co2) == 0:
            prop_name = "AMFG" if scenario == Scenario.AQUIFER else "AMFS"
            min_y = np.min(y)
            where_min_amf_co2 = np.where(y < min_y + THRESHOLD_DISSOLVED)[0]
            msg = (
                f"WARNING: Lack of cells with low (<{THRESHOLD_DISSOLVED}) "
                f"{prop_name}, needed for estimation of water density."
                f"\n         Using cells with {prop_name} < "
                f"{min_y + THRESHOLD_DISSOLVED} for estimation."
            )
            logging.warning(format_warning(msg))
        # Where amfg is 0, or the closest approximation available
        dwat = source_data.DWAT[source_data.DATES[0]]
        water_density = np.array(
            [
                (
                    x[1]
                    if y[x[0]] < THRESHOLD_DISSOLVED
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
                (x[1] if np.isclose((y[x[0]]), 1) else np.mean(dgas[where_max_ymfg]))
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
        where_min_xmf2 = np.where(y < THRESHOLD_DISSOLVED)[0]
        if len(where_min_xmf2) == 0:
            min_y = np.min(y)
            where_min_xmf2 = np.where(y < min_y + THRESHOLD_DISSOLVED)[0]
            msg = (
                f"WARNING: Lack of cells with low (<{THRESHOLD_DISSOLVED}) XMF2, "
                f"needed for estimation of water density."
                f"\n         Using cells with XMF2 < "
                f"{min_y + THRESHOLD_DISSOLVED} for estimation."
            )
            logging.warning(format_warning(msg))
        # Where xmf2 is 0, or the closest approximation available
        bwat = source_data.BWAT[source_data.DATES[0]]
        water_density = np.array(
            [
                (
                    water_molar_mass * x[1]
                    if y[x[0]] < THRESHOLD_DISSOLVED
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
    return molar_vols_co2


def _calc_co2_amount_cell_volume(
    scenario: Scenario,
    source_data,
    props_check: List[str],
) -> Co2Data:
    props_idx = np.where([getattr(source_data, x) is not None for x in props_check])[0]
    props_names = [props_check[i] for i in props_idx]
    plume_props_names = [x for x in props_names if x in ["SGAS", "AMFG", "XMF2"]]
    if scenario != Scenario.AQUIFER:
        plume_props_names[plume_props_names.index("AMFG")] = "AMFS"
    properties = {x: getattr(source_data, x) for x in plume_props_names}
    inactive_gas_cells = {
        x: identify_gas_less_cells(
            {x: properties[plume_props_names[0]][x]},
            {x: properties[plume_props_names[1]][x]},
        )
        for x in source_data.DATES
    }
    vols_ext = {t: np.array([0] * len(source_data.VOL[t])) for t in source_data.DATES}
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
    return co2_amount


def _raise_missing_props_error(
    active_props: List[str],
    props_needed_pflotran: List[str],
    props_needed_eclipse: List[str],
):
    if any(prop in props_needed_pflotran for prop in active_props):
        missing_props = [x for x in props_needed_pflotran if x not in active_props]
        error_text = "Lacking some required properties to compute CO2 mass/volume."
        error_text += "\nAssumed source: PFlotran"
        error_text += "\nMissing properties: "
        error_text += ", ".join(missing_props)
        raise ValueError(format_error(error_text))
    if any(prop in props_needed_eclipse for prop in active_props):
        missing_props = [x for x in props_needed_eclipse if x not in active_props]
        error_text = "Lacking some required properties to compute CO2 mass/volume."
        error_text += "\nAssumed source: Eclipse"
        error_text += "\nMissing properties: "
        error_text += ", ".join(missing_props)
        raise ValueError(format_error(error_text))
    error_text = "Lacking all required properties to compute CO2 mass/volume."
    error_text += "\nNeed either:"
    error_text += f"\n  PFlotran: \
        {', '.join(props_needed_pflotran)}"
    error_text += f"\n  Eclipse : \
        {', '.join(props_needed_eclipse)}"
    raise ValueError(format_error(error_text))


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
    cirrus_info_file: Optional[str] = None,
) -> Co2Data:
    """
    Calculates the desired amount (calc_type_input) of CO2

    Args:
      grid_file (str): Path to EGRID-file
      unrst_file (str): Path to UNRST-file
      calc_type_input (str): Input string with calculation type to perform
      init_file (str): Path to INIT-file
      zone_info (ZoneInfo): Zone information
      region_info (RegionInfo): Region information
      residual_trapping (bool): Calculate residual trapping or not
      cirrus_info_file (str): Path to cirrus info file

    Returns:
      CO2Data

    """
    timer = Timer()
    source_data_updated, props_to_extract = _find_props_to_extract(
        unrst_file, residual_trapping
    )
    timer.start("extract_source_data")
    source_data = _extract_source_data(
        grid_file,
        unrst_file,
        source_data_updated,
        props_to_extract,
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
        cirrus_info_file=cirrus_info_file,
    )
    timer.stop("calculate_co2")
    return co2_data


if __name__ == "__main__":
    pass
