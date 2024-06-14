"""Methods for CO2 containment calculations"""

import logging
from dataclasses import dataclass, fields
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import xtgeo
from resdata.grid import Grid
from resdata.resfile import ResdataFile

DEFAULT_CO2_MOLAR_MASS = 44.0
DEFAULT_WATER_MOLAR_MASS = 18.0
TRESHOLD_SGAS = 1e-16
TRESHOLD_AMFG = 1e-16
PROPERTIES_NEEDED_PFLOTRAN = ["PORV", "DGAS", "DWAT", "AMFG", "YMFG"]
PROPERTIES_NEEDED_ELCIPSE = ["RPORV", "BGAS", "BWAT", "XMF2", "YMF2"]

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
    "XMF2",
    "YMF2",
    "SGSTRAND",
    "SGTRH",
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


@dataclass
class SourceData:
    """
    Dataclass with grid cell (x,y) coordinates, dates
    and properties (if available)

    Args:
      x_coord (np.ndarray): x coordinates for grid cells
      y_coord (np.ndarray): y coordinates for grid cells
      DATES (List): Time steps each property is available for
      VOL (Dict): Grid cells volume (m3) at each date
      SWAT (Dict): Water saturation for each grid cell at each date
      SGAS (Dict): Gas saturation for each grid cell at each date
      RPORV (Dict): Pore volume (VOL x Porosity) for each grid cell at each date
      PORV (Dict): Pore volume (VOL x Porosity) for each grid cell at each date
      AMFG (Dict): Aqueous mole fraction of gas for each grid cell at each date
      YMFG (Dict): Gaseous mole fraction of gas for each grid cell at each date
      XMF2 (Dict): Aqueous mole fraction of gas for each grid cell at each date
      YMF2 (Dict): Gaseous mole fraction of gas for each grid cell at each date
      DWAT (Dict): Water density (kg/m3) for each grid cell at each date
      DGAS (Dict): Gas density (kg/m3) for each grid cell at each date
      BWAT (Dict): Molar water density (kg-mol/m3) for each grid cell at each date
      BGAS (Dict): Molar gas density (kg-mol/m3) for each grid cell at each date
      zone (np.ndarray): Zone information
      region (np.ndarray): Region information

    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=invalid-name
    x_coord: np.ndarray
    y_coord: np.ndarray
    DATES: List[str]
    VOL: Optional[Dict[str, np.ndarray]] = None
    SWAT: Optional[Dict[str, np.ndarray]] = None
    SGAS: Optional[Dict[str, np.ndarray]] = None
    SGSTRAND: Optional[Dict[str, np.ndarray]] = None
    SGTRH: Optional[Dict[str, np.ndarray]] = None
    RPORV: Optional[Dict[str, np.ndarray]] = None
    PORV: Optional[Dict[str, np.ndarray]] = None
    AMFG: Optional[Dict[str, np.ndarray]] = None
    YMFG: Optional[Dict[str, np.ndarray]] = None
    XMF2: Optional[Dict[str, np.ndarray]] = None
    YMF2: Optional[Dict[str, np.ndarray]] = None
    DWAT: Optional[Dict[str, np.ndarray]] = None
    DGAS: Optional[Dict[str, np.ndarray]] = None
    BWAT: Optional[Dict[str, np.ndarray]] = None
    BGAS: Optional[Dict[str, np.ndarray]] = None
    zone: Optional[np.ndarray] = None
    region: Optional[np.ndarray] = None
    # pylint: enable=invalid-name

    def get_vol(self):
        """Get VOL"""
        if self.VOL is not None:
            return self.VOL
        return {}

    def get_swat(self):
        """Get SWAT"""
        if self.SWAT is not None:
            return self.SWAT
        return {}

    def get_sgas(self):
        """Get SGAS"""
        if self.SGAS is not None:
            return self.SGAS
        return {}

    def get_sgstrand(self):
        """Get SGSTRAND"""
        if self.SGSTRAND is not None:
            return self.SGSTRAND
        return {}

    def get_sgtrh(self):
        """Get SGTRH"""
        if self.SGTRH is not None:
            return self.SGTRH
        return {}

    def get_rporv(self):
        """Get RPORV"""
        if self.RPORV is not None:
            return self.RPORV
        return {}

    def get_porv(self):
        """Get PORV"""
        if self.PORV is not None:
            return self.PORV
        return {}

    def get_amfg(self):
        """Get AMFG"""
        if self.AMFG is not None:
            return self.AMFG
        return {}

    def get_ymfg(self):
        """Get YMFG"""
        if self.YMFG is not None:
            return self.YMFG
        return {}

    def get_xmf2(self):
        """Get XMF2"""
        if self.XMF2 is not None:
            return self.XMF2
        return {}

    def get_ymf2(self):
        """Get YMF2"""
        if self.YMF2 is not None:
            return self.YMF2
        return {}

    def get_dwat(self):
        """Get DWAT"""
        if self.DWAT is not None:
            return self.DWAT
        return {}

    def get_dgas(self):
        """Get DGAS"""
        if self.DGAS is not None:
            return self.DGAS
        return {}

    def get_bwat(self):
        """Get BWAT"""
        if self.BWAT is not None:
            return self.BWAT
        return {}

    def get_bgas(self):
        """Get BGAS"""
        if self.BGAS is not None:
            return self.BGAS
        return {}

    def get_zone(self):
        """Get zone"""
        if self.zone is not None:
            return self.zone
        return None

    def get_region(self):
        """Get region"""
        if self.region is not None:
            return self.region
        return None


@dataclass
class Co2DataAtTimeStep:
    """
    Dataclass with amount of co2 for each phase (dissolved/gas/undefined)
    at a given time step.

    Args:
      date (str): The time step
      aqu_phase (np.ndarray): The amount of CO2 in aqueous phase
      gas_phase (np.ndarray): The amount of CO2 in gaseous phase
      volume_coverage (np.ndarray): The volume of a cell (specific of
                                    calc_type_input = volume_extent)
    """

    date: str
    aqu_phase: np.ndarray
    gas_phase: np.ndarray
    volume_coverage: np.ndarray
    trapped_gas_phase: np.ndarray
    free_gas_phase: np.ndarray

    def total_mass(self) -> np.ndarray:
        """
        Computes total mass as the sum of gas in aqueous and gas
        phase.
        """
        return self.aqu_phase + self.gas_phase


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
    units: Literal["kg", "m3"]
    zone: Optional[np.ndarray] = None
    region: Optional[np.ndarray] = None


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


def _identify_gas_less_cells(sgases: dict, amfgs: dict) -> np.ndarray:
    """
    Identifies those cells that do not have gas. This is done based on thresholds for
    SGAS and AMFG.

    Args:
      sgases (dict): The values of SGAS for each grid cell
      amfgs (dict): The values of AMFG for each grid cell

    Returns:
      np.ndarray

    """
    gas_less = np.logical_and.reduce(
        [np.abs(sgases[s]) < TRESHOLD_SGAS for s in sgases]
    )
    gas_less &= np.logical_and.reduce([np.abs(amfgs[a]) < TRESHOLD_AMFG for a in amfgs])
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


# pylint: disable=too-many-arguments
def _extract_source_data(
    grid_file: str,
    unrst_file: str,
    properties_to_extract: List[str],
    zone_info: Dict,
    region_info: Dict,
    init_file: Optional[str] = None,
) -> SourceData:
    # pylint: disable=too-many-locals, too-many-statements
    """
    Extracts the properties in properties_to_extract from Grid files

    Args:
      grid_file (str): Path to EGRID-file
      unrst_file (str): Path to UNRST-file
      properties_to_extract (List): Names of the properties to be extracted
      init_file (str): Path to INIT-file
      zone_info (Dict): Dictionary containing zone information
      region_info (Dict): Dictionary containing region information

    Returns:
      SourceData

    """
    logging.info("Start extracting source data")
    grid = Grid(grid_file)
    unrst = ResdataFile(unrst_file)
    init = ResdataFile(init_file)
    properties, dates = _fetch_properties(unrst, properties_to_extract)
    logging.info("Done fetching properties")

    act_num = grid.export_actnum().numpy_copy()
    active = np.where(act_num > 0)[0]
    logging.info(f"Number of grid cells                    : {len(act_num)}")
    logging.info(f"Number of active grid cells             : {len(active)}")
    if _is_subset(["SGAS", "AMFG"], list(properties.keys())):
        gasless = _identify_gas_less_cells(properties["SGAS"], properties["AMFG"])
    elif _is_subset(["SGAS", "XMF2"], list(properties.keys())):
        gasless = _identify_gas_less_cells(properties["SGAS"], properties["XMF2"])
    else:
        error_text = (
            "CO2 containment calculation failed. Cannot find required properties "
        )
        error_text += "SGAS+AMFG or SGAS+XMF2."
        raise RuntimeError(error_text)
    global_active_idx = active[~gasless]
    logging.info(f"Number of active non-gasless grid cells : {len(global_active_idx)}")

    properties_reduced = _reduce_properties(properties, ~gasless)
    # Tuple with (x,y,z) for each cell:
    xyz = [grid.get_xyz(global_index=a) for a in global_active_idx]
    cells_x = np.array([coord[0] for coord in xyz])
    cells_y = np.array([coord[1] for coord in xyz])

    zone = _process_zones(zone_info, grid, grid_file, global_active_idx)
    region = _process_regions(region_info, grid, grid_file, init, active, gasless)

    vol0 = [grid.cell_volume(global_index=x) for x in global_active_idx]
    properties_reduced["VOL"] = {d: vol0 for d in dates}
    try:
        porv = init["PORV"]
        properties_reduced["PORV"] = {
            d: porv[0].numpy_copy()[global_active_idx] for d in dates
        }
    except KeyError:
        pass
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
    zone_info: Dict,
    grid: Grid,
    grid_file: str,
    global_active_idx: np.ndarray,
) -> Optional[np.ndarray]:
    zone = None
    if zone_info["source"] is None:
        logging.info("No zone info specified")
    if zone_info["source"] is not None:
        logging.info("Using zone info")
        if zone_info["zranges"] is not None:
            zone_array = np.zeros(
                (grid.get_nx(), grid.get_ny(), grid.get_nz()), dtype=int
            )
            zonevals = [int(x) for x in range(len(zone_info["zranges"]))]
            zone_info["int_to_zone"] = [f"Zone_{x}" for x in range(len(zonevals))]
            for zv, zr, zn in zip(
                zonevals,
                list(zone_info["zranges"].values()),
                zone_info["zranges"].keys(),
            ):
                zone_array[:, :, zr[0] - 1 : zr[1]] = zv
                zone_info["int_to_zone"][zv] = zn
            zone = zone_array.flatten(order="F")[global_active_idx]
        else:
            xtg_grid = xtgeo.grid_from_file(grid_file)
            _check_grid_dimensions(
                zone_info["source"],
                grid_file,
                xtg_grid.ncol,
                xtg_grid.nrow,
                xtg_grid.nlay,
            )
            zone = xtgeo.gridproperty_from_file(zone_info["source"], grid=xtg_grid)
            zone = zone.values.data.flatten(order="F")
            zonevals = np.unique(zone)
            intvals = np.array(zonevals, dtype=int)
            if np.sum(intvals == zonevals) != len(zonevals):
                logging.info(
                    "Warning: Grid provided in zone file contains non-integer values. "
                    "This might cause problems with the calculations for "
                    "containment in different zones."
                )
            zone_info["int_to_zone"] = [None] * (np.max(intvals) + 1)
            for zv in intvals:
                if zv >= 0:
                    zone_info["int_to_zone"][zv] = f"Zone_{zv}"
                else:
                    logging.info("Ignoring negative value in grid from zone file.")
            zone = np.array(zone[global_active_idx], dtype=int)
    return zone


def _process_regions(
    region_info: Dict,
    grid: Grid,
    grid_file: str,
    init: ResdataFile,
    active: np.ndarray,
    gasless: np.ndarray,
) -> Optional[np.ndarray]:
    region = None
    if region_info["source"] is not None:
        logging.info("Using regions info")
        xtg_grid = xtgeo.grid_from_file(grid_file)
        _check_grid_dimensions(
            region_info["source"],
            grid_file,
            xtg_grid.ncol,
            xtg_grid.nrow,
            xtg_grid.nlay,
        )
        region = xtgeo.gridproperty_from_file(region_info["source"], grid=xtg_grid)
        region = region.values.data.flatten(order="F")
        regvals = np.unique(region)
        intvals = np.array(regvals, dtype=int)
        if np.sum(intvals == regvals) != len(regvals):
            logging.info(
                "Warning: Grid provided in region file contains non-integer values. "
                "This might cause problems with the calculations for "
                "containment in different regions."
            )
        region_info["int_to_region"] = [None] * (np.max(intvals) + 1)
        for rv in intvals:
            if rv >= 0:
                region_info["int_to_region"][rv] = f"Region_{rv}"
            else:
                logging.info("Ignoring negative value in grid from region file.")
        region = np.array(region[active[~gasless]], dtype=int)
    elif region_info["property_name"] is not None:
        try:
            logging.info(
                f"Try reading region information ({region_info['property_name']}"
                f" property) from INIT-file."
            )
            region = np.array(init[region_info["property_name"]][0], dtype=int)
            if region.shape[0] == grid.get_nx() * grid.get_ny() * grid.get_nz():
                region = region[active]
            regvals = np.unique(region)
            region_info["int_to_region"] = [None] * (np.max(regvals) + 1)
            for rv in regvals:
                if rv >= 0:
                    region_info["int_to_region"][rv] = f"Region_{rv}"
                else:
                    logging.info(
                        f"Ignoring negative value in {region_info['property_name']}."
                    )
            logging.info("Region information successfully read from INIT-file")
            region = region[~gasless]
        except KeyError:
            logging.info("Region information not found in INIT-file.")
            region = None
            region_info["int_to_region"] = None
    return region


def _mole_to_mass_fraction(prop: np.ndarray, m_co2: float, m_h20: float) -> np.ndarray:
    """
    Converts from mole fraction to mass fraction

    Args:
      prop (np.ndarray): Information with mole fractions to be converted
      m_co2 (float): Molar mass of CO2
      m_h20 (float): Molar mass of H2O

    Returns:
      np.ndarray

    """
    return prop * m_co2 / (m_h20 + (m_co2 - m_h20) * prop)


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
    source_data: SourceData,
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
) -> Dict:
    """
    Calculates CO2 mass based on the existing properties in PFlotran

    Args:
      source_data (SourceData): Data with the information of the necessary properties
                                for the calculation of CO2 mass
      co2_molar_mass (float): CO2 molar mass - Default is 44 g/mol
      water_molar_mass (float): Water molar mass - Default is 18 g/mol

    Returns:
      Dict

    """
    dates = source_data.DATES
    dwat = source_data.get_dwat()
    dgas = source_data.get_dgas()
    amfg = source_data.get_amfg()
    ymfg = source_data.get_ymfg()
    sgas = source_data.get_sgas()
    sgstrand = source_data.get_sgstrand()
    eff_vols = source_data.get_porv()
    co2_mass = {}
    for date in dates:
        co2_mass[date] = [
            eff_vols[date]
            * (1 - sgas[date])
            * dwat[date]
            * _mole_to_mass_fraction(amfg[date], co2_molar_mass, water_molar_mass),
            eff_vols[date]
            * sgas[date]
            * dgas[date]
            * _mole_to_mass_fraction(ymfg[date], co2_molar_mass, water_molar_mass),
        ]
        if len(sgstrand) != 0:
            co2_mass[date].extend(
                [
                    eff_vols[date]
                    * sgstrand[date]
                    * dgas[date]
                    * _mole_to_mass_fraction(
                        ymfg[date], co2_molar_mass, water_molar_mass
                    ),
                    eff_vols[date]
                    * (sgas[date] - sgstrand[date])
                    * dgas[date]
                    * _mole_to_mass_fraction(
                        ymfg[date], co2_molar_mass, water_molar_mass
                    ),
                ]
            )
    return co2_mass


def _eclipse_co2mass(
    source_data: SourceData, co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS
) -> Dict:
    """
    Calculates CO2 mass based on the existing properties in Eclipse

    Args:
      source_data (SourceData): Data with the information of the necessary properties
                                for the calculation of CO2 mass
      co2_molar_mass (float): CO2 molar mass - Default is 44 g/mol

    Returns:
      Dict

    """
    dates = source_data.DATES
    bgas = source_data.get_bgas()
    bwat = source_data.get_bwat()
    xmf2 = source_data.get_xmf2()
    ymf2 = source_data.get_ymf2()
    sgas = source_data.get_sgas()
    sgtrh = source_data.get_sgtrh()
    eff_vols = source_data.get_rporv()
    conv_fact = co2_molar_mass
    co2_mass = {}
    for date in dates:
        co2_mass[date] = [
            conv_fact * bwat[date] * xmf2[date] * (1 - sgas[date]) * eff_vols[date],
            conv_fact * bgas[date] * ymf2[date] * sgas[date] * eff_vols[date],
        ]
        if len(sgtrh) != 0:
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
    source_data: SourceData,
    water_density: np.ndarray,
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
) -> Dict:
    """
    Calculates CO2 molar volume (mol/m3) based on the existing properties in PFlotran

    Args:
      source_data (SourceData): Data with the information of the necessary properties
                                for the calculation of CO2 molar volume
      water_density (float): Water density - Default is 1000 kg/m3
      co2_molar_mass (float): CO2 molar mass - Default is 44 g/mol
      water_molar_mass (float): Water molar mass - Default is 18 g/mol

    Returns:
      Dict

    """
    dates = source_data.DATES
    dgas = source_data.get_dgas()
    dwat = source_data.get_dwat()
    ymfg = source_data.get_ymfg()
    amfg = source_data.get_amfg()
    co2_molar_vol = {}
    for date in dates:
        co2_molar_vol[date] = [
            [
                (
                    (1 / amfg[date][x])
                    * (
                        -water_molar_mass
                        * (1 - amfg[date][x])
                        / (1000 * water_density[x])
                        + (
                            co2_molar_mass * amfg[date][x]
                            + water_molar_mass * (1 - amfg[date][x])
                        )
                        / (1000 * dwat[date][x])
                    )
                    if not amfg[date][x] == 0
                    else 0
                )
                for x in range(len(amfg[date]))
            ],
            [
                (
                    (1 / ymfg[date][x])
                    * (
                        -water_molar_mass
                        * (1 - ymfg[date][x])
                        / (1000 * water_density[x])
                        + (
                            co2_molar_mass * ymfg[date][x]
                            + water_molar_mass * (1 - ymfg[date][x])
                        )
                        / (1000 * dgas[date][x])
                    )
                    if not ymfg[date][x] == 0
                    else 0
                )
                for x in range(len(ymfg[date]))
            ],
        ]
        co2_molar_vol[date][0] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(co2_molar_vol[date][0], amfg[date])
        ]
        co2_molar_vol[date][1] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(co2_molar_vol[date][1], ymfg[date])
        ]
    return co2_molar_vol


def _eclipse_co2_molar_volume(
    source_data: SourceData,
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
    bgas = source_data.get_bgas()
    bwat = source_data.get_bwat()
    xmf2 = source_data.get_xmf2()
    ymf2 = source_data.get_ymf2()
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
        co2_molar_vol[date][0] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(co2_molar_vol[date][0], xmf2[date])
        ]
        co2_molar_vol[date][1] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(co2_molar_vol[date][1], ymf2[date])
        ]
    return co2_molar_vol


def _calculate_co2_data_from_source_data(
    source_data: SourceData,
    calc_type: CalculationType,
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
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

    Returns:
      Co2Data
    """
    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=too-many-branches
    # pylint: disable-msg=too-many-statements
    logging.info(f"Start calculating CO2 {calc_type.name.lower()} from source data")
    props_check = [
        x.name
        for x in fields(source_data)
        if x.name not in ["x_coord", "y_coord", "DATES", "zone", "VOL"]
    ]
    active_props_idx = np.where(
        [getattr(source_data, x) is not None for x in props_check]
    )[0]
    active_props = [props_check[i] for i in active_props_idx]

    if _is_subset(["SGAS"], active_props):
        if _is_subset(["PORV", "RPORV"], active_props):
            active_props.remove("PORV")
            logging.info("Using attribute RPORV instead of PORV")
        if _is_subset(PROPERTIES_NEEDED_PFLOTRAN, active_props):
            source = "PFlotran"
        elif _is_subset(PROPERTIES_NEEDED_ELCIPSE, active_props):
            source = "Eclipse"
        elif any(prop in PROPERTIES_NEEDED_PFLOTRAN for prop in active_props):
            missing_props = [
                x for x in PROPERTIES_NEEDED_PFLOTRAN if x not in active_props
            ]
            error_text = "Lacking some required properties to compute CO2 mass/volume."
            error_text += "\nAssumed source: PFlotran"
            error_text += "\nMissing properties: "
            error_text += ", ".join(missing_props)
            raise ValueError(error_text)
        elif any(prop in PROPERTIES_NEEDED_ELCIPSE for prop in active_props):
            missing_props = [
                x for x in PROPERTIES_NEEDED_ELCIPSE if x not in active_props
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
                {', '.join(PROPERTIES_NEEDED_PFLOTRAN)}"
            error_text += f"\n  Eclipse : \
                {', '.join(PROPERTIES_NEEDED_ELCIPSE)}"
            raise ValueError(error_text)
    else:
        error_text = "Lacking required property SGAS to compute CO2 mass/volume."
        raise ValueError(error_text)

    logging.info("Found valid properties")
    logging.info(f"Data source: {source}")
    logging.info(f"Properties used in the calculations: {', '.join(active_props)}")

    if calc_type in (CalculationType.ACTUAL_VOLUME, CalculationType.MASS):
        if source == "PFlotran":
            co2_mass_cell = _pflotran_co2mass(
                source_data, co2_molar_mass, water_molar_mass
            )
        else:
            co2_mass_cell = _eclipse_co2mass(source_data, co2_molar_mass)
        if source_data.SGSTRAND is None and source_data.SGTRH is None:
            co2_mass_output = Co2Data(
                source_data.x_coord,
                source_data.y_coord,
                [
                    Co2DataAtTimeStep(
                        key,
                        value[0],
                        value[1],
                        np.zeros_like(value[1]),
                        np.zeros_like(value[1]),
                        np.zeros_like(value[1]),
                    )
                    for key, value in co2_mass_cell.items()
                ],
                "kg",
                source_data.get_zone(),
                source_data.get_region(),
            )
        else:
            co2_mass_output = Co2Data(
                source_data.x_coord,
                source_data.y_coord,
                [
                    Co2DataAtTimeStep(
                        key,
                        value[0],
                        value[1],
                        np.zeros_like(value[1]),
                        value[2],
                        value[3],
                    )
                    for key, value in co2_mass_cell.items()
                ],
                "kg",
                source_data.get_zone(),
                source_data.get_region(),
            )
        if calc_type != CalculationType.MASS:
            if source == "PFlotran":
                y = source_data.get_amfg()[source_data.DATES[0]]
                min_y = np.min(y)
                where_min_amfg = np.where(np.isclose(y, min_y))[0]
                # Where amfg is 0, or the closest approximation available
                dwat = source_data.get_dwat()[source_data.DATES[0]]
                water_density = np.array(
                    [
                        (
                            x[1]
                            if np.isclose((y[x[0]]), 0)
                            else np.mean(dwat[where_min_amfg])
                        )
                        for x in enumerate(dwat)
                    ]
                )
                molar_vols_co2 = _pflotran_co2_molar_volume(
                    source_data,
                    water_density,
                    co2_molar_mass,
                    water_molar_mass,
                )
            else:
                y = source_data.get_xmf2()[source_data.DATES[0]]
                min_y = np.min(y)
                where_min_xmf2 = np.where(np.isclose(y, min_y))[0]
                # Where xmf2 is 0, or the closest approximation available
                bwat = source_data.get_bwat()[source_data.DATES[0]]
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
                co2_mass_output.data_list[t].date: [
                    co2_mass_output.data_list[t].aqu_phase,
                    co2_mass_output.data_list[t].gas_phase,
                ]
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
                        np.zeros_like(np.array(vols_co2[t][1])),
                        np.zeros_like(np.array(vols_co2[t][1])),
                        np.zeros_like(np.array(vols_co2[t][1])),
                    )
                    for t in vols_co2
                ],
                "m3",
                source_data.get_zone(),
                source_data.get_region(),
            )
        else:
            co2_amount = co2_mass_output
    elif calc_type == CalculationType.CELL_VOLUME:
        props_idx = np.where(
            [getattr(source_data, x) is not None for x in props_check]
        )[0]
        props_names = [props_check[i] for i in props_idx]
        plume_props_names = [x for x in props_names if x in ["SGAS", "AMFG", "XMF2"]]
        properties = {x: getattr(source_data, x) for x in plume_props_names}
        inactive_gas_cells = {
            x: _identify_gas_less_cells(
                {x: properties[plume_props_names[0]][x]},
                {x: properties[plume_props_names[1]][x]},
            )
            for x in source_data.DATES
        }
        vols_ext = {
            t: np.array([0] * len(source_data.get_vol()[t])) for t in source_data.DATES
        }
        for date in source_data.DATES:
            vols_ext[date][~inactive_gas_cells[date]] = np.array(
                source_data.get_vol()[date]
            )[~inactive_gas_cells[date]]
        co2_amount = Co2Data(
            source_data.x_coord,
            source_data.y_coord,
            [
                Co2DataAtTimeStep(
                    t,
                    np.zeros_like(np.array(vols_ext[t])),
                    np.zeros_like(np.array(vols_ext[t])),
                    np.array(vols_ext[t]),
                    np.zeros_like(np.array(vols_ext[t])),
                    np.zeros_like(np.array(vols_ext[t])),
                )
                for t in vols_ext
            ],
            "m3",
            source_data.get_zone(),
            source_data.get_region(),
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


def calculate_co2(
    grid_file: str,
    unrst_file: str,
    zone_info: Dict,
    region_info: Dict,
    residual_trapping: bool = False,
    calc_type_input: str = "mass",
    init_file: Optional[str] = None,
) -> Co2Data:
    """
    Calculates the desired amount (calc_type_input) of CO2

    Args:
      grid_file (str): Path to EGRID-file
      unrst_file (str): Path to UNRST-file
      calc_type_input (str): Input string with calculation type to perform
      init_file (str): Path to INIT-file
      zone_info (Dict): Dictionary with zone information
      region_info (Dict): Dictionary with region information
      residual_trapping (bool): Calculate residual trapping or not

    Returns:
      CO2Data

    """

    PROPERTIES_TO_EXTRACT = RELEVANT_PROPERTIES
    if not residual_trapping:
        PROPERTIES_TO_EXTRACT = [
            prop for prop in RELEVANT_PROPERTIES if prop not in ["SGSTRAND", "SGTRH"]
        ]
    source_data = _extract_source_data(
        grid_file, unrst_file, PROPERTIES_TO_EXTRACT, zone_info, region_info, init_file
    )
    calc_type = _set_calc_type_from_input_string(calc_type_input)
    co2_data = _calculate_co2_data_from_source_data(source_data, calc_type=calc_type)
    return co2_data


if __name__ == "__main__":
    pass
