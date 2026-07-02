"""CO2 calculation methods"""

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.prepared import prep

from ccs_scripts.co2_containment.co2_calculation import (
    Co2Data,
    Co2DataAtTimeStep,
)
from ccs_scripts.co2_containment.input import CalculationType
from ccs_scripts.co2_containment.source_data import Scenario
from ccs_scripts.utils.timer import Timer


@dataclass
class ContainedCo2:
    """
    Dataclass with amount of Co2 in/out a given area for a given phase
    at different time steps

    Args:
        date (str): A given time step
        amount (float): Numerical value with the computed amount at "date"
        phase (Literal): One of gas (or trapped_gas/free_gas)/dissolved/undefined.
            The phase of "amount".
        containment (Literal): One of contained/outside/nogo. The location
            that "amount" corresponds to.
        zone (str):
        region (str):
        plume (str): The plume group (a single injection well or a list of wells)

    """

    date: str
    amount: np.float64
    phase: str
    containment: str
    zone: Optional[str] = None
    region: Optional[str] = None
    plume_group: Optional[str] = None

    def __post_init__(self):
        """
        If the slot "data" of a ContainedCo2 object does not contain "-", this
        function converts it to the format yyyy-mm-dd

        """
        if "-" not in self.date:
            date = self.date
            self.date = f"{date[:4]}-{date[4:6]}-{date[6:]}"


def _construct_containment_table(
    contained_co2: List[ContainedCo2],
) -> pd.DataFrame:
    """
    Creates a data frame from calculated CO2 data.

    Args:
        contained_co2 (list of ContainedCo2): CO2 data divided into phases/locations

    Returns:
        pd.DataFrame
    """
    records = [asdict(c) for c in contained_co2]
    return pd.DataFrame.from_records(records)


# pylint: disable = too-many-arguments, too-many-locals
def _calculate_co2_containment(
    co2_data: Co2Data,
    containment_polygon: Union[Polygon, MultiPolygon],
    nogo_polygon: Optional[Union[Polygon, MultiPolygon]],
    int_to_zone: Optional[List[Optional[str]]],
    int_to_region: Optional[List[Optional[str]]],
    calc_type: CalculationType,
    residual_trapping: Optional[bool] = False,
    plume_groups: Optional[List[List[str]]] = None,
) -> List[ContainedCo2]:
    """
    Calculates the amount (mass/volume) of CO2 within given boundaries
    (contained/outside/nogo) at each time step for each phase
    (dissolved/gaseous). Result is a list of ContainedCo2 objects.

    Args:
        co2_data (Co2Data): Information of the amount of CO2 at each cell in
            each time step
        containment_polygon (Union[Polygon,Multipolygon]): The polygon that defines
            the containment area
        nogo_polygon (Union[Polygon,Multipolygon]): The polygon that defines
             the nogo area
        int_to_zone (List): List of zone names
        int_to_region (List): List of region names
        calc_type (CalculationType): Which calculation is to be performed
             (mass / cell_volume / actual_volume)
        residual_trapping (Optional[bool]): Indicate if residual trapping should be calculated
        plume_groups (Optional[List[List[str]]]): For each time step, plume group for each grid cell

    Returns:
        List[ContainedCo2]
    """
    timer = Timer()
    logging.info(
        f"Calculate contained CO2 {calc_type.name.lower()} using input polygons"
    )

    timer.start("make_location_filters", "calculate_co2_containment")
    # Dict with boolean arrays indicating location
    locations = _make_location_filters(
        co2_data,
        containment_polygon,
        nogo_polygon,
    )
    timer.stop("make_location_filters")
    _log_summary_of_grid_node_location(locations)
    phases = _lists_of_phases(calc_type, co2_data.scenario, residual_trapping)

    # List of tuple with (zone/None, None/region, boolean array over grid)
    zone_region_info = _zone_and_region_mapping(co2_data, int_to_zone, int_to_region)

    if plume_groups is not None:
        plume_groups = [
            [x if x != "" else "undetermined" for x in y] for y in plume_groups
        ]
        plume_names = set(name for values in plume_groups for name in values)
    else:
        plume_names = set()

    group_entries, group_masks = _build_group_masks(zone_region_info, locations)

    containment = []
    dtype = np.int64 if calc_type == CalculationType.CELL_VOLUME else np.float64
    n_cells = len(co2_data.x_coord)
    for i, co2_at_timestep in enumerate(co2_data.data_list):
        co2_amounts_for_each_phase = _lists_of_co2_for_each_phase(
            co2_at_timestep,
            calc_type,
            residual_trapping,
        )
        if plume_groups is not None:
            timer.start("plume_group_mapping", "calculate_co2_containment")
            plume_group_info = _plume_group_mapping(plume_names, plume_groups[i])
            plume_names_at_t, plume_mask_matrix = _plume_masks_as_matrix(
                plume_group_info
            )
            timer.stop("plume_group_mapping")
        else:
            plume_names_at_t = ["all"]
            plume_mask_matrix = np.ones((1, n_cells), dtype=bool)
        # Stack phase arrays once (n_phases x n_cells).
        phase_matrix = np.vstack(
            [np.asarray(arr, dtype=dtype) for arr in co2_amounts_for_each_phase]
        )
        timer.start("sum_and_store", "calculate_co2_containment")
        # Vectorized grouped reductions:
        # (n_phases x n_cells) @ (n_cells x n_groups) -> (n_phases x n_groups)
        for plume_idx, plume_name in enumerate(plume_names_at_t):
            combined_masks = group_masks & plume_mask_matrix[plume_idx]
            sums = phase_matrix @ combined_masks.T
            for group_idx, (zone, region, location) in enumerate(group_entries):
                for phase_idx, phase in enumerate(phases):
                    containment.append(
                        ContainedCo2(
                            co2_at_timestep.date,
                            np.float64(sums[phase_idx, group_idx]),
                            phase,
                            location,
                            zone,
                            region,
                            plume_name,
                        )
                    )
        timer.stop("sum_and_store")
    logging.info(f"Done calculating contained CO2 {calc_type.name.lower()}")
    return containment


def _build_group_masks(
    zone_region_info: List,
    locations: Dict[str, np.ndarray],
) -> tuple[List[tuple[Optional[str], Optional[str], str]], np.ndarray]:
    """
    Build static (zone/region x location) masks

    Returns:
      - group entries with labels
      - stacked bool mask matrix with shape (n_groups, n_cells)
    """
    group_entries: List[tuple[Optional[str], Optional[str], str]] = []
    group_masks: List[np.ndarray] = []
    for zone, region, is_in_section in zone_region_info:
        section_mask = np.asarray(is_in_section, dtype=bool)
        for location, is_in_location in locations.items():
            group_entries.append((zone, region, location))
            group_masks.append(section_mask & np.asarray(is_in_location, dtype=bool))

    if not group_masks:
        return group_entries, np.zeros((0, 0), dtype=bool)

    return group_entries, np.vstack(group_masks)


def _plume_masks_as_matrix(
    plume_group_info: Dict[str, np.ndarray],
) -> tuple[List[str], np.ndarray]:
    """
    Convert plume mapping dictionary to deterministic name list and stacked masks.
    """
    names = list(plume_group_info.keys())
    if not names:
        return ["all"], np.ones((1, 0), dtype=bool)
    masks = np.vstack(
        [np.asarray(plume_group_info[name], dtype=bool) for name in names]
    )
    return names, masks


def _make_location_filters(
    co2_data: Co2Data,
    containment_polygon: Union[Polygon, MultiPolygon],
    nogo_polygon: Union[Polygon, MultiPolygon, None],
) -> Dict:
    """
    Return a dictionary connecting location (contained/outside/nogo) to boolean
    arrays over all grid nodes indicating membership to said location
    """
    locations = {}
    if containment_polygon is not None:
        locations["contained"] = _calculate_containment(
            co2_data.x_coord,
            co2_data.y_coord,
            containment_polygon,
        )
    else:
        locations["contained"] = np.ones(len(co2_data.x_coord), dtype=bool)
        logging.info("Containment polygon not specified.")
    if nogo_polygon is not None:
        locations["nogo"] = _calculate_containment(
            co2_data.x_coord,
            co2_data.y_coord,
            nogo_polygon,
        )
    else:
        locations["nogo"] = np.zeros(len(co2_data.x_coord), dtype=bool)
        logging.info("No-go polygon not specified.")

    # Count as no-go if the two boundaries overlap.
    locations["contained"] = np.logical_and(locations["contained"], ~locations["nogo"])
    locations["outside"] = np.logical_not(
        np.logical_or(locations["contained"], locations["nogo"])
    )
    locations["total"] = np.ones(len(co2_data.x_coord), dtype=bool)
    return locations


def _log_summary_of_grid_node_location(locations: Dict) -> None:
    logging.info("Number of grid nodes:")
    logging.info(
        "  * Inside containment polygon                        :"
        f"{locations['contained'].sum():>10}"
    )
    logging.info(
        "  * Inside no-go polygon                              :"
        f"{locations['nogo'].sum():>10}"
    )
    logging.info(
        "  * Outside containment polygon and no-go polygon     :"
        f"{locations['outside'].sum():>10}"
    )
    logging.info(
        "  * Total                                             :"
        f"{len(locations['contained']):>10}"
    )


def _lists_of_phases(
    calc_type: CalculationType,
    scenario: Scenario,
    residual_trapping: Optional[bool] = False,
) -> List[str]:
    """
    Returns a list of the relevant phases depending on calculation type and whether
    residual trapping should be calculated
    """
    if calc_type == CalculationType.CELL_VOLUME:
        phases = ["undefined"]
    else:
        phases = ["total", "dissolved_water"]
        phases += ["trapped_gas", "free_gas"] if residual_trapping else ["gas"]
        phases += (
            ["dissolved_oil"] if scenario == Scenario.DEPLETED_OIL_GAS_FIELD else []
        )
    return phases


def _lists_of_co2_for_each_phase(
    co2_at_date: Co2DataAtTimeStep,
    calc_type: CalculationType,
    residual_trapping: Optional[bool] = False,
) -> List[np.ndarray]:
    """
    Returns a list of the relevant arrays of different phases of co2 depending on
    calculation type and whether residual trapping should be calculated
    """
    if calc_type == CalculationType.CELL_VOLUME:
        arrays = [co2_at_date.volume_coverage]
    else:
        arrays = [co2_at_date.total_mass(), co2_at_date.dis_water_phase]
        arrays += (
            [co2_at_date.trapped_gas_phase, co2_at_date.free_gas_phase]
            if residual_trapping
            else [co2_at_date.gas_phase]
        )
        arrays += [co2_at_date.dis_oil_phase]
    return arrays


def _zone_map(co2_data: Co2Data, int_to_zone: Optional[List[Optional[str]]]) -> Dict:
    """
    Returns a dictionary connecting each zone to a boolean array over the grid,
    indicating whether the grid point belongs to said zone
    """
    if co2_data.zone is None:
        return {}
    if int_to_zone is None:
        return {z: np.array(co2_data.zone == z) for z in np.unique(co2_data.zone)}
    return {
        int_to_zone[z]: np.array(co2_data.zone == z)
        for z in range(len(int_to_zone))
        if int_to_zone[z] is not None
    }


def _region_map(
    co2_data: Co2Data, int_to_region: Optional[List[Optional[str]]]
) -> Dict:
    """
    Returns a dictionary connecting each region to a boolean array over the grid,
    indicating whether the grid point belongs to said region
    """
    if co2_data.region is None:
        return {}
    if int_to_region is None:
        return {r: np.array(co2_data.region == r) for r in np.unique(co2_data.region)}
    return {
        int_to_region[r]: np.array(co2_data.region == r)
        for r in range(len(int_to_region))
        if int_to_region[r] is not None
    }


def _plume_group_mapping(plume_names: Set[str], plume_groups: List[str]):
    np_plume_groups = np.asarray(plume_groups)
    out = {"all": np.ones(len(plume_groups), dtype=bool)}
    out.update({plume: np_plume_groups == plume for plume in plume_names})
    return out


def _zone_and_region_mapping(
    co2_data: Co2Data,
    int_to_zone: Optional[List[Optional[str]]],
    int_to_region: Optional[List[Optional[str]]],
) -> List:
    """
    List containing a tuple for each zone / region (and no zone, no region),
    with the name of the respective zone / region and a boolean array
    indicating membership of each grid node to the zone / region
    """
    zone_map = _zone_map(co2_data, int_to_zone)
    region_map = _region_map(co2_data, int_to_region)
    return (
        [(None, None, np.ones(len(co2_data.x_coord), dtype=bool))]
        + [(zone, None, is_in_zone) for zone, is_in_zone in zone_map.items()]
        + [(None, region, is_in_region) for region, is_in_region in region_map.items()]
    )


def _calculate_containment(
    x_coord: np.ndarray, y_coord: np.ndarray, poly: Union[Polygon, MultiPolygon]
) -> np.ndarray:
    """
    Determines if (x,y) coordinates belong to a given polygon.

    Args:
        x_coord (np.ndarray): x coordinates
        y_coord (np.ndarray): y coordinates
        poly (Union[Polygon, MultiPolygon]): The polygon that determines the
                                             containment of the (x,y) coordinates

    Returns:
        np.ndarray
    """
    # Prefer vectorized operations when available (shapely>=2).
    # Fall back to prepared geometry for compatibility and lower overhead.
    try:
        points = shapely.points(x_coord, y_coord)
        return np.asarray(shapely.contains(poly, points), dtype=bool)
    except Exception:
        prepared = prep(poly)
        return np.fromiter(
            (prepared.contains(Point(_x, _y)) for _x, _y in zip(x_coord, y_coord)),
            dtype=bool,
            count=len(x_coord),
        )


def calculate_containment(
    co2_data: Co2Data,
    cont_polygon: Polygon,
    nogo_polygon: Optional[Polygon],
    calc_type: CalculationType,
    int_to_zone: Optional[List[Optional[str]]],
    int_to_region: Optional[List[Optional[str]]],
    residual_trapping: Optional[bool] = False,
    plume_groups: Optional[List[List[str]]] = None,
) -> Union[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Use polygons (inside / outside / nogo) and/or regions and/or zones
    and/or plume groups to divide co2 mass or volume into different categories.
    Result is a data frame.

    Args:
        co2_data (Co2Data): Mass/volume of CO2 at each time step
        cont_polygon (Polygon): Polygon defining the containment area
        nogo_polygon (Optional[Polygon]): Polygon defining the nogo area
        calc_type (CalculationType): Choose mass / cell_volume / actual_volume
        int_to_zone (Optional[List[Optional[str]]]): List of zone names
        int_to_region (Optional[List[Optional[str]]]): List of region names
        residual_trapping (Optional[bool]): Indicate if residual trapping should be calculated
        plume_groups (Optional[List[List[str]]]): For each time step, plume group for each grid cell

    Returns:
        Union[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]
    """
    timer = Timer()
    timer.start("calculate_co2_containment")
    contained_co2 = _calculate_co2_containment(
        co2_data,
        cont_polygon,
        nogo_polygon,
        int_to_zone,
        int_to_region,
        calc_type,
        residual_trapping,
        plume_groups,
    )
    containment_table = _construct_containment_table(contained_co2)
    timer.stop("calculate_co2_containment")
    return containment_table
