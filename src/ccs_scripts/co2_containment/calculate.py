"""CO2 calculation methods"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

from ccs_scripts.co2_containment.co2_calculation import (
    CalculationType,
    Co2Data,
    Co2DataAtTimeStep,
    Scenario,
)


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
        containment (Literal): One of contained/outside/hazardous. The location
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


# pylint: disable = too-many-arguments, too-many-locals
def calculate_co2_containment(
    co2_data: Co2Data,
    containment_polygon: Union[Polygon, MultiPolygon],
    hazardous_polygon: Union[Polygon, MultiPolygon, None],
    int_to_zone: Optional[List[Optional[str]]],
    int_to_region: Optional[List[Optional[str]]],
    calc_type: CalculationType,
    residual_trapping: bool,
    plume_groups: Optional[List[List[str]]] = None,
) -> List[ContainedCo2]:
    """
    Calculates the amount (mass/volume) of CO2 within given boundaries
    (contained/outside/hazardous) at each time step for each phase
    (dissolved/gaseous). Result is a list of ContainedCo2 objects.

    Args:
        co2_data (Co2Data): Information of the amount of CO2 at each cell in
            each time step
        containment_polygon (Union[Polygon,Multipolygon]): The polygon that defines
            the containment area
        hazardous_polygon (Union[Polygon,Multipolygon]): The polygon that defines
             the hazardous area
        int_to_zone (List): List of zone names
        int_to_region (List): List of region names
        calc_type (CalculationType): Which calculation is to be performed
             (mass / cell_volume / actual_volume)
        residual_trapping (bool): Indicate if residual trapping should be calculated
        plume_groups (List): For each time step, list of plume group for each grid cell

    Returns:
        List[ContainedCo2]
    """
    logging.info(
        f"Calculate contained CO2 {calc_type.name.lower()} using input polygons"
    )

    # Dict with boolean arrays indicating location
    locations = _make_location_filters(
        co2_data,
        containment_polygon,
        hazardous_polygon,
    )
    _log_summary_of_grid_node_location(locations)
    phases = _lists_of_phases(calc_type, residual_trapping, co2_data.scenario)

    # List of tuple with (zone/None, None/region, boolean array over grid)
    zone_region_info = _zone_and_region_mapping(co2_data, int_to_zone, int_to_region)

    if plume_groups is not None:
        plume_groups = [
            [x if x != "" else "undetermined" for x in y] for y in plume_groups
        ]
        plume_names = set(name for values in plume_groups for name in values)
    else:
        plume_names = set()

    containment = []
    for zone, region, is_in_section in zone_region_info:
        for location, is_in_location in locations.items():
            for i, co2_at_timestep in enumerate(co2_data.data_list):
                co2_amounts_for_each_phase = _lists_of_co2_for_each_phase(
                    co2_at_timestep,
                    calc_type,
                    residual_trapping,
                )

                if plume_groups is not None:
                    plume_group_info = _plume_group_mapping(
                        plume_names, plume_groups[i]
                    )
                else:
                    plume_group_info = {
                        "all": np.ones(len(co2_data.x_coord), dtype=bool)
                    }
                for plume_name, is_in_plume in plume_group_info.items():
                    for co2_amount, phase in zip(co2_amounts_for_each_phase, phases):
                        dtype = (
                            np.int64
                            if calc_type == CalculationType.CELL_VOLUME
                            else np.float64
                        )
                        amount = np.sum(
                            co2_amount[is_in_section & is_in_location & is_in_plume],
                            dtype=dtype,
                        )
                        containment += [
                            ContainedCo2(
                                co2_at_timestep.date,
                                np.float64(amount),
                                phase,
                                location,
                                zone,
                                region,
                                plume_name,
                            )
                        ]
    logging.info(f"Done calculating contained CO2 {calc_type.name.lower()}")
    return containment


def _make_location_filters(
    co2_data: Co2Data,
    containment_polygon: Union[Polygon, MultiPolygon],
    hazardous_polygon: Union[Polygon, MultiPolygon, None],
) -> Dict:
    """
    Return a dictionary connecting location (contained/outside/hazardous) to boolean
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
        logging.info("No containment polygon specified.")
    if hazardous_polygon is not None:
        locations["hazardous"] = _calculate_containment(
            co2_data.x_coord,
            co2_data.y_coord,
            hazardous_polygon,
        )
    else:
        locations["hazardous"] = np.zeros(len(co2_data.x_coord), dtype=bool)
        logging.info("No hazardous polygon specified.")

    # Count as hazardous if the two boundaries overlap:
    locations["contained"] = np.array(
        [
            x if not y else False
            for x, y in zip(locations["contained"], locations["hazardous"])
        ]
    )
    locations["outside"] = np.array(
        [
            not x and not y
            for x, y in zip(locations["contained"], locations["hazardous"])
        ]
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
        "  * Inside hazardous polygon                          :"
        f"{locations['hazardous'].sum():>10}"
    )
    logging.info(
        "  * Outside containment polygon and hazardous polygon :"
        f"{locations['outside'].sum():>10}"
    )
    logging.info(
        "  * Total                                             :"
        f"{len(locations['contained']):>10}"
    )


def _lists_of_phases(
    calc_type: CalculationType,
    residual_trapping: bool,
    scenario: Scenario,
) -> List[str]:
    """
    Returns a list of the relevant phases depending on calculation type and whether
    residual trapping should be calculated
    """
    if calc_type == CalculationType.CELL_VOLUME:
        phases = ["undefined"]
    else:
        phases = ["total", "dissolved"]
        phases += ["trapped_gas", "free_gas"] if residual_trapping else ["gas"]
        phases += ["oil"] if scenario == Scenario.DEPLETED_OIL_GAS_FIELD else []
    return phases


def _lists_of_co2_for_each_phase(
    co2_at_date: Co2DataAtTimeStep,
    calc_type: CalculationType,
    residual_trapping: bool,
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
    elif int_to_zone is None:
        return {z: np.array(co2_data.zone == z) for z in np.unique(co2_data.zone)}
    else:
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
    elif int_to_region is None:
        return {r: np.array(co2_data.region == r) for r in np.unique(co2_data.region)}
    else:
        return {
            int_to_region[r]: np.array(co2_data.region == r)
            for r in range(len(int_to_region))
            if int_to_region[r] is not None
        }


def _plume_group_mapping(plume_names: Set[str], plume_groups: List[str]):
    out = {"all": np.ones(len(plume_groups), dtype=bool)}
    out.update(
        {plume: np.array([x == plume for x in plume_groups]) for plume in plume_names}
    )
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
    return np.array([poly.contains(Point(_x, _y)) for _x, _y in zip(x_coord, y_coord)])
