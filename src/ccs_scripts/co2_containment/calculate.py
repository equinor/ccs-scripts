"""CO2 calculation methods"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

from ccs_scripts.co2_containment.co2_calculation import (
    CalculationType,
    Co2Data,
    Co2DataAtTimeStep,
)


@dataclass
class ContainedCo2:
    """
    Dataclass with amount of Co2 in/out a given area for a given phase
    at different time steps

    Args:
        date (str): A given time step
        amount (float): Numerical value with the computed amount at "date"
        phase (Literal): One of gas (or trapped_gas/free_gas)/aqueous/undefined.
            The phase of "amount".
        containment (Literal): One of contained/outside/hazardous. The location
            that "amount" corresponds to.
        zone (str):
        region (str):

    """

    date: str
    amount: float
    phase: str  # Literal["gas", "aqueous", "trapped_gas", "free_gas", "total", "undefined"]
    containment: str  # Literal["contained", "outside", "hazardous", "total"]
    zone: Optional[str] = None
    region: Optional[str] = None

    def __post_init__(self):
        """
        If the slot "data" of a ContainedCo2 object does not contain "-", this
        function converts it to the format yyyy-mm-dd

        """
        if "-" not in self.date:
            date = self.date
            self.date = f"{date[:4]}-{date[4:6]}-{date[6:]}"


# pylint: disable = too-many-arguments
def calculate_co2_containment(
    co2_data: Co2Data,
    containment_polygon: Union[Polygon, MultiPolygon],
    hazardous_polygon: Union[Polygon, MultiPolygon, None],
    zone_info: Dict,
    region_info: Dict,
    calc_type: CalculationType,
    residual_trapping: bool,
) -> List[ContainedCo2]:
    """
    Calculates the amount (mass/volume) of CO2 within given boundaries
    (contained/outside/hazardous) at each time step for each phase
    (aqueous/gaseous). Result is a list of ContainedCo2 objects.

    Args:
        co2_data (Co2Data): Information of the amount of CO2 at each cell in
            each time step
        containment_polygon (Union[Polygon,Multipolygon]): The polygon that defines
            the containment area
        hazardous_polygon (Union[Polygon,Multipolygon]): The polygon that defines
             the hazardous area
        zone_info (Dict): Dictionary containing zone information
        region_info (Dict): Dictionary containing region information
        calc_type (CalculationType): Which calculation is to be performed
             (mass / cell_volume / actual_volume)
        residual_trapping (bool): Indicate if residual trapping is calculated

    Returns:
        List[ContainedCo2]
    """
    logging.info(
        f"Calculate contained CO2 {calc_type.name.lower()} using input polygons"
    )
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
    _log_summary_of_grid_node_location(locations)

    phases = _lists_of_phases(calc_type, residual_trapping)
    zone_map, region_map = _zone_and_region_maps(co2_data, zone_info, region_info)
    zone_region_info = (
        [(None, None, np.ones(len(co2_data.x_coord), dtype=bool))]
        + [(zone, None, is_in_zone) for zone, is_in_zone in zone_map.items()]
        + [(None, region, is_in_region) for region, is_in_region in region_map.items()]
    )
    containment = []
    for zone, region, is_in_section in zone_region_info:
        for w in co2_data.data_list:
            co2_amounts = _lists_of_co2_for_each_phase(w, calc_type, residual_trapping)
            for co2_amount, phase in zip(co2_amounts, phases):
                for location, is_in_location in locations.items():
                    amount = sum(co2_amount[is_in_section & is_in_location])
                    containment += [
                        ContainedCo2(
                            w.date,
                            amount,
                            phase,
                            location,
                            zone,
                            region,
                        )
                    ]
    logging.info(
        f"Done calculating contained CO2 {calc_type.name.lower()} using input polygons"
    )
    return containment


def _log_summary_of_grid_node_location(locations: Dict) -> None:
    logging.info("Number of grid nodes:")
    logging.info(
        f"  * Inside containment polygon                        :\
            {locations['contained'].sum()}"
    )
    logging.info(
        f"  * Inside hazardous polygon                          :\
            {locations['hazardous'].sum()}"
    )
    logging.info(
        f"  * Outside containment polygon and hazardous polygon :\
            {locations['outside'].sum()}"
    )
    logging.info(
        f"  * Total                                             :\
            {len(locations['contained'])}"
    )


def _lists_of_phases(
    calc_type: CalculationType,
    residual_trapping: bool,
) -> List[str]:
    """
    Returns a list of the relevant phases depending on calculation type and whether
    residual trapping is calculated
    """
    if calc_type == CalculationType.CELL_VOLUME:
        phases = ["undefined"]
    else:
        phases = ["total", "aqueous"]
        phases += ["trapped_gas", "free_gas"] if residual_trapping else ["gas"]
    return phases


def _lists_of_co2_for_each_phase(
    co2_at_date: Co2DataAtTimeStep,
    calc_type: CalculationType,
    residual_trapping: bool,
) -> List[np.ndarray]:
    """
    Returns a list of the relevant arrays of different phases of co2 depending on
    calculation type and whether residual trapping is calculated
    """
    if calc_type == CalculationType.CELL_VOLUME:
        arrays = [co2_at_date.volume_coverage]
    else:
        arrays = [co2_at_date.total_mass(), co2_at_date.aqu_phase]
        arrays += (
            [co2_at_date.trapped_gas_phase, co2_at_date.free_gas_phase]
            if residual_trapping
            else [co2_at_date.gas_phase]
        )
    return arrays


def _zone_and_region_maps(
    co2_data: Co2Data, zone_info: Dict, region_info: Dict
) -> Tuple[Dict, Dict]:
    """
    Returns dictionaries connecting each zone/region to a boolean array over the grid,
    indicating whether the grid point belongs to said zone/region
    """
    zone_map = (
        {}
        if co2_data.zone is None
        else (
            {z: np.array(co2_data.zone == z) for z in np.unique(co2_data.zone)}
            if zone_info["int_to_zone"] is None
            else {
                zone_info["int_to_zone"][z]: np.array(co2_data.zone == z)
                for z in np.unique(co2_data.zone)
                if z >= 0 and zone_info["int_to_zone"][z] is not None
            }
        )
    )
    region_map = (
        {}
        if co2_data.region is None
        else (
            {r: np.array(co2_data.region == r) for r in np.unique(co2_data.region)}
            if region_info["int_to_region"] is None
            else {
                region_info["int_to_region"][r]: np.array(co2_data.region == r)
                for r in np.unique(co2_data.region)
                if r >= 0 and region_info["int_to_region"][r] is not None
            }
        )
    )
    return zone_map, region_map


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
