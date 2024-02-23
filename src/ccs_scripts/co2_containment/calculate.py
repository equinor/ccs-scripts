"""CO2 calculation methods"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

from ccs_scripts.co2_containment.co2_calculation import CalculationType, Co2Data


@dataclass
class ContainedCo2:
    """
    Dataclass with amount of Co2 in/out a given area for a given phase
    at different time steps

    Args:
        date (str): A given time step
        amount (float): Numerical value with the computed amount at "date"
        phase (Literal): One of gas/aqueous/undefined. The phase of "amount".
        location (Literal): One of contained/outside/hazardous. The location
            that "amount" corresponds to.
        zone (str):
        region (str):

    """

    date: str
    amount: float
    phase: Literal["gas", "aqueous", "undefined"]
    location: Literal["contained", "outside", "hazardous"]
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

    Returns:
        List[ContainedCo2]
    """
    logging.info(
        f"Calculate contained CO2 {calc_type.name.lower()} using input polygons"
    )
    if containment_polygon is not None:
        is_contained = _calculate_containment(
            co2_data.x_coord,
            co2_data.y_coord,
            containment_polygon,
        )
    else:
        is_contained = np.array([True] * len(co2_data.x_coord))
        logging.info("No containment polygon specified.")
    if hazardous_polygon is not None:
        is_hazardous = _calculate_containment(
            co2_data.x_coord,
            co2_data.y_coord,
            hazardous_polygon,
        )
    else:
        is_hazardous = np.array([False] * len(co2_data.x_coord))
        logging.info("No hazardous polygon specified.")

    # Count as hazardous if the two boundaries overlap:
    is_inside = [x if not y else False for x, y in zip(is_contained, is_hazardous)]
    is_outside = [not x and not y for x, y in zip(is_contained, is_hazardous)]
    logging.info("Number of grid nodes:")
    logging.info(
        f"  * Inside containment polygon                        :\
        {is_inside.count(True)}"
    )
    logging.info(
        f"  * Inside hazardous polygon                          :\
        {list(is_hazardous).count(True)}"
    )
    logging.info(
        f"  * Outside containment polygon and hazardous polygon :\
        {is_outside.count(True)}"
    )
    logging.info(
        f"  * Total                                             :\
        {len(is_inside)}"
    )

    if co2_data.zone is None and co2_data.region is None:
        if calc_type == CalculationType.CELL_VOLUME:
            return [
                c
                for w in co2_data.data_list
                for c in [
                    ContainedCo2(
                        w.date,
                        sum(w.volume_coverage[is_inside]),
                        "undefined",
                        "contained",
                    ),
                    ContainedCo2(
                        w.date,
                        sum(w.volume_coverage[is_outside]),
                        "undefined",
                        "outside",
                    ),
                    ContainedCo2(
                        w.date,
                        sum(w.volume_coverage[is_hazardous]),
                        "undefined",
                        "hazardous",
                    ),
                ]
            ]
        return [
            c
            for w in co2_data.data_list
            for c in [
                ContainedCo2(w.date, sum(w.gas_phase[is_inside]), "gas", "contained"),
                ContainedCo2(w.date, sum(w.gas_phase[is_outside]), "gas", "outside"),
                ContainedCo2(
                    w.date, sum(w.gas_phase[is_hazardous]), "gas", "hazardous"
                ),
                ContainedCo2(
                    w.date, sum(w.aqu_phase[is_inside]), "aqueous", "contained"
                ),
                ContainedCo2(
                    w.date, sum(w.aqu_phase[is_outside]), "aqueous", "outside"
                ),
                ContainedCo2(
                    w.date, sum(w.aqu_phase[is_hazardous]), "aqueous", "hazardous"
                ),
            ]
        ]
    zone_map = (
        {"all": np.array([True] * len(co2_data.x_coord))}
        if co2_data.zone is None
        else (
            {z: co2_data.zone == z for z in np.unique(co2_data.zone)}
            if zone_info["int_to_zone"] is None
            else {
                zone_info["int_to_zone"][z]: co2_data.zone == z
                for z in np.unique(co2_data.zone)
                if z >= 0 and zone_info["int_to_zone"][z] is not None
            }
        )
    )
    region_map = (
        {"all": np.array([True] * len(co2_data.x_coord))}
        if co2_data.region is None
        else (
            {r: co2_data.region == r for r in np.unique(co2_data.region)}
            if region_info["int_to_region"] is None
            else {
                region_info["int_to_region"][r]: co2_data.region == r
                for r in np.unique(co2_data.region)
                if r >= 0 and region_info["int_to_region"][r] is not None
            }
        )
    )
    if calc_type == CalculationType.CELL_VOLUME:
        return [
            c
            for w in co2_data.data_list
            for zn, zm in zone_map.items()
            for rn, rm in region_map.items()
            for c in [
                ContainedCo2(
                    w.date,
                    sum(w.volume_coverage[is_inside & zm & rm]),
                    "gas",
                    "contained",
                    zn,
                    rn,
                ),
                ContainedCo2(
                    w.date,
                    sum(w.volume_coverage[is_outside & zm & rm]),
                    "gas",
                    "outside",
                    zn,
                    rn,
                ),
                ContainedCo2(
                    w.date,
                    sum(w.volume_coverage[is_hazardous & zm & rm]),
                    "gas",
                    "hazardous",
                    zn,
                    rn,
                ),
            ]
        ]
    logging.info(
        f"Done calculating contained CO2 {calc_type.name.lower()} using input polygons"
    )
    return [
        c
        for w in co2_data.data_list
        for zn, zm in zone_map.items()
        for rn, rm in region_map.items()
        for c in [
            ContainedCo2(
                w.date,
                sum(w.gas_phase[is_inside & zm & rm]),
                "gas",
                "contained",
                zn,
                rn,
            ),
            ContainedCo2(
                w.date, sum(w.gas_phase[is_outside & zm & rm]), "gas", "outside", zn, rn
            ),
            ContainedCo2(
                w.date,
                sum(w.gas_phase[is_hazardous & zm & rm]),
                "gas",
                "hazardous",
                zn,
                rn,
            ),
            ContainedCo2(
                w.date,
                sum(w.aqu_phase[is_inside & zm & rm]),
                "aqueous",
                "contained",
                zn,
                rn,
            ),
            ContainedCo2(
                w.date,
                sum(w.aqu_phase[is_outside & zm & rm]),
                "aqueous",
                "outside",
                zn,
                rn,
            ),
            ContainedCo2(
                w.date,
                sum(w.aqu_phase[is_hazardous & zm & rm]),
                "aqueous",
                "hazardous",
                zn,
                rn,
            ),
        ]
    ]


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
