from enum import Enum
from typing import List, Optional

import numpy as np
import xtgeo

from ccs_scripts.aggregate._config import CO2MassSettings
from ccs_scripts.co2_containment.co2_calculation import (
    Co2Data,
    Co2DataAtTimeStep,
)
from ccs_scripts.co2_containment.source_data import Scenario
from ccs_scripts.utils.xtgeo_logging import setup_xtgeo_logging

setup_xtgeo_logging()

CO2_MASS_PNAME = "CO2Mass"

# pylint: disable=invalid-name,too-many-instance-attributes


class MapName(Enum):
    MASS_TOT = "co2_mass_total"
    MASSDISW = "co2_mass_dissolved_water_phase"
    MASSDISO = "co2_mass_dissolved_oil_phase"
    MASS_GAS = "co2_mass_gas_phase"
    MASSTGAS = "co2_mass_trapped_gas_phase"
    MASSFGAS = "co2_mass_free_gas_phase"
    MigrationTime_MASS_TOT = "co2_mass_migration_time_total"


def translate_co2data_to_gridproperties(
    co2_data: Co2Data,
    grid_file: str,
    co2_mass_settings: CO2MassSettings,
    grid: Optional[xtgeo.Grid] = None,
) -> List[xtgeo.GridProperty]:
    """
    Convert CO2 data into in-memory 3D GridProperty objects.
    """
    maps = co2_mass_settings.maps
    if maps is None:
        maps = []
    elif isinstance(maps, str):
        maps = [maps]
    maps = [map_name.lower() for map_name in maps]

    store_all = "all" in maps or len(maps) == 0
    if grid is None:
        grid = xtgeo.grid_from_file(grid_file)
    property_template = xtgeo.GridProperty(grid)

    out: List[xtgeo.GridProperty] = []
    for co2_at_date in co2_data.data_list:
        tmp_props: dict[MapName, xtgeo.GridProperty] = _convert_to_grid(
            co2_at_date, property_template, co2_data.active_cells
        )
        if store_all or "total_co2" in maps:
            out.append(tmp_props[MapName.MASS_TOT])
        if store_all or "dissolved_water_co2" in maps:
            out.append(tmp_props[MapName.MASSDISW])
        if (
            store_all or "dissolved_oil_co2" in maps
        ) and co2_data.scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
            out.append(tmp_props[MapName.MASSDISO])
        if (
            store_all or "free_co2" in maps
        ) and not co2_mass_settings.residual_trapping:
            out.append(tmp_props[MapName.MASS_GAS])
        if (store_all or "free_co2" in maps) and co2_mass_settings.residual_trapping:
            out.append(tmp_props[MapName.MASSFGAS])
            out.append(tmp_props[MapName.MASSTGAS])
    return out


def _convert_to_grid(
    co2_at_date: Co2DataAtTimeStep,
    property_template: xtgeo.GridProperty,
    active_cells: np.ndarray,
) -> dict[MapName, xtgeo.GridProperty]:
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

    def _create_prop(name: MapName, data: np.ndarray) -> xtgeo.GridProperty:
        prop = property_template.copy(newname=name.value)
        prop.date = co2_at_date.date
        prop.values[active_cells] = data
        return prop

    props = {
        m: _create_prop(m, mass)
        for m, mass in [
            (MapName.MASS_TOT, co2_at_date.total_mass()),
            (MapName.MASSDISW, co2_at_date.dis_water_phase),
            (MapName.MASSDISO, co2_at_date.dis_oil_phase),
            (MapName.MASS_GAS, co2_at_date.gas_phase),
            (MapName.MASSTGAS, co2_at_date.trapped_gas_phase),
            (MapName.MASSFGAS, co2_at_date.free_gas_phase),
        ]
    }

    return props
