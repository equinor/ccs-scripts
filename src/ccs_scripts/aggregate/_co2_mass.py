import logging
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import resfo
import xtgeo

from ccs_scripts.aggregate._config import CO2MassSettings
from ccs_scripts.co2_containment.co2_calculation import (
    Co2Data,
    Co2DataAtTimeStep,
)
from ccs_scripts.co2_containment.source_data import Scenario
from ccs_scripts.utils.utils import format_error
from ccs_scripts.utils.xtgeo_logging import setup_xtgeo_logging

setup_xtgeo_logging()

logger = logging.getLogger(__name__)

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
    co2_mass_settings: CO2MassSettings,
    grid: xtgeo.Grid,
    grid_out_dir: str | None = None,
    date_indices: list[int] | None = None,
) -> list[xtgeo.GridProperty]:
    """
    Convert CO2 data into in-memory 3D GridProperty objects.

    When ``grid_out_dir`` is set, also write one EGRID/UNRST pair containing
    the selected mass properties. The returned properties are unchanged.
    """
    maps = co2_mass_settings.maps
    if maps is None:
        maps = []
    elif isinstance(maps, str):
        maps = [maps]
    maps = [map_name.lower() for map_name in maps]

    store_all = "all" in maps or len(maps) == 0
    property_template = xtgeo.GridProperty(grid)

    out: list[xtgeo.GridProperty] = []
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

    if grid_out_dir is not None:
        _write_gridproperties(
            out,
            grid,
            co2_mass_settings.unrst_source,
            grid_out_dir,
            date_indices,
        )
    return out


def _write_gridproperties(
    properties: list[xtgeo.GridProperty],
    grid: xtgeo.Grid,
    unrst_file: str,
    grid_out_dir: str,
    date_indices: list[int] | None,
) -> None:
    """Write selected in-memory properties as EGRID/UNRST property series."""
    output_dir = _prepare_grid_output_directory(grid_out_dir)
    if date_indices is None:
        date_indices = list(range(len({prop.date for prop in properties})))

    property_dates = list(dict.fromkeys(prop.date for prop in properties))
    if len(date_indices) != len(property_dates):
        raise ValueError(
            format_error(
                "Unable to write CO2 mass grid files, problem with UNRST date values"
            )
        )
    source_index_by_date = dict(zip(property_dates, date_indices))

    restart_headers = _restart_headers_for_grid(unrst_file, date_indices, grid)
    grid_active = grid.actnum_array.astype(bool).ravel(order="F")

    properties_by_date: dict[str, list[xtgeo.GridProperty]] = {}
    for prop in properties:
        if prop.name is None or prop.date is None:
            raise ValueError(
                format_error("CO2 mass properties must have a name and date")
            )
        properties_by_date.setdefault(prop.date, []).append(prop)

    restart_keywords: list[tuple[str, Any]] = []
    for property_date, properties_at_date in properties_by_date.items():
        date_index = source_index_by_date[property_date]
        intehead, logihead = restart_headers[date_index]
        restart_keywords.extend(
            [
                ("SEQNUM  ", [np.int32(date_index)]),
                ("INTEHEAD", intehead),
                ("LOGIHEAD", logihead),
            ]
        )
        for prop in properties_at_date:
            keyword_name = MapName(prop.name).name
            restart_keywords.append(
                (keyword_name, prop.values.data.ravel(order="F")[grid_active])
            )

    resfo.write(output_dir / "co2_mass.UNRST", restart_keywords)
    grid.to_file(output_dir / "co2_mass.EGRID", fformat="egrid")


def _prepare_grid_output_directory(grid_out_dir: str) -> Path:
    output_dir = Path(grid_out_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(
                format_error(f"3D grid output path is not a directory: {output_dir}")
            )
        return output_dir

    parent_dir = output_dir.parent
    if not parent_dir.exists():
        raise FileNotFoundError(
            format_error(
                f"Parent directory for 3D grid output does not exist: {parent_dir}"
            )
        )
    output_dir.mkdir()
    logger.info("\nCreated new grid folder: %s", output_dir)
    return output_dir


def _restart_headers_for_grid(
    unrst_file: str,
    date_indices: list[int],
    grid: xtgeo.Grid,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    expected_dimensions = (grid.ncol, grid.nrow, grid.nlay)
    expected_active = int(np.count_nonzero(grid.actnum_array))
    requested_indices = set(date_indices)
    headers: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    report_index = -1
    intehead: np.ndarray | None = None

    for entry in resfo.lazy_read(unrst_file):
        keyword = entry.read_keyword().strip()
        if keyword == "SEQNUM":
            report_index += 1
            intehead = None
        elif report_index not in requested_indices:
            continue
        elif keyword == "INTEHEAD":
            intehead = np.asarray(entry.read_array())
        elif keyword == "LOGIHEAD" and intehead is not None:
            logihead = np.asarray(entry.read_array())
            nx, ny, nz, active_count = map(int, intehead[8:12])
            if (nx, ny, nz) == expected_dimensions and active_count == expected_active:
                headers[report_index] = (intehead, logihead)
            intehead = None

    missing_indices = requested_indices - headers.keys()
    if missing_indices:
        raise ValueError(
            format_error(
                "Could not find restart headers matching grid "
                f"{expected_dimensions} with {expected_active} active cells "
                f"at restart indices {sorted(missing_indices)}"
            )
        )
    return headers


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
