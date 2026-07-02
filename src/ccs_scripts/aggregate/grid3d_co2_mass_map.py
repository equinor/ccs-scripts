#!/usr/bin/env python
import logging
import sys
from typing import Dict, List, Optional

import yaml
from xtgeo import Grid, GridProperty

from ccs_scripts.aggregate import (
    _config,
    _parser,
    grid3d_aggregate_map,
    grid3d_migration_time,
)
from ccs_scripts.aggregate._co2_mass import (
    MapName,
    translate_co2data_to_gridproperties,
)
from ccs_scripts.aggregate._config import AggregationMethod, RootConfig
from ccs_scripts.aggregate._utils import log_input_configuration
from ccs_scripts.co2_containment.co2_calculation import (
    calculate_co2,
)
from ccs_scripts.co2_containment.input import CalculationType, RegionInfo, ZoneInfo
from ccs_scripts.co2_containment.source_data import extract_source_data
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import format_error, format_warning


def generate_co2_mass_maps(config_: RootConfig):
    """
    Calculates and exports 2D and 3D CO2 mass properties from the provided config file

    Args:
        config_: Arguments in the config file
    """
    assert config_.co2_mass_settings is not None
    co2_mass_settings = config_.co2_mass_settings
    grid_file = config_.input.grid
    zone_info = ZoneInfo(
        source=None,
        zranges=None,
        int_to_zone=None,
    )
    region_info = RegionInfo(
        source=None,
        int_to_region=None,
        property_name=None,
    )
    logging.info("\nCalculate CO2 mass 3D grid")
    source_data, grid = extract_source_data(
        grid_file,
        co2_mass_settings.unrst_source,
        zone_info,
        region_info,
        co2_mass_settings.residual_trapping,
        co2_mass_settings.init_source,
        return_grid=True,
    )
    co2_data = calculate_co2(
        source_data,
        CalculationType.MASS,
        co2_mass_settings.residual_trapping,
        co2_mass_settings.cirrus_info_file,
    )

    dates = config_.input.dates
    if len(dates) > 0:
        co2_data.data_list = [x for x in co2_data.data_list if x.date in dates]
    # Keep 3D properties in memory for aggregation.
    in_memory_properties = translate_co2data_to_gridproperties(
        co2_data,
        grid_file,
        co2_mass_settings,
        grid=grid,
    )
    co2_mass_property_to_map_in_memory(config_, in_memory_properties, grid)

    # Migration time maps from the same in-memory properties
    if co2_mass_settings.calculate_migration_time_map:
        _co2_mass_migration_time_in_memory(
            config_, in_memory_properties, co2_mass_settings, co2_data.cell_size
        )


def _co2_mass_migration_time_in_memory(
    config_: RootConfig,
    properties: List[GridProperty],
    co2_mass_settings: _config.CO2MassSettings,
    cell_size: Optional[float] = None,
):
    """
    Compute migration time maps directly from in-memory CO2 mass properties.
    """
    from ccs_scripts.aggregate._migration_time import generate_migration_time_property

    if co2_mass_settings.migration_time_threshold is not None:
        threshold = co2_mass_settings.migration_time_threshold
    elif cell_size is not None:
        factor = 0.1
        threshold = factor * cell_size * 0.001  # From kg to tons
    else:
        threshold = 0.01

    logging.info(
        f"\nThreshold for co2 total mass migration time maps: {threshold:.2f} tons"
    )

    # Filter for total mass properties only
    mass_tot_props = [p for p in properties if MapName.MASS_TOT.value in (p.name or "")]
    if not mass_tot_props:
        logging.info("No total mass properties found for migration time calculation")
        return

    first_injection_year = (
        config_.migration_time_settings.first_injection_year
        if config_.migration_time_settings is not None
        else None
    )

    t_prop = generate_migration_time_property(
        mass_tot_props, threshold, first_injection_year
    )

    # Set up config for migration time aggregation
    config_.computesettings.aggregation = _config.AggregationMethod.MIN
    config_.output.aggregation_tag = False
    config_.output.replace_masked_with_zero = False
    config_.computesettings.aggregate_map = True
    config_.computesettings.indicator_map = False

    grid3d_migration_time.migration_time_property_to_map_in_memory(config_, t_prop)


def co2_mass_property_to_map_in_memory(
    config_: RootConfig,
    properties: List[GridProperty],
    grid: Grid,
):
    """
    Aggregate already loaded CO2 mass 3D properties without writing temp .grd files.
    """
    config_.input.properties = []
    for _ in properties:
        config_.input.properties.append(
            _config.Property(
                source="in_memory",
                name=None,
                lower_threshold=1e-6,  # 0.001 kg
            )
        )

    grid3d_aggregate_map.generate_maps(
        config_.input,
        config_.zonation,
        config_.computesettings,
        config_.mapsettings,
        config_.output,
        preloaded_properties=properties,
        preloaded_grid=grid,
    )


def read_yml_file(file_path: str) -> Dict[str, List]:
    """
    Reads a yml from a given path in file_path argument
    """
    with open(file_path, "r", encoding="utf8") as stream:
        try:
            zfile = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()
    if "zranges" not in zfile:
        error_text = "The yaml zone file must be in the format:\nzranges:\
        \n    - Zone1: [1, 5]\n    - Zone2: [6, 10]\n    - Zone3: [11, 14])"
        raise Exception(format_error(error_text))
    return zfile


def _check_config(config_: RootConfig) -> None:
    if config_.input.properties:
        error_text = "CO2 mass computation does not take a property as input"
        raise ValueError(format_error(error_text))
    if config_.co2_mass_settings is None:
        error_text = "CO2 mass computation needs co2_mass_settings as input"
        raise ValueError(format_error(error_text))
    if (
        not config_.computesettings.aggregate_map
        and not config_.computesettings.indicator_map
    ):
        error_text = (
            "As neither indicator_map nor aggregate_map were requested,"
            " no map is produced"
        )
        raise ValueError(format_error(error_text))
    if config_.computesettings.indicator_map:
        warning_text = (
            "\nWARNING: Indicator maps cannot be calculated for CO2 mass maps. "
            "Changing 'indicator_map' to 'no'."
        )
        logging.warning(format_warning(warning_text))
        config_.computesettings.indicator_map = False


def _init_timer():
    timer = Timer()
    timer.reset_timings()
    timer.code_parts = {
        "extract_source_data": "Extract source data",
        "calculate_co2": "Calculate CO2 mass per grid cell from source data",
        "read_xtgeo_grid": "Aggregate: Read grid using xtgeo",
        "extract_properties": "Aggregate: Extract properties from files",
        "aggregate_maps": "Aggregate: Aggregate 3D grid to 2D maps",
        "ndarray_to_regsurfs": "Aggregate: Convert results to xtgeo.RegularSurface",
        "write_surfaces": "Aggregate: Write maps to files",
        "logging": "Various logging",
    }


def main(arguments=None):
    """
    Takes input arguments and calculates co2 mass as a property and aggregates
    it to a 2D map at each time step, divided into different phases and locations.
    """
    if arguments is None:
        arguments = sys.argv[1:]
    _init_timer()
    timer = Timer()
    timer.start("total")

    config_ = _parser.process_arguments(arguments, map_type="co2_mass")
    config_.computesettings.aggregation = AggregationMethod.DISTRIBUTE
    config_.output.aggregation_tag = False
    _check_config(config_)
    log_input_configuration(config_, map_type="co2_mass")
    generate_co2_mass_maps(config_)

    timer.stop("total")
    timer.report()


if __name__ == "__main__":
    main()
