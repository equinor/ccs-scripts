#!/usr/bin/env python
import logging
import os
import shutil
import sys
import tempfile
from typing import List, Optional

import numpy as np
import xtgeo
from xtgeo.common import XTGeoDialog

from ccs_scripts.aggregate import (
    _config,
    _migration_time,
    _parser,
    grid3d_aggregate_map,
)
from ccs_scripts.aggregate._config import DEFAULT_LOWER_THRESHOLD, RootConfig
from ccs_scripts.aggregate._utils import log_input_configuration
from ccs_scripts.aggregate.grid3d_aggregate_map import _distribute_config_property
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import format_error, format_warning

_XTG = XTGeoDialog()

MIGRATION_TIME_PROPERTIES = [
    "AMFG",
    "AMFW",
    "AMFS",
    "YMFG",
    "YMFW",
    "YMFS",
    "XMF1",
    "XMF2",
    "XMFS",
    "ZMFS",  # NBNB: Not clear what it is
    "YMF1",
    "YMF2",
    "SGAS",
    "SWAT",
    "SOIL",
]


def _check_config(config_: RootConfig) -> None:
    config_.input.properties = _distribute_config_property(config_.input.properties)
    if config_.computesettings.indicator_map:
        warning_str = (
            "\nWARNING: Indicator maps cannot be calculated for migration time maps. "
            "Changing 'indicator_map' to 'no'."
        )
        logging.warning(format_warning(warning_str))
        config_.computesettings.indicator_map = False
    config_.computesettings.aggregation = _config.AggregationMethod.MIN
    config_.output.aggregation_tag = False
    config_.output.replace_masked_with_zero = False
    config_.computesettings.aggregate_map = True


def _check_threshold(
    lower_threshold: float,
    properties: List[xtgeo.GridProperty],
) -> float:
    min_value_props = min([p.values.min() for p in properties])
    max_value_props = max([p.values.max() for p in properties])
    if lower_threshold < 0:
        if min_value_props >= 0:
            warning_str = "\nWARNING: Specified lower threshold is negative, "
            warning_str += "but no property values are negative."
            warning_str += "\n         => Changing the lower threshold value:"
            warning_str += f"\n            - Specified value: {lower_threshold:>8}"
            lower_threshold = DEFAULT_LOWER_THRESHOLD
            warning_str += f"\n            - Changed to     : {lower_threshold:>8}"
            logging.warning(format_warning(warning_str))
    else:
        if lower_threshold > max_value_props:
            warning_str = "\nWARNING: Specified lower threshold is "
            warning_str += "higher than the maximum property value in the grid."
            warning_str += f"\n         - Specified value       : {lower_threshold:>8}"
            warning_str += (
                f"\n         - Maximum property value: {max_value_props:>8.4f}"
            )
            logging.warning(format_warning(warning_str))
    return lower_threshold


def _log_t_prop(t_prop: xtgeo.GridProperty, prop_name: str):
    col1 = 20
    col2 = 8

    n_finite = np.sum(np.isfinite(t_prop.values))
    logging.info(f"\nSummary of time migration 3D grid property {prop_name}:")
    logging.info(f"{'  - Minimum':<{col1}} : {t_prop.values.min():>{col2}.1f}")
    logging.info(f"{'  - Mean':<{col1}} : {t_prop.values.mean():>{col2}.1f}")
    logging.info(f"{'  - Maximum':<{col1}} : {t_prop.values.max():>{col2}.1f}")
    logging.info(
        f"{'  - # cells with CO2':<{col1}} : "
        f"{n_finite:>{col2}} ({100.0 * n_finite / t_prop.values.size:.1f}%)"
    )


def calculate_migration_time_property(
    properties_files: str,
    property_name: str,
    lower_threshold: float,
    grid_file: Optional[str],
    dates: List[str],
) -> xtgeo.GridProperty:
    """
    Calculates a 3D migration time property from the provided grid and grid property
    files
    """
    timer = Timer()
    logging.info("\nStart calculating time migration property in 3D grid")
    prop_spec = [_config.Property(source=properties_files, name=property_name)]
    timer.start("read_xtgeo_grid_migration_time")
    grid = None if grid_file is None else xtgeo.grid_from_file(grid_file)
    timer.stop("read_xtgeo_grid_migration_time")
    timer.start("extract_properties_migration_time")
    properties = _parser.extract_properties(
        prop_spec, grid, dates, mask_low_values=False
    )
    timer.stop("extract_properties_migration_time")
    lower_threshold = _check_threshold(lower_threshold, properties)
    grid3d_aggregate_map._log_properties_info(properties)

    timer.start("generate_migration_time_property")
    t_prop = _migration_time.generate_migration_time_property(
        properties, lower_threshold
    )
    timer.stop("generate_migration_time_property")
    _log_t_prop(t_prop, property_name)

    return t_prop


def migration_time_property_to_map(
    config_: RootConfig,
    prop: xtgeo.GridProperty,
    temp_path: str,
):
    """
    Aggregates and writes a migration time property to file using `grid3d_aggregate_map`
    The migration time property is written to a temporary file while performing the
    aggregation.
    """
    logging.info(
        "\nStart aggregating time migration property from "
        "temporary 3D grid file to 2D map"
    )
    config_.input.properties = [_config.Property(temp_path, None, 0)]
    prop.to_file(temp_path)
    grid3d_aggregate_map.generate_from_config(config_)


def _init_timer():
    timer = Timer()
    timer.reset_timings()
    timer.code_parts = {
        "read_xtgeo_grid_migration_time": "Read input grid using xtgeo",
        "extract_properties_migration_time": "Extract input properties",
        "generate_migration_time_property": "Generate migration time property",
        "read_xtgeo_grid": "Aggregate: Read grid using xtgeo",
        "extract_properties": "Aggregate: Extract properties from files",
        "aggregate_maps": "Aggregate: Aggregate 3D grid to 2D maps",
        "ndarray_to_regsurfs": "Aggregate: Convert results to xtgeo.RegularSurface",
        "write_surfaces": "Aggregate: Write maps to files",
        "logging": "Various logging",
    }


def main(arguments=None):
    """
    Calculates a migration time property and aggregates it to a 2D map
    """
    if arguments is None:
        arguments = sys.argv[1:]
    _init_timer()
    timer = Timer()
    timer.start("total")

    config_ = _parser.process_arguments(arguments, map_type="migration_time")
    _check_config(config_)
    log_input_configuration(config_, map_type="migration_time")
    p_spec = []
    if any(x.name in MIGRATION_TIME_PROPERTIES for x in config_.input.properties):
        removed_props = [
            x.name
            for x in config_.input.properties
            if x.name not in MIGRATION_TIME_PROPERTIES
        ]
        p_spec.extend(
            [x for x in config_.input.properties if x.name in MIGRATION_TIME_PROPERTIES]
        )
        if len(removed_props) > 0:
            warning_str = (
                "\nWARNING: Migration time maps are "
                "not supported for these properties: ",
                ", ".join(str(x) for x in removed_props),
            )
            logging.warning(format_warning(warning_str))
    else:
        error_text = (
            "Migration time maps are not supported for "
            "any of the properties provided: "
        )
        error_text += f"{', '.join([x.name for x in config_.input.properties])}"
        raise ValueError(format_error(error_text))
    config_.input.properties = p_spec
    temp_dir = tempfile.mkdtemp()
    logging.info(f"\nMaking temporary directory for 3D grids: {temp_dir}")
    try:
        for prop in config_.input.properties:
            t_prop = calculate_migration_time_property(
                prop.source,
                prop.name,
                prop.lower_threshold,
                config_.input.grid,
                config_.input.dates,
            )
            temp_path = os.path.join(temp_dir, prop.name)
            migration_time_property_to_map(config_, t_prop, temp_path)
    finally:
        logging.info(f"\nDeleting temporary directory for 3D grids: {temp_dir}")
        shutil.rmtree(temp_dir)

    timer.stop("total")
    timer.report()


if __name__ == "__main__":
    main()
