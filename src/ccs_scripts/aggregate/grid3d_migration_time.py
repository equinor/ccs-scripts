#!/usr/bin/env python
import logging
import os
import sys
import tempfile
from typing import Dict, List, Optional

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
from ccs_scripts.utils.utils import Timer

_XTG = XTGeoDialog()

MIGRATION_TIME_PROPERTIES = [
    "AMFG",
    "AMFW",
    "YMFG",
    "YMFW",
    "XMF1",
    "XMF2",
    "YMF1",
    "YMF2",
    "SGAS",
    "SWAT",
]


def _check_config(config_: RootConfig) -> None:
    config_.input.properties = _distribute_config_property(config_.input.properties)
    if config_.computesettings.indicator_map:
        logging.warning(
            "\nWARNING: Indicator maps cannot be calculated for CO2 mass maps. "
            "Changing 'indicator_map' to 'no'."
        )
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
            logging.warning(warning_str)
    else:
        if lower_threshold > max_value_props:
            warning_str = "\nWARNING: Specified lower threshold is "
            warning_str += "higher than the maximum property value in the grid."
            warning_str += f"\n         - Specified value       : {lower_threshold:>8}"
            warning_str += (
                f"\n         - Maximum property value: {max_value_props:>8.4f}"
            )
            logging.warning(warning_str)
    return lower_threshold


def _log_t_prop(t_prop: dict[str, xtgeo.GridProperty]):
    col1 = 20
    col2 = 8
    for k, v in t_prop.items():
        n_finite = np.sum(np.isfinite(v.values))
        logging.info(f"\nSummary of time migration 3D grid property {k}:")
        logging.info(f"{'  - Minimum':<{col1}} : {v.values.min():>{col2}.1f}")
        logging.info(f"{'  - Mean':<{col1}} : {v.values.mean():>{col2}.1f}")
        logging.info(f"{'  - Maximum':<{col1}} : {v.values.max():>{col2}.1f}")
        logging.info(
            f"{'  - # cells with CO2':<{col1}} : "
            f"{n_finite:>{col2}} ({100.0 * n_finite / v.values.size:.1f}%)"
        )


def calculate_migration_time_property(
    properties_files: str,
    property_name: str,
    lower_threshold: float,
    grid_file: Optional[str],
    dates: List[str],
) -> dict[str, xtgeo.GridProperty]:
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
    _log_t_prop(t_prop)

    return t_prop


def migration_time_property_to_map(
    config_: RootConfig,
    t_prop: Dict[str, xtgeo.GridProperty],
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
    for prop in t_prop.values():
        temp_file, temp_path = tempfile.mkstemp()
        os.close(temp_file)
        config_.input.properties = [
            _config.Property(temp_path, None)
        ]  # NBNB-AS: Input threshold?
        prop.to_file(temp_path)
    grid3d_aggregate_map.generate_from_config(config_)
    os.unlink(temp_path)


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

    config_ = _parser.process_arguments(arguments)
    _check_config(config_)
    log_input_configuration(config_, calc_type="time_migration")
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
            logging.warning(
                "\nWARNING: Time migration maps are "
                "not supported for these properties: ",
                ", ".join(str(x) for x in removed_props),
            )
    else:
        error_text = (
            "Time migration maps are not supported for "
            "any of the properties provided: "
        )
        error_text += f"{', '.join(p_spec.name)}"
        raise ValueError(error_text)
    config_.input.properties = p_spec
    for prop in config_.input.properties:
        t_prop = calculate_migration_time_property(
            prop.source,
            prop.name,
            prop.lower_threshold,
            config_.input.grid,
            config_.input.dates,
        )
        migration_time_property_to_map(config_, t_prop)

    timer.stop("total")
    timer.report()


if __name__ == "__main__":
    main()
