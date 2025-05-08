#!/usr/bin/env python
import logging
import os
import sys
import tempfile
from typing import Dict, List, Optional, Union

import numpy as np
import xtgeo
from xtgeo.common import XTGeoDialog

from ccs_scripts.aggregate import (
    _config,
    _migration_time,
    _parser,
    grid3d_aggregate_map,
)
from ccs_scripts.aggregate._config import RootConfig
from ccs_scripts.aggregate._utils import log_input_configuration
from ccs_scripts.aggregate.grid3d_aggregate_map import _distribute_config_property

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

# Module variables for ERT hook implementation:
DESCRIPTION = "Generate migration time property maps."
CATEGORY = "modelling.reservoir"
EXAMPLES = """
.. code-block:: console

  FORWARD_MODEL GRID3D_MIGRATION_TIME(<CONFIG_MIGTIME>=conf.yml, <ECLROOT>=<ECLBASE>)
"""


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
    config_.computesettings.aggregate_map = True


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
    lower_threshold: Union[float, List],
    grid_file: Optional[str],
    dates: List[str],
) -> dict[str, xtgeo.GridProperty]:
    """
    Calculates a 3D migration time property from the provided grid and grid property
    files
    """
    logging.info("\nStart calculating time migration property in 3D grid")
    prop_spec = [_config.Property(source=properties_files, name=property_name)]
    grid = None if grid_file is None else xtgeo.grid_from_file(grid_file)
    properties = _parser.extract_properties(prop_spec, grid, dates)
    grid3d_aggregate_map._log_properties_info(properties)
    t_prop = _migration_time.generate_migration_time_property(
        properties, lower_threshold
    )
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
        config_.input.properties = [_config.Property(temp_path, None, None)]
        prop.to_file(temp_path)
    grid3d_aggregate_map.generate_from_config(config_)
    os.unlink(temp_path)


def main(arguments=None):
    """
    Calculates a migration time property and aggregates it to a 2D map
    """
    if arguments is None:
        arguments = sys.argv[1:]
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


if __name__ == "__main__":
    main()
