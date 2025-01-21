#!/usr/bin/env python
import os
import shutil
import sys
from typing import Dict, List, Optional, Union

import yaml

from ccs_scripts.aggregate import _config, _parser, grid3d_aggregate_map
from ccs_scripts.aggregate._co2_mass import translate_co2data_to_property
from ccs_scripts.aggregate._config import AggregationMethod
from ccs_scripts.co2_containment.co2_calculation import (
    RELEVANT_PROPERTIES,
    RegionInfo,
    ZoneInfo,
    calculate_co2,
)

# Module variables for ERT hook implementation:
DESCRIPTION = """
    Produces maps of CO2 mass per date, formation and phase (gas/dissolved).
    Outputs are .gri files (one per requested combination of date, phase,
    formation).

    A yaml config file is the input file to co2_mass_maps. Through this file
    the user can decide on which dates, phases or formations the maps are
    produced. See tests/yaml for examples of yaml files.
    """

CATEGORY = "modelling.reservoir"

# EXAMPLES = """
# .. code-block:: console
#
#   FORWARD_MODEL GRID3D_MIGRATION_TIME(<CONFIG_MIGTIME>=conf.yml, <ECLROOT>=<ECLBASE>)
# """


def generate_co2_mass_maps(config_):
    """
    Calculates and exports 2D and 3D CO2 mass properties from the provided config file

    Args:
        config_: Arguments in the config file
    """
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
    co2_data = calculate_co2(
        grid_file=grid_file,
        unrst_file=co2_mass_settings.unrst_source,
        calc_type_input="mass",
        init_file=co2_mass_settings.init_source,
        zone_info=zone_info,
        region_info=region_info,
        residual_trapping=co2_mass_settings.residual_trapping,
    )
    dates = config_.input.dates
    if len(dates) > 0:
        co2_data.data_list = [x for x in co2_data.data_list if x.date in dates]
    out_property_list = translate_co2data_to_property(
        co2_data,
        grid_file,
        co2_mass_settings,
        RELEVANT_PROPERTIES,
        config_.output.gridfolder,
    )
    co2_mass_property_to_map(config_, out_property_list)


def clean_tmp(out_property_list: List[Union[str, None]]):
    """
    Removes the 3d grids produced if not specific output folder is provided

    Args:
        out_property_list: List with paths of the 3d GridProperties
    """
    for props in out_property_list:
        if isinstance(props, str):
            directory_path = os.path.dirname(props[0])
            os.remove(props)
            if os.path.isdir(directory_path) and not os.listdir(directory_path):
                shutil.rmtree(directory_path)


def co2_mass_property_to_map(
    config_: _config.RootConfig,
    out_property_list: List[Optional[str]],
):
    """
    Aggregates with SUM and writes a list of CO2 mass property to files
    using `grid3d_aggregate_maps`.

    Args:
        config_:           Arguments in the config file
        out_property_list: List with paths of the GridProperties objects
                           to be aggregated

    """
    config_.input.properties = []
    config_.computesettings.aggregation = AggregationMethod.DISTRIBUTE
    config_.output.aggregation_tag = False
    for props in out_property_list:
        if isinstance(props, str):
            config_.input.properties.append(
                _config.Property(
                    props,
                    None,
                    None,
                )
            )
    grid3d_aggregate_map.generate_from_config(config_)
    if not config_.output.gridfolder:
        clean_tmp(out_property_list)


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
        raise Exception(error_text)
    return zfile


def main(arguments=None):
    """
    Takes input arguments and calculates co2 mass as a property and aggregates
    it to a 2D map at each time step, divided into different phases and locations.
    """
    if arguments is None:
        arguments = sys.argv[1:]
    config_ = _parser.process_arguments(arguments)
    if config_.input.properties:
        raise ValueError("CO2 mass computation does not take a property as input")
    if config_.co2_mass_settings is None:
        raise ValueError("CO2 mass computation needs co2_mass_settings as input")
    generate_co2_mass_maps(config_)


if __name__ == "__main__":
    main()
