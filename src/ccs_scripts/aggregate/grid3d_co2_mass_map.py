#!/usr/bin/env python
import copy
import logging
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import yaml

from ccs_scripts.aggregate import _config, _parser, grid3d_aggregate_map
from ccs_scripts.aggregate._co2_mass import translate_co2data_to_property
from ccs_scripts.aggregate._config import AggregationMethod, RootConfig
from ccs_scripts.aggregate._utils import log_input_configuration
from ccs_scripts.co2_containment.co2_calculation import (
    RELEVANT_PROPERTIES,
    RegionInfo,
    ZoneInfo,
    _detect_eclipse_mole_fraction_props,
    calculate_co2,
    source_data_,
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

EXAMPLES = """
.. code-block:: console

  FORWARD_MODEL GRID3D_CO2_MASS_MAP(<CONFIG_CO2_MASS_MAP>=conf.yml, <ECLROOT>=<ECLBASE>)
"""


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
    all_dates = [x.date for x in co2_data.data_list]
    dates_idx = list(range(len(all_dates)))
    if len(dates) > 0:
        co2_data.data_list = [x for x in co2_data.data_list if x.date in dates]
        dates_idx = [i for i, val in enumerate(all_dates) if val in dates]
    grid_folder, delete_tmp_grid_folder = _process_grid_dir(config_.output.gridfolder)
    try:
        properties_to_extract = copy.deepcopy(RELEVANT_PROPERTIES)
        current_source_data = copy.deepcopy(source_data_)
        _, properties_to_extract = _detect_eclipse_mole_fraction_props(
            co2_mass_settings.unrst_source, properties_to_extract, current_source_data
        )
        out_property_list = translate_co2data_to_property(
            co2_data,
            grid_file,
            co2_mass_settings,
            grid_folder,
            properties_to_extract,
            dates_idx,
        )
        co2_mass_property_to_map(config_, out_property_list)
    finally:
        # Make sure temp directory is deleted even if exception is thrown above
        if delete_tmp_grid_folder:
            clean_tmp(grid_folder)


def _process_grid_dir(grid_folder: Optional[str]) -> Tuple[str, bool]:
    """
    Setting up the grid folder to store the gridproperties
    """
    if grid_folder is not None:
        if not os.path.exists(grid_folder):
            parent_dir = os.path.dirname(grid_folder)
            if os.path.exists(parent_dir):
                os.mkdir(grid_folder)
                logging.info(f"\nCreated new grid folder: {grid_folder}")
            else:
                error_txt = (
                    "\nERROR: Specified grid folder is invalid (no parent folder):"
                )
                error_txt += f"\n    Path            : {grid_folder}"
                if not os.path.isabs(grid_folder):
                    error_txt += (
                        f"\n    -> Absolute path: {os.path.abspath(grid_folder)}"
                    )
                error_txt += f"\n    Parent folder   : {parent_dir}"
                if not os.path.isabs(parent_dir):
                    error_txt += (
                        f"\n    -> Absolute path: {os.path.abspath(parent_dir)}"
                    )
                logging.error(error_txt)
                raise FileNotFoundError(error_txt)
        return grid_folder, False
    else:
        grid_folder = tempfile.mkdtemp()
        logging.info(f"\nMaking temporary directory for 3D grids: {grid_folder}")
        return grid_folder, True


def clean_tmp(grid_folder: str):
    """
    Removes the 3d grids produced if not specific output folder is provided

    Args:
        grid_folder: Path to directory of files for 3d GridProperties
    """
    logging.info(
        f'\nDeleting temp grid folder "{grid_folder}"'
        f" containing {len(os.listdir(grid_folder))} files"
    )
    for file_name in os.listdir(grid_folder):
        if file_name.endswith(".EGRID") or file_name.endswith(".UNRST"):
            os.remove(os.path.join(grid_folder, file_name))
    if len(os.listdir(grid_folder)) == 0:
        shutil.rmtree(grid_folder)


def co2_mass_property_to_map(
    config_: RootConfig,
    out_property_list: List[Optional[str]],
):
    """
    Aggregates with SUM and writes a list of CO2 mass property to files
    using `grid3d_aggregate_map`.

    Args:
        config_:           Arguments in the config file
        out_property_list: List with paths of the GridProperties objects
                           to be aggregated

    """
    config_.input.properties = []
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


def _check_config(config_: RootConfig) -> None:
    if config_.input.properties:
        raise ValueError("CO2 mass computation does not take a property as input")
    if config_.co2_mass_settings is None:
        raise ValueError("CO2 mass computation needs co2_mass_settings as input")
    if (
        not config_.computesettings.aggregate_map
        and not config_.computesettings.indicator_map
    ):
        error_text = (
            "As neither indicator_map nor aggregate_map were requested,"
            " no map is produced"
        )
        raise ValueError(error_text)
    if config_.computesettings.indicator_map:
        logging.warning(
            "\nWARNING: Indicator maps cannot be calculated for CO2 mass maps. "
            "Changing 'indicator_map' to 'no'."
        )
        config_.computesettings.indicator_map = False


def main(arguments=None):
    """
    Takes input arguments and calculates co2 mass as a property and aggregates
    it to a 2D map at each time step, divided into different phases and locations.
    """
    if arguments is None:
        arguments = sys.argv[1:]
    config_ = _parser.process_arguments(arguments)
    config_.computesettings.aggregation = AggregationMethod.DISTRIBUTE
    config_.output.aggregation_tag = False
    _check_config(config_)
    log_input_configuration(config_, calc_type="co2_mass")
    generate_co2_mass_maps(config_)


if __name__ == "__main__":
    main()
