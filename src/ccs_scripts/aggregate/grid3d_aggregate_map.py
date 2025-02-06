#!/usr/bin/env python
from datetime import datetime
import getpass
import logging
import os
import pathlib
import platform
import socket
import subprocess
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import xtgeo
from xtgeo.common import XTGeoDialog

from ccs_scripts.aggregate._co2_mass import MapName
from ccs_scripts.aggregate._config import (
    AggregationMethod,
    ComputeSettings,
    Input,
    MapSettings,
    Output,
    Zonation,
)
from ccs_scripts.aggregate._parser import (
    create_map_template,
    extract_properties,
    extract_zonations,
    process_arguments,
)

from . import _config, _grid_aggregation

_XTG = XTGeoDialog()


# Module variables for ERT hook implementation:
DESCRIPTION = (
    "Aggregate property maps from 3D grids. Docs:\n"
    + "https://fmu-docs.equinor.com/docs/xtgeoapp-grd3dmaps/"
)
CATEGORY = "modelling.reservoir"
EXAMPLES = """
.. code-block:: console

  FORWARD_MODEL GRID3D_AGGREGATE_MAP(<CONFIG_AGGREGATE>=conf.yml, <ECLROOT>=<ECLBASE>)
"""


def write_map(x_nodes, y_nodes, map_, filename):
    """
    Writes a 2D map to file as an xtgeo.RegularSurface. Returns the xtgeo.RegularSurface
    instance.
    """
    dx = x_nodes[1] - x_nodes[0]
    dy = y_nodes[1] - y_nodes[0]
    masked_map = np.ma.array(map_)
    masked_map.mask = np.isnan(map_)
    surface = xtgeo.RegularSurface(
        ncol=x_nodes.size,
        nrow=y_nodes.size,
        xinc=dx,
        yinc=dy,
        xori=x_nodes[0],
        yori=y_nodes[0],
        values=masked_map,
    )
    surface.to_file(filename)
    return surface


def write_plot_using_plotly(surf: xtgeo.RegularSurface, filename):
    """
    Writes a 2D map to an html using the plotly library
    """
    # pylint: disable=import-outside-toplevel
    import plotly.express as px

    x_nodes = surf.xori + np.arange(0, surf.ncol) * surf.xinc
    y_nodes = surf.yori + np.arange(0, surf.nrow) * surf.yinc
    px.imshow(
        surf.values.filled(np.nan).T, x=x_nodes, y=y_nodes, origin="lower"
    ).write_html(filename.with_suffix(".html"), include_plotlyjs="cdn")


def write_plot_using_quickplot(surface, filename):
    """
    Writes a 2D map using quickplot from xtgeoviz
    """
    # pylint: disable=import-outside-toplevel
    from xtgeoviz import quickplot

    quickplot(surface, filename=filename.with_suffix(".png"))


def modify_mass_property_names(properties: List[xtgeo.GridProperty]):
    if any("MASS" in p.name for p in properties):
        for p in properties:
            if "MASS" in p.name:
                mass_prop_name = p.name.split("--")[0]
                mass_prop_date = p.name.split("--")[1]
                p.name = f"{MapName[mass_prop_name].value}--{mass_prop_date}"


def _log_grid_info(grid: xtgeo.Grid) -> None:
    col1 = 25
    logging.info("\nGrid read from file:")
    logging.info(f"{'  - Name':<{col1}} : {grid.name if grid.name is not None else '-'}")
    logging.info(f"{'  - Number of columns (x)':<{col1}} : {grid.ncol}")
    logging.info(f"{'  - Number of rows (y)':<{col1}} : {grid.nrow}")
    logging.info(f"{'  - Number of layers':<{col1}} : {grid.nlay}")
    logging.info(f"{'  - Units':<{col1}} : {grid.units.name.lower() if grid.units is not None else '?'}")


def _log_properties_info(properties: List[xtgeo.GridProperty]) -> None:
    logging.info("\nProperties read from file:")
    logging.info(f"\n{'Name':<21} {'Date':<10} {'Mean':<7} {'Max':<7}")
    logging.info("-"*48)
    for p in properties:
        name_stripped = p.name.split("--")[0] if "--" in p.name else p.name
        logging.info(f"{name_stripped:<21} {p.date if p.date is not None else '-':<10} {p.values.mean():<7.3f} {p.values.max():<7.3f}")

def generate_maps(
    input_: Input,
    zonation: Zonation,
    computesettings: ComputeSettings,
    map_settings: MapSettings,
    output: Output,
):
    """
    Calculate and write aggregated property maps to file
    """
    logging.info("\nReading grid, properties and zone(s)")
    grid = xtgeo.grid_from_file(input_.grid)
    _log_grid_info(grid)
    properties = extract_properties(input_.properties, grid, input_.dates)
    _log_properties_info(properties)
    modify_mass_property_names(properties)
    _filters: List[Tuple[str, Optional[Union[np.ndarray, None]]]] = []
    if computesettings.all:
        _filters.append(("all", None))
    if computesettings.zone:
        _filters += extract_zonations(zonation, grid)

    logging.info(f"\nGenerating property maps for: {', '.join([f[0] for f in _filters])}")
    xn, yn, p_maps = _grid_aggregation.aggregate_maps(
        create_map_template(map_settings),
        grid,
        properties,
        [f[1] for f in _filters],
        computesettings.aggregation,
        computesettings.weight_by_dz,
    )
    logging.info(f"\nDone calculating properties")
    prop_tags = [
        _property_tag(p.name, computesettings.aggregation, output.aggregation_tag)
        for p in properties
    ]
    if computesettings.aggregate_map:
        surfs = _ndarray_to_regsurfs(
            [f[0] for f in _filters],
            prop_tags,
            xn,
            yn,
            p_maps,
            output.lowercase,
        )
        _write_surfaces(surfs, output.mapfolder, output.plotfolder, output.use_plotly)
        logging.info(f"\nDone exporting the following {len(prop_tags)} aggregate maps:\n{', '.join(prop_tags)}")
    if computesettings.indicator_map:
        prop_tags_indicator = [p.replace("max", "indicator") for p in prop_tags]
        p_maps_indicator = [
            [np.where(np.isfinite(p), 1, p) for p in map_] for map_ in p_maps
        ]
        surfs_indicator = _ndarray_to_regsurfs(
            [f[0] for f in _filters],
            prop_tags_indicator,
            xn,
            yn,
            p_maps_indicator,
            output.lowercase,
        )
        _write_surfaces(
            surfs_indicator, output.mapfolder, output.plotfolder, output.use_plotly
        )
        logging.info(f"\nDone exporting the following {len(prop_tags_indicator)} indicator maps:\n{', '.join(prop_tags_indicator)}")
    if not computesettings.aggregate_map and not computesettings.indicator_map:
        error_text = (
            "As neither indicator_map nor aggregate_map were requested,"
            " no map is produced"
        )
        raise Exception(error_text)


def _property_tag(prop: str, agg_method: AggregationMethod, agg_tag: bool):
    agg = f"{agg_method.value}_" if agg_tag else ""
    return f"{agg}{prop}"


# pylint: disable=too-many-arguments
def _ndarray_to_regsurfs(
    filter_names: List[str],
    prop_names: List[str],
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    maps: List[List[np.ndarray]],
    lowercase: bool,
) -> List[xtgeo.RegularSurface]:
    return [
        xtgeo.RegularSurface(
            ncol=x_nodes.size,
            nrow=y_nodes.size,
            xinc=x_nodes[1] - x_nodes[0],
            yinc=y_nodes[1] - y_nodes[0],
            xori=x_nodes[0],
            yori=y_nodes[0],
            values=np.ma.array(map_, mask=np.isnan(map_)),
            name=_deduce_surface_name(fn, prop, lowercase),
        )
        for fn, inner in zip(filter_names, maps)
        for prop, map_ in zip(prop_names, inner)
    ]


def _deduce_surface_name(filter_name, property_name, lowercase):
    name = f"{filter_name}--{property_name}"
    if lowercase:
        name = name.lower()
    return name


def _write_surfaces(
    surfaces: List[xtgeo.RegularSurface],
    map_folder: str,
    plot_folder: Optional[str],
    use_plotly: bool,
):
    for surface in surfaces:
        surface.to_file((pathlib.Path(map_folder) / surface.name).with_suffix(".gri"))
        if plot_folder:
            pn = pathlib.Path(plot_folder) / surface.name
            if use_plotly:
                write_plot_using_plotly(surface, pn)
            else:
                write_plot_using_quickplot(surface, pn)


def generate_from_config(config: _config.RootConfig):
    """
    Wrapper around `generate_maps` based on a configuration object (RootConfig)
    """
    generate_maps(
        config.input,
        config.zonation,
        config.computesettings,
        config.mapsettings,
        config.output,
    )


def _log_input_configuration(config_: _config.RootConfig, calc_type: str = "aggregate") -> None:
    """
    Log the provided input
    """
    version = "v0.9.0"
    is_dev_version = True
    if is_dev_version:
        version += "_dev"
        try:
            source_dir = os.path.dirname(os.path.abspath(__file__))
            short_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], cwd=source_dir
                )
                .decode("ascii")
                .strip()
            )
        except subprocess.CalledProcessError:
            short_hash = "-"
        version += " (latest git commit: " + short_hash + ")"

    col1 = 37
    now = datetime.now()
    date_time = now.strftime("%B %d, %Y %H:%M:%S")
    if calc_type == "aggregate":
        logging.info("CCS-scripts - Aggregate maps")
        logging.info("============================")
    elif calc_type == "time_migration":
        logging.info("CCS-scripts - Time migration maps")
        logging.info("=================================")
    elif calc_type == "co2_mass":
        logging.info("CCS-scripts - CO2 mass maps")
        logging.info("===========================")
    logging.info(f"{'Version':<{col1}} : {version}")
    logging.info(f"{'Date and time':<{col1}} : {date_time}")
    logging.info(f"{'User':<{col1}} : {getpass.getuser()}")
    logging.info(f"{'Host':<{col1}} : {socket.gethostname()}")
    logging.info(f"{'Platform':<{col1}} : {platform.system()} ({platform.release()})")
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    logging.info(f"{'Python version':<{col1}} : {py_version}")

    logging.info("\nInput configuration:")
    logging.info(f"{'  Grid file':<{col1}} : {config_.input.grid}")
    if calc_type != "co2_mass":
        logging.info("  Properties:")
        if config_.input.properties is None:
            logging.info("    No properties specified")
        else:
            for p in config_.input.properties:
                logging.info(f"{'    - Name':<{col1}} : {p.name}")
                logging.info(f"{'      Source':<{col1}} : {p.source if p.source is not None else '-'}")
                logging.info(f"{'      Lower threshold':<{col1}} : {p.lower_threshold if p.lower_threshold is not None else '-'}")
    if len(config_.input.dates) > 0:
        logging.info(f"{'  Dates':<{col1}} : {', '.join(config_.input.dates)}")
    else:
        logging.info(f"{'  Dates':<{col1}} : - (not specified => using all dates)")

    if calc_type == "time_migration":
        return  # Everything else in the config is irrelevant for time migration, or will be overwritten later

    logging.info("\nOutput configuration:")
    logging.info(f"{'  Map folder':<{col1}} : {config_.output.mapfolder}")
    logging.info(f"{'  Plot folder':<{col1}} : {config_.output.plotfolder if config_.output.plotfolder is not None else '- (plot export not selected)'}")
    logging.info(f"{'  Grid folder':<{col1}} : {config_.output.gridfolder if config_.output.gridfolder is not None else '- (export of 3D grids not selected)'}")
    logging.info(f"{'  Use lower case in file names':<{col1}} : {'yes' if config_.output.lowercase else 'no'}")
    logging.info(f"{'  Module/method for 2D plots':<{col1}} : {'plotly library' if config_.output.use_plotly else 'quickplot from xtgeoviz'}")
    logging.info(f"{'  Aggregation tag':<{col1}} : {config_.output.aggregation_tag}")  # NBNB-AS: Remove this from logging?

    logging.info("\nZonation configuration:")
    logging.info("  Z-property:")
    if config_.zonation.zproperty is None:
        logging.info("    No z-property specified")
    else:
        logging.info(f"{'    Source':<{col1}} : {config_.zonation.zproperty.source}")
        logging.info(f"{'    Name':<{col1}} : {config_.zonation.zproperty.name if config_.zonation.zproperty.name is not None else '-'}")
        logging.info("    Zones:")
        zones = config_.zonation.zproperty.zones
        if len(zones) == 0:
            logging.info("      No zones specified")
        else:
            for z in zones:
                for i, (k, v) in enumerate(z.items()):
                    if i == 0:
                        logging.info(f"{f'      - {k}':<{col1}} : {v}")
                    else:
                        logging.info(f"{f'        {k}':<{col1}} : {v}")
    logging.info("  Z-ranges:")
    if len(config_.zonation.zranges) == 0:
        logging.info("    No z-ranges specified")
    else:
        for z in config_.zonation.zranges:
            for i, (k, v) in enumerate(z.items()):
                if i == 0:
                    logging.info(f"{f'    - {k}':<{col1}} : {v}")
                else:
                    logging.info(f"{f'      {k}':<{col1}} : {v}")

    logging.info("\nComputation configuration:")
    logging.info(f"{'  Aggregation method':<{col1}} : {config_.computesettings.aggregation.name}")
    logging.info(f"{'  Weight by dz':<{col1}} : {config_.computesettings.weight_by_dz}")
    logging.info(f"{'  Make maps for full grid (all zones)':<{col1}} : {config_.computesettings.all}")
    logging.info(f"{'  Make maps per zone':<{col1}} : {config_.computesettings.zone}")
    logging.info(f"{'  Calculate aggregate maps':<{col1}} : {config_.computesettings.aggregate_map}")
    logging.info(f"{'  Calculate indicator maps':<{col1}} : {config_.computesettings.indicator_map}")

    logging.info("\nMap configuration:")
    ms = config_.mapsettings
    logging.info(f"{'  Origo x':<{col1}} : {ms.xori if ms.xori is not None else '-'}")
    logging.info(f"{'  Origo y':<{col1}} : {ms.yori if ms.yori is not None else '-'}")
    logging.info(f"{'  Increment x':<{col1}} : {ms.xinc if ms.xinc is not None else '-'}")
    logging.info(f"{'  Increment y':<{col1}} : {ms.yinc if ms.yinc is not None else '-'}")
    logging.info(f"{'  Number of columns (x)':<{col1}} : {ms.ncol if ms.ncol is not None else '-'}")
    logging.info(f"{'  Number of rows (y)':<{col1}} : {ms.nrow if ms.nrow is not None else '-'}")
    if ms.xinc is not None and ms.ncol is not None:
        logging.info(f"{'  => Size x-direction':<{col1}} : {ms.xinc * ms.ncol}")
    if ms.yinc is not None and ms.nrow is not None:
        logging.info(f"{'  => Size y-direction':<{col1}} : {ms.yinc * ms.nrow}")
    logging.info(f"{'  Template file':<{col1}} : {ms.templatefile if ms.templatefile is not None else '- (not specified)'}")
    logging.info(f"{'  Pixel-to-cell-size ratio':<{col1}} : {ms.pixel_to_cell_ratio}")  # NBNB-AS: Only used if ...

    config_.co2_mass_settings.
    if calc_type == "co2_mass":
        config_.co
        pass
        # config_.co2_mass_settings
    # unrst_source: str
    # init_source: str
    # maps: Optional[List[str]] = None
    # residual_trapping: Optional[bool] = False
    exit()

def _distribute_config_property(config_: _config.RootConfig):
    if config_.input.properties is None:
        return
    if not isinstance(config_.input.properties[0].name, list):
        return
    tmp_props = config_.input.properties.pop()
    if isinstance(tmp_props.lower_threshold, list) and len(tmp_props.name) == len(
        tmp_props.lower_threshold
    ):
        config_.input.properties.extend(
            [
                _config.Property(tmp_props.source, name, threshold)
                for name, threshold in zip(tmp_props.name, tmp_props.lower_threshold)
            ]
        )
    elif isinstance(tmp_props.lower_threshold, float) or (
        isinstance(tmp_props.lower_threshold, list)
        and len(tmp_props.lower_threshold) == 1
    ):
        logging.info(
            f"Only one value of threshold for {str(len(tmp_props.name))}."
            f"properties. The same threshold will be assumed for all the"
            f"properties."
        )
        if (
            isinstance(tmp_props.lower_threshold, list)
            and len(tmp_props.lower_threshold) == 1
        ):
            tmp_props.lower_threshold = tmp_props.lower_threshold * len(tmp_props.name)
        else:
            tmp_props.lower_threshold = [tmp_props.lower_threshold] * len(
                tmp_props.name
            )
        config_.input.properties.extend(
            [
                _config.Property(tmp_props.source, name, threshold)
                for name, threshold in zip(tmp_props.name, tmp_props.lower_threshold)
            ]
        )
    else:
        error_text = (
            f"{str(len(tmp_props.lower_threshold))} values of co2_threshold"
            f"provided, but {str(len(tmp_props.name))} properties in config"
            f" file input. Fix the amount of values in co2_threshold or"
            f"the amount of properties in config file"
        )
        raise Exception(error_text)
    return


def main(arguments=None):
    """
    Main function that wraps `generate_from_config` with argument parsing
    """
    if arguments is None:
        arguments = sys.argv[1:]
    config_ = process_arguments(arguments)
    _distribute_config_property(config_)
    _log_input_configuration(config_, calc_type="aggregate")
    generate_from_config(config_)


if __name__ == "__main__":
    main()
