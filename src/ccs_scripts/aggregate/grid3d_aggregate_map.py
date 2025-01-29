#!/usr/bin/env python
import logging
import pathlib
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
    _XTG.say("Reading grid, properties and zone(s)")
    grid = xtgeo.grid_from_file(input_.grid)
    properties = extract_properties(input_.properties, grid, input_.dates)
    modify_mass_property_names(properties)
    _filters: List[Tuple[str, Optional[Union[np.ndarray, None]]]] = []
    if computesettings.all:
        _filters.append(("all", None))
    if computesettings.zone:
        _filters += extract_zonations(zonation, grid)
    _XTG.say("Generating Property Maps")
    xn, yn, p_maps = _grid_aggregation.aggregate_maps(
        create_map_template(map_settings),
        grid,
        properties,
        [f[1] for f in _filters],
        computesettings.aggregation,
        computesettings.weight_by_dz,
    )
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
    if computesettings.indicator_map:
        prop_tags_indicator = [p.replace("max", "indicator") for p in prop_tags]
        p_maps_indicator = [
            [np.where(np.isfinite(p), 1, p) for p in map] for map in p_maps
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
    print("Running grid3d_aggregate_map using code from ccs-scripts")
    logging.info("Running grid3d_aggregate_map using code from ccs-scripts")
    _XTG.say("Running grid3d_aggregate_map using code from ccs-scripts")
    if arguments is None:
        arguments = sys.argv[1:]
    config_ = process_arguments(arguments)
    _distribute_config_property(config_)
    generate_from_config(config_)


if __name__ == "__main__":
    main()
