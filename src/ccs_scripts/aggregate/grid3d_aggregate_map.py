#!/usr/bin/env python
import logging
import os
import pathlib
import sys
import warnings
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
from ccs_scripts.aggregate._utils import log_input_configuration
from ccs_scripts.utils.utils import Timer

from . import _config, _grid_aggregation

_XTG = XTGeoDialog()


# Module variables for ERT hook implementation:
DESCRIPTION = "Aggregate property maps from 3D grids."
CATEGORY = "modelling.reservoir"
EXAMPLES = """
.. code-block:: console

  FORWARD_MODEL GRID3D_AGGREGATE_MAP(<CONFIG_AGGREGATE>=conf.yml, <ECLROOT>=<ECLBASE>)
"""


def write_plot_using_plotly(surf: xtgeo.RegularSurface, filename: pathlib.Path):
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


def write_plot_using_quickplot(surf: xtgeo.RegularSurface, filename: pathlib.Path):
    """
    Writes a 2D map using quickplot from xtgeoviz
    """
    # pylint: disable=import-outside-toplevel
    from xtgeoviz import quickplot

    if surf.values.count() >= 5:
        quickplot(surf, filename=filename.with_suffix(".png"))


def _check_input(computesettings: ComputeSettings) -> None:
    if not computesettings.aggregate_map and not computesettings.indicator_map:
        error_text = (
            "As neither indicator_map nor aggregate_map were requested,"
            " no map is produced"
        )
        raise Exception(error_text)


def modify_mass_property_names(properties: List[xtgeo.GridProperty]):
    if any("MASS" in p.name for p in properties):
        for p in properties:
            if "MASS" in p.name:
                mass_prop_name = p.name.split("--")[0]
                mass_prop_date = p.name.split("--")[1]
                p.name = f"{MapName[mass_prop_name].value}--{mass_prop_date}"


def _log_grid_info(grid: xtgeo.Grid) -> None:
    timer = Timer()
    timer.start("logging")
    col1 = 25
    logging.info("\nGrid read from file:")
    logging.info(
        f"{'  - Name':<{col1}} : {grid.name if grid.name is not None else '-'}"
    )
    logging.info(f"{'  - Number of columns (x)':<{col1}} : {grid.ncol}")
    logging.info(f"{'  - Number of rows (y)':<{col1}} : {grid.nrow}")
    logging.info(f"{'  - Number of layers':<{col1}} : {grid.nlay}")
    logging.info(
        f"{'  - Units':<{col1}} : "
        f"{grid.units.name.lower() if grid.units is not None else '?'}"
    )
    timer.stop("logging")


def _log_properties_info(properties: List[xtgeo.GridProperty]) -> None:
    timer = Timer()
    timer.start("logging")
    logging.info("\nProperties read from file:")  # NBNB-AS: Not always from file
    logging.info(
        f"\n{'Name':<21} {'Date':>10} {'Mean':>10} {'Max':>10} "
        f"{'n_values':>10} {'n_masked':>10}"
    )
    logging.info("-" * 76)
    for p in properties:
        n_values = p.values.count()
        name_stripped = p.name.split("--")[0] if "--" in p.name else p.name
        mean_val = f"{p.values.mean():.3f}" if n_values > 0 else "-"
        max_val = f"{p.values.max():.3f}" if n_values > 0 else "-"
        logging.info(
            f"{name_stripped:<21} "
            f"{p.date if p.date is not None else '-':>10} "
            f"{mean_val:>10} "
            f"{max_val:>10} "
            f"{n_values:>10} "
            f"{np.ma.count_masked(p.values):>10}"
        )
    timer.stop("logging")


def _log_surfaces_exported(
    surfs: List[xtgeo.RegularSurface], zone_names: List[str], map_type: str
) -> None:
    timer = Timer()
    timer.start("logging")
    categories = [s.name.split("--") for s in surfs]
    types = set([v[1] for v in categories])
    logging.info(f"\nDone exporting {len(surfs)} {map_type} maps")
    logging.info(f"  - {len(types):>2} types: {', '.join(types)}")
    logging.info(f"  - {len(zone_names):>2} zones: {', '.join(zone_names)}")
    if len(categories[0]) == 3:  # No date for time migration maps
        dates = list(set([v[2] for v in categories]))
        dates.sort()
        logging.info(f"  - {len(dates):>2} dates: {', '.join(dates)}")
    timer.stop("logging")


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
    timer = Timer()
    _check_input(computesettings)
    logging.info("\nReading grid, properties and zone(s)")
    timer.start("read_xtgeo_grid")
    grid = xtgeo.grid_from_file(input_.grid)
    timer.stop("read_xtgeo_grid")
    _log_grid_info(grid)

    timer.start("extract_properties")
    properties = extract_properties(input_.properties, grid, input_.dates)
    timer.stop("extract_properties")
    _log_properties_info(properties)

    modify_mass_property_names(properties)
    _filters: List[Tuple[str, Optional[Union[np.ndarray, None]]]] = []
    if computesettings.all:
        _filters.append(("all", None))
    if computesettings.zone:
        _filters += extract_zonations(zonation, grid)
        logging.info("\nNumber of grid cells for each zone")
        for filt in _filters:
            if filt[0] == "all" or filt[1] is None:
                continue
            logging.info(
                f"{filt[0]:<14}: {np.count_nonzero(filt[1]):>9} "
                f"({100.0 * np.count_nonzero(filt[1]) / len(filt[1]):.1f} %)"
            )

    logging.info(
        f"\nGenerating property maps for: {', '.join([f[0] for f in _filters])}"
    )
    xn, yn, p_maps = _grid_aggregation.aggregate_maps(
        create_map_template(map_settings),
        grid,
        properties,
        [f[1] for f in _filters],
        computesettings.aggregation,
        computesettings.weight_by_dz,
    )
    logging.info("\nDone calculating properties")
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
        _write_surfaces(
            surfs,
            output.mapfolder,
            output.plotfolder,
            output.use_plotly,
            output.replace_masked_with_zero,
        )
        _log_surfaces(surfs)
        _log_surfaces_exported(surfs, [f[0] for f in _filters], "aggregate")
    if computesettings.indicator_map:
        prop_tags_indicator = [p.replace("max", "indicator") for p in prop_tags]
        p_maps_indicator = [
            [np.where(np.isfinite(p), 1, 0) for p in map_] for map_ in p_maps
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
            surfs_indicator,
            output.mapfolder,
            output.plotfolder,
            output.use_plotly,
            output.replace_masked_with_zero,
        )
        _log_surfaces_exported(surfs_indicator, [f[0] for f in _filters], "indicator")


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
    timer = Timer()
    timer.start("ndarray_to_regsurfs")
    out = [
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
    timer.stop("ndarray_to_regsurfs")
    return out


def _deduce_surface_name(filter_name, property_name, lowercase):
    name = f"{filter_name}--{property_name}"
    if lowercase:
        name = name.lower()
    return name


def _log_surfaces(surfaces: List[xtgeo.RegularSurface]):
    logging.info("\nSummary of calculated 2D maps:")
    logging.info(
        f"\n{'Name':<40} {'Mean':>10} {'Max':>10} "
        f"{'n_values':>10} {'n_pos':>10} {'n_masked':>10}"
    )
    logging.info("-" * 95)
    for s in surfaces:
        n_values = s.values.count()
        n_pos = np.sum(s.values > 1e-10) if n_values != 0 else 0
        mean_val = f"{s.values.mean():.3f}" if n_values > 0 else "-"
        max_val = f"{s.values.max():.3f}" if n_values > 0 else "-"
        txt = f"{s.name:<40} {mean_val:>10} {max_val:>10} "
        txt += f"{n_values:>10} {n_pos:>10} {np.ma.count_masked(s.values):>10}"
        if "all" in s.name:
            logging.info(txt)
        else:
            logging.debug(txt)


def _write_surfaces(
    surfaces: List[xtgeo.RegularSurface],
    map_folder: str,
    plot_folder: Optional[str],
    use_plotly: bool,
    replace_masked_with_zero: bool = True,
):
    timer = Timer()
    timer.start("write_surfaces")
    logging.info("\nWriting to map folder")
    logging.info(f"     Path         : {map_folder}")
    if not os.path.isabs(map_folder):
        logging.info(f"     Absolute path: {os.path.abspath(map_folder)}")
    # Note: Error handling of invalid map folder happens earlier

    if plot_folder:
        logging.info("\nWriting to plot folder")
        logging.info(f"     Path         : {plot_folder}")
        if not os.path.isabs(plot_folder):
            logging.info(f"     Absolute path: {os.path.abspath(plot_folder)}")
        if not os.path.exists(plot_folder):
            logging.warning("WARNING: Specified plot folder does not exist")

    for surface in surfaces:
        if replace_masked_with_zero:
            surface.values = surface.values.filled(0)
        with warnings.catch_warnings():
            # Can ignore xtgeo-warning for few/zero active nodes
            # (can happen for first map, before injection)
            warnings.filterwarnings("ignore", message=r"Number of maps nodes are*")
            surface.to_file(
                (pathlib.Path(map_folder) / surface.name).with_suffix(".gri")
            )
        if plot_folder and os.path.exists(plot_folder):
            pn = pathlib.Path(plot_folder) / surface.name
            if use_plotly:
                write_plot_using_plotly(surface, pn)
            else:
                write_plot_using_quickplot(surface, pn)
    timer.stop("write_surfaces")


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


def _distribute_config_property(
    properties: Optional[List[_config.Property]],
) -> Union[List[_config.Property], None]:
    if properties is None:
        return properties
    distributed_props = []
    for prop in properties:
        if not isinstance(prop.name, list):
            distributed_props.append(prop)
            continue
        if isinstance(prop.lower_threshold, list):
            if len(prop.name) == len(prop.lower_threshold):
                distributed_props.extend(
                    [
                        _config.Property(prop.source, name, threshold)
                        for name, threshold in zip(prop.name, prop.lower_threshold)
                    ]
                )
            elif len(prop.lower_threshold) == 1:
                logging.info(
                    f"Only one value of threshold for {str(len(prop.name))} "
                    f"properties. The same threshold will be assumed for all the "
                    f"properties."
                )
                distributed_props.extend(
                    [
                        _config.Property(prop.source, name, threshold)
                        for name, threshold in zip(prop.name, prop.lower_threshold)
                    ]
                )
            else:
                error_text = (
                    f"{str(len(prop.lower_threshold))} values of co2_threshold "
                    f"provided, but {str(len(prop.name))} properties in config"
                    f" file input. Fix the amount of values in co2_threshold or "
                    f"the amount of properties in config file"
                )
                raise Exception(error_text)
        elif isinstance(prop.lower_threshold, (float, int)):
            logging.info(
                f"Only one value of threshold for {str(len(prop.name))} "
                f"properties. The same threshold will be assumed for all the "
                f"properties."
            )
            distributed_props.extend(
                [
                    _config.Property(prop.source, name, prop.lower_threshold)
                    for name in prop.name
                ]
            )
        else:
            raise Exception("Unsupported type for lower_threshold")

    return distributed_props


def _init_timer():
    timer = Timer()
    timer.reset_timings()
    timer.code_parts = {
        "read_xtgeo_grid": "Aggregate: Read grid using xtgeo",
        "extract_properties": "Aggregate: Extract properties from files",
        "aggregate_maps": "Aggregate: Aggregate 3D grid to 2D maps",
        "ndarray_to_regsurfs": "Aggregate: Convert results to xtgeo.RegularSurface",
        "write_surfaces": "Aggregate: Write maps to files",
        "logging": "Various logging",
    }


def main(arguments=None):
    """
    Main function that wraps `generate_from_config` with argument parsing
    """
    if arguments is None:
        arguments = sys.argv[1:]
    _init_timer()
    timer = Timer()
    timer.start("total")

    config_ = process_arguments(arguments)
    config_.input.properties = _distribute_config_property(config_.input.properties)
    log_input_configuration(config_, calc_type="aggregate")
    generate_from_config(config_)

    timer.stop("total")
    timer.report()


if __name__ == "__main__":
    main()
