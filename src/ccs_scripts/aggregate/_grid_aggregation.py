import dataclasses
import logging
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse
import xtgeo

from ccs_scripts.aggregate._config import AggregationMethod
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import format_error, format_warning


def aggregate_maps(
    map_template: Union[xtgeo.RegularSurface, float],
    grid: xtgeo.Grid,
    grid_props: List[xtgeo.GridProperty],
    filters: List[Tuple[str, Optional[np.ndarray]]],
    method: AggregationMethod,
    weight_by_dz: bool,
) -> Tuple[np.ndarray, np.ndarray, List[List[np.ndarray]]]:
    """
    Aggregate multiple grid properties, using multiple grid cell filters, to 2D maps.

    Args:
        map_template: Template to use for the generated maps. If a float is provided, it
            will be used as an approximate pixel-to-cell-size ratio to automatically set
            map bounds and resolution from the grid.
        grid: The 3D grid
        grid_props: List of the grid properties to be aggregated
        filters: List of tuples (filter_name, inclusion_filter). Each inclusion_filter
            is either a numpy array or `None`. If a numpy array is used, it must be a
            boolean 1D array representing which cells (among the active cells) that are
            to be included. A `1` indicates inclusion. If `None` is provided, all of the
            grid cells are included.
        method: The aggregation method to apply for pixels that overlap more than one
            grid cell in the xy-plane
        weight_by_dz: Weights cells by thickness (dz) when aggregating. Not relevant if
            method is MIN or MAX

    Returns:
        Doubly nested list of maps. The first index corresponds to `filters`,
        and the second index to `grid_props`.
    """
    # pylint: disable=too-many-arguments
    timer = Timer()
    timer.start("aggregate_maps")
    filter_names = [f[0] for f in filters]
    inclusion_filters = [f[1] for f in filters]
    props, active_cells, inclusion_filters = _read_properties_and_find_active_cells(
        grid, grid_props, inclusion_filters
    )
    weights = grid.get_dz().values1d[active_cells] if weight_by_dz else None
    # Map nodes (pixel locations) and connections
    conn_data = _find_connections(
        grid,
        active_cells,
        map_template,
    )
    # Iterate filters
    results = _properties_to_maps(
        inclusion_filters,
        props,
        weights,
        method,
        conn_data,
    )

    if method in [AggregationMethod.SUM, AggregationMethod.DISTRIBUTE]:
        prop_names = [p.name for p in grid_props]
        _check_cell_coverage(
            props,
            inclusion_filters,
            conn_data,
            prop_names,
            filter_names,
            grid,
            active_cells,
        )

    timer.stop("aggregate_maps")
    return conn_data.x_nodes, conn_data.y_nodes, results


def _read_properties_and_find_active_cells(
    grid: xtgeo.Grid,
    grid_props: List[xtgeo.GridProperty],
    inclusion_filters: List[Optional[np.ndarray]],
):
    active = grid.actnum_array.flatten().astype(bool)
    props = [p.values1d[active] for p in grid_props]
    all_masked = np.all([p.mask for p in props], axis=0)
    if isinstance(all_masked, np.bool_):
        all_masked = np.full(len(props[0]), all_masked)
    if not any(i is None for i in inclusion_filters):
        inclusion_filters_copy = [x for x in inclusion_filters if x is not None]
        all_masked |= ~np.any(inclusion_filters_copy, axis=0)
    active[active] = ~all_masked
    if not any(active):
        error_text = (
            "\nERROR: All grid cells are inactive. "
            "This can be due to a high input threshold."
        )
        logging.error(format_error(error_text))
        sys.exit(1)
    props = [p[~all_masked] for p in props]
    inclusion_filters = [
        None if inc is None else inc[~all_masked] for inc in inclusion_filters
    ]
    return props, active, inclusion_filters


@dataclass
class _ConnectionData:
    """
    Helper dataclass containing information connecting map nodes to grid cells
    """

    x_nodes: np.ndarray
    y_nodes: np.ndarray
    node_indices: np.ndarray
    grid_indices: np.ndarray


def _check_cell_coverage(
    props: List[np.ma.MaskedArray],
    inclusion_filters: List[Optional[np.ndarray]],
    conn_data: _ConnectionData,
    prop_names: List[str],
    filter_names: Optional[List[str]] = None,
    grid: Optional[xtgeo.Grid] = None,
    active_cells: Optional[np.ndarray] = None,
) -> None:
    """
    Check if all valid (non-masked) grid cells are included in
    the aggregation. Warns if significant values are lost, and
    reports the worst affected property. Distinguishes between
    cells outside map extent vs cells within extent but unmapped.

    Args:
        props: List of properties (after filtering for active cells)
        inclusion_filters: List of inclusion filters
        conn_data: Connection data between grid and map nodes
        prop_names: List of property names for better reporting
        filter_names: Names of the filters (e.g., zone names) for better reporting
        grid: The 3D grid (for getting cell coordinates)
        active_cells: Boolean array of active cells
    """
    logging.info(
        "\nChecking cell coverage for aggregation with SUM or DISTRIBUTE method."
    )

    # Get cell centers if grid is provided
    cell_centers_xy = None
    if grid is not None and active_cells is not None:
        cell_centers = grid.get_xyz()
        cell_x = cell_centers[0].values1d[active_cells]
        cell_y = cell_centers[1].values1d[active_cells]
        cell_centers_xy = np.column_stack((cell_x, cell_y))

    # Determine map extent
    map_x_min, map_x_max = conn_data.x_nodes.min(), conn_data.x_nodes.max()
    map_y_min, map_y_max = conn_data.y_nodes.min(), conn_data.y_nodes.max()
    # Add half pixel buffer
    x_inc = (
        conn_data.x_nodes[1] - conn_data.x_nodes[0] if len(conn_data.x_nodes) > 1 else 0
    )
    y_inc = (
        conn_data.y_nodes[1] - conn_data.y_nodes[0] if len(conn_data.y_nodes) > 1 else 0
    )
    map_x_min -= x_inc / 2
    map_x_max += x_inc / 2
    map_y_min -= y_inc / 2
    map_y_max += y_inc / 2

    for filter_idx, incl in enumerate(inclusion_filters):
        grd_ix = conn_data.grid_indices

        if incl is not None:
            to_remove = ~np.isin(grd_ix, np.argwhere(incl).flatten())
            grd_ix = grd_ix[~to_remove]
        unique_mapped_cells = np.unique(grd_ix)

        # Find cells with ANY valid data across all properties
        any_valid_cells = np.zeros(len(props[0]), dtype=bool)
        for prop in props:
            if hasattr(prop, "mask"):
                valid_cells = ~prop.mask
            else:
                valid_cells = np.ones(len(prop), dtype=bool)
            if incl is not None:
                valid_cells = valid_cells & incl
            any_valid_cells |= valid_cells
        total_cells_with_data = np.sum(any_valid_cells)

        if total_cells_with_data == 0:
            continue

        # Find unmapped cells:
        cell_indices_with_data = np.argwhere(any_valid_cells).flatten()
        unmapped_cell_indices = np.setdiff1d(
            cell_indices_with_data, unique_mapped_cells
        )

        # Split unmapped cells into those inside vs outside map extent
        unmapped_inside = unmapped_cell_indices
        unmapped_outside = np.array([], dtype=int)

        if cell_centers_xy is not None and len(unmapped_cell_indices) > 0:
            unmapped_coords = cell_centers_xy[unmapped_cell_indices]
            inside_extent = (
                (unmapped_coords[:, 0] >= map_x_min)
                & (unmapped_coords[:, 0] <= map_x_max)
                & (unmapped_coords[:, 1] >= map_y_min)
                & (unmapped_coords[:, 1] <= map_y_max)
            )
            unmapped_inside = unmapped_cell_indices[inside_extent]
            unmapped_outside = unmapped_cell_indices[~inside_extent]

        num_unmapped = len(unmapped_cell_indices)
        num_unmapped_inside = len(unmapped_inside)
        num_unmapped_outside = len(unmapped_outside)
        percentage_unmapped = (num_unmapped / total_cells_with_data) * 100.0

        if filter_names is not None and filter_idx < len(filter_names):
            filter_name = filter_names[filter_idx]
        else:
            filter_name = (
                f"filter_{filter_idx}" if len(inclusion_filters) > 1 else "all"
            )

        if num_unmapped == 0:
            logging.info(f"Cell coverage for '{filter_name}': all cells mapped")
            continue

        percentage_inside = (num_unmapped_inside / total_cells_with_data) * 100.0
        percentage_outside = (num_unmapped_outside / total_cells_with_data) * 100.0

        # Find the property with the highest lost value for each category
        def _find_worst_property(cell_indices):
            if len(cell_indices) == 0:
                return None
            worst = None
            for prop_idx, prop in enumerate(props):
                if hasattr(prop, "mask"):
                    valid = ~prop.mask[cell_indices]
                else:
                    valid = np.ones(len(cell_indices), dtype=bool)
                if not np.any(valid):
                    continue
                lost_val = np.sum(prop[cell_indices][valid])
                total_val = (
                    np.sum(prop[~prop.mask]) if hasattr(prop, "mask") else np.sum(prop)
                )
                lost_pct = (lost_val / total_val * 100.0) if total_val != 0 else 0.0
                name = (
                    prop_names[prop_idx]
                    if prop_idx < len(prop_names)
                    else f"property_{prop_idx}"
                )
                if worst is None or abs(lost_val) > abs(worst[1]):
                    worst = (name, lost_val, lost_pct)
            return worst

        worst_res = _find_worst_property(unmapped_inside)
        worst_ext = _find_worst_property(unmapped_outside)

        # Determine worst lost percentage across both categories
        worst_lost_pct = 0.0
        if worst_res is not None:
            worst_lost_pct = max(worst_lost_pct, abs(worst_res[2]))
        if worst_ext is not None:
            worst_lost_pct = max(worst_lost_pct, abs(worst_ext[2]))

        lost_value_threshold = 1.0  # percent of total value

        if worst_lost_pct < lost_value_threshold:
            logging.info(
                f"Cell coverage for '{filter_name}': "
                f"{percentage_unmapped:.2f}% of cells "
                f"({num_unmapped}/{total_cells_with_data}) are "
                f"unmapped, but worst lost value is only "
                f"{worst_lost_pct:.3f}% of total"
            )
        else:
            warning_text = (
                f"\nWARNING: {percentage_unmapped:.2f}% of grid cells with data "
                f"({num_unmapped}/{total_cells_with_data}) are not used in the "
                f"aggregation for '{filter_name}'."
                f"\n         Due to map resolution : {percentage_inside:.2f}% "
                f"({num_unmapped_inside} cells within extent but between pixels)"
                f"\n         Due to map extent     : {percentage_outside:.2f}% "
                f"({num_unmapped_outside} cells outside the map boundaries)"
            )
            if num_unmapped_inside > 0 and num_unmapped_outside > 0:
                warning_text += (
                    "\n         Consider both finer resolution"
                    " and extending map extent."
                )
            elif num_unmapped_inside > 0:
                warning_text += (
                    "\n         Consider using a finer map"
                    " resolution (smaller pixel size)."
                )
            elif num_unmapped_outside > 0:
                warning_text += (
                    "\n         Consider extending the map"
                    " extent or using automatic bounds."
                )
            logging.warning(format_warning(warning_text))

            if worst_res is not None or worst_ext is not None:
                logging.warning("         Worst affected property per category:")
            if worst_res is not None:
                logging.warning(
                    f"           Resolution : {worst_res[0]} "
                    f"(lost {worst_res[1]:.2f}, {worst_res[2]:.2f}% of total)"
                )
            if worst_ext is not None:
                logging.warning(
                    f"           Extent     : {worst_ext[0]} "
                    f"(lost {worst_ext[1]:.2f}, {worst_ext[2]:.2f}% of total)"
                )


def _find_connections(
    grid: xtgeo.Grid,
    active_cells: np.ndarray,
    map_template: Union[xtgeo.RegularSurface, float],
) -> _ConnectionData:
    footprints_x, footprints_y = _cell_footprints(grid, active_cells)
    if isinstance(map_template, xtgeo.RegularSurface):
        x_nodes = map_template.xori + map_template.xinc * np.arange(
            0, map_template.ncol, dtype=float
        )
        y_nodes = map_template.yori + map_template.yinc * np.arange(
            0, map_template.nrow, dtype=float
        )
    else:
        x_nodes, y_nodes = _derive_map_nodes(
            footprints_x, footprints_y, pixel_to_cell_size_ratio=map_template
        )
    # Find connections
    return _connect_grid_and_map(
        x_nodes,
        y_nodes,
        footprints_x,
        footprints_y,
    )


def _derive_map_nodes(footprints_x, footprints_y, pixel_to_cell_size_ratio):
    box_x = _footprint_bounds(footprints_x)
    box_y = _footprint_bounds(footprints_y)
    res = np.mean([np.mean(box_x[1] - box_x[0]), np.mean(box_y[1] - box_y[0])])
    res /= pixel_to_cell_size_ratio
    x_nodes = np.arange(np.min(box_x) + res / 2, np.max(box_x) - res / 2 + 1e-12, res)
    y_nodes = np.arange(np.min(box_y) + res / 2, np.max(box_y) - res / 2 + 1e-12, res)
    return x_nodes, y_nodes


def _extract_all_overlaps(i_starts, i_range, j_starts, j_range):
    ij_pairs = []
    indices = []
    for ni in range(1, i_range.max() + 1):
        for nj in range(1, j_range.max() + 1):
            ix = (i_range == ni) & (j_range == nj)
            if ix.sum() == 0:
                continue
            i0 = i_starts[ix]
            j0 = j_starts[ix]
            for qi in range(ni):
                for qj in range(nj):
                    ij_pairs.append(
                        np.column_stack(
                            (
                                i0 + qi,
                                j0 + qj,
                            )
                        )
                    )
            indices.append(
                np.kron(np.ones(ni * nj, dtype=int), np.argwhere(ix).flatten())
            )
    return np.vstack(ij_pairs), np.hstack(indices)


def _connect_grid_and_map(
    x_nodes,
    y_nodes,
    footprints_x,
    footprints_y,
) -> _ConnectionData:
    """
    Returns a mapping between the provided grid nodes and map pixels as an np.ndarray
    pair, the first referring to pixel indices and the second to corresponding grid
    indices. A grid node may be mapped to multiple pixels, and vice verse.
    """
    box_x = _footprint_bounds(footprints_x)
    box_y = _footprint_bounds(footprints_y)
    i0, i_range = _find_overlapped_nodes(x_nodes, box_x[0], box_x[1])
    j0, j_range = _find_overlapped_nodes(y_nodes, box_y[0], box_y[1])
    pixels_ij, grd_ix = _extract_all_overlaps(i0, i_range, j0, j_range)
    map_ix = np.ravel_multi_index(pixels_ij.T, (x_nodes.size, y_nodes.size))
    box_conn_data = _ConnectionData(x_nodes, y_nodes, map_ix, grd_ix)
    return _filter_on_footprint(box_conn_data, footprints_x, footprints_y)


def _find_overlapped_nodes(nodes, cell_lower, cell_upper):
    i0 = np.searchsorted(nodes, cell_lower)
    i1 = np.searchsorted(nodes, cell_upper)
    lengths = i1 - i0
    return i0, lengths


def _cell_footprints(grid: xtgeo.Grid, active_cells):
    corners = grid.get_xyz_corners()
    xyz = [c.values1d[active_cells] for c in corners]
    avg_xyz = [(xyz[i] + xyz[i + 12]) / 2 for i in range(12)]
    x_corners = avg_xyz[::3]
    y_corners = avg_xyz[1::3]
    # z_corners = avg_xyz[2::3]
    fp_x = np.array(x_corners)
    fp_y = np.array(y_corners)
    fp_x = fp_x[[0, 1, 3, 2]].T
    fp_y = fp_y[[0, 1, 3, 2]].T
    return fp_x, fp_y


def _footprint_bounds(footprints):
    return np.min(footprints, axis=1), np.max(footprints, axis=1)


def _filter_on_footprint(conn_data: _ConnectionData, footprint_x, footprint_y):
    """
    Filters pixel-grid cell connections based on the actual polygonal footprint of a
    grid cell, not only the bounding box around the grid cell.
    """
    nodes_xx, nodes_yy = np.meshgrid(
        conn_data.x_nodes, conn_data.y_nodes, indexing="ij"
    )
    nodes_xx = nodes_xx.flatten()
    nodes_yy = nodes_yy.flatten()
    # N - number of pixel-grid cell pairs
    pt_in_cell = _point_in_quadrangle(
        quad_x=footprint_x[conn_data.grid_indices],  # N x 4
        quad_y=footprint_y[conn_data.grid_indices],  # N x 4
        point_x=nodes_xx[conn_data.node_indices],  # N
        point_y=nodes_yy[conn_data.node_indices],  # N
    )
    return dataclasses.replace(
        conn_data,
        grid_indices=conn_data.grid_indices[pt_in_cell],
        node_indices=conn_data.node_indices[pt_in_cell],
    )


def _point_in_quadrangle(
    quad_x: np.ndarray,  # N x 4
    quad_y: np.ndarray,  # N x 4
    point_x: np.ndarray,  # N
    point_y: np.ndarray,  # N
):
    # Wrap polygons so that first point is duplicated as the last. Transpose for easier
    # indexing.
    quad_x = np.column_stack((quad_x, quad_x[:, 0])).T  # 5 x N
    quad_y = np.column_stack((quad_y, quad_y[:, 0])).T  # 5 x N
    # The following checks if a given node is contained within its corresponding grid
    # cell based on the following. A line is drawn in y-direction from the pixel
    # location to negative infinity. The number of intersections between this line and
    # the lines comprising the polygon is counted. If this number is odd, the pixel is
    # within the grid cell, otherwise, it is outside.
    dx = np.diff(quad_x, axis=0)
    too_small = np.abs(dx) < 1e-12
    dx[too_small] = (dx[too_small] >= 0.0) * 1e-12
    # w and interp_y is calculated everywhere, but technically only used where overlap_x
    # is True. For performance, we can calculate overlap_x first, then interp_y and w
    # only where overlap_x is True. Future work...
    rx = (point_x - quad_x[:-1]) / dx
    interp_y = (1 - rx) * quad_y[:-1] + rx * quad_y[1:]
    overlap_x = (rx <= 1.0) & (rx >= 0.0)
    overlap_y = interp_y >= point_y
    crosses = (overlap_x & overlap_y).astype(int).sum(axis=0)
    return (crosses % 2) == 1


def _properties_to_maps(
    inclusion_filters: List[Optional[np.ndarray]],
    properties: List[np.ma.MaskedArray],
    weights: Optional[np.ma.MaskedArray],
    method: AggregationMethod,
    conn_data: _ConnectionData,
):
    results: List[Any] = []
    for incl in inclusion_filters:
        map_ix = conn_data.node_indices
        grd_ix = conn_data.grid_indices
        if incl is not None:
            to_remove = ~np.isin(grd_ix, np.argwhere(incl).flatten())
            map_ix = map_ix[~to_remove]
            grd_ix = grd_ix[~to_remove]
        results.append([])
        for prop in properties:
            results[-1].append(
                _property_to_map(
                    dataclasses.replace(
                        conn_data,
                        node_indices=map_ix,
                        grid_indices=grd_ix,
                    ),
                    prop,
                    weights,
                    method,
                )
            )
    return results


def _property_to_map(
    conn_data: _ConnectionData,
    prop: np.ndarray,
    weights: Optional[np.ndarray],
    method: AggregationMethod,
):
    rows = conn_data.node_indices
    cols = conn_data.grid_indices
    assert rows.shape == cols.shape
    assert weights is None or weights.shape == prop.shape
    if weights is not None:
        assert method in [AggregationMethod.MEAN, AggregationMethod.SUM]
    data = prop[cols]
    weights = np.ones_like(data) if weights is None else weights[cols]
    if data.mask.any():
        invalid = data.mask
        rows = rows[~invalid]
        cols = cols[~invalid]
        data = data[~invalid]
        if weights is not None:
            weights = weights[~invalid]

    nx, ny = conn_data.x_nodes.size, conn_data.y_nodes.size
    if data.size == 0:
        return np.full((nx, ny), fill_value=np.nan)
    # Calculate temporary data shift to avoid unintended deletion of data by tocsc:
    if method == AggregationMethod.MAX:
        shift = data.min() - 1
    elif method == AggregationMethod.MIN:
        shift = data.max() + 1
    elif method in (
        AggregationMethod.MEAN,
        AggregationMethod.SUM,
        AggregationMethod.DISTRIBUTE,
    ):
        shift = 0.0
    else:
        raise NotImplementedError
    shape = (nx * ny, max(cols) + 1)
    res = _aggregate_sparse_data(
        values=scipy.sparse.coo_matrix(
            (data - shift, (rows, cols)), shape=shape
        ).tocsc(),
        weight=scipy.sparse.coo_matrix((weights, (rows, cols)), shape=shape).tocsc(),
        method=method,
    )
    res += shift
    res = res.reshape(nx, ny)
    return res


def _aggregate_sparse_data(
    values: scipy.sparse.csc_matrix,
    weight: scipy.sparse.csc_matrix,
    method: AggregationMethod,
):
    total_weight = weight.sum(axis=1)
    # Make sure to shift data to avoid
    if method == AggregationMethod.MAX:
        res = values.max(axis=1).toarray()
    elif method == AggregationMethod.MIN:
        res = values.min(axis=1).toarray()
    elif method == AggregationMethod.MEAN:
        div = np.where(total_weight > 0, total_weight, 1)  # Avoid division by zero
        res = (values.multiply(weight)).sum(axis=1) / div
        res = np.asarray(res)
    elif method == AggregationMethod.SUM:
        # res = np.asarray(sarr.sum(axis=1))
        res = np.asarray((values.multiply(weight)).sum(axis=1))
    elif method == AggregationMethod.DISTRIBUTE:
        # This method distributes the values from a grid cell to all the pixels that
        # overlap with this grid cell. This is especially relevant for maps where the
        # total sum needs to be preserved, e.g. when aggregating mass.
        w_csr = weight.tocsr()
        col_sum = np.asarray(w_csr.sum(axis=0))
        div = np.where(col_sum > 0, col_sum, 1)
        w_csr = w_csr / div
        res = np.asarray((values.multiply(w_csr.tocsc())).sum(axis=1))
    else:
        raise NotImplementedError
    res = res.flatten()
    total_weight = np.array(total_weight).flatten()
    res[total_weight == 0] = np.nan
    return res
