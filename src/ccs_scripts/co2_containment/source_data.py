import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xtgeo

from ccs_scripts.co2_containment.input import RegionInfo, ZoneInfo
from ccs_scripts.utils.gridproperty_tools import GridHandler
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import (
    format_error,
    format_warning,
    identify_gas_less_cells_from_iterator,
    log_saturation_summaries,
)
from ccs_scripts.utils.xtgeo_logging import suppress_xtgeo_warning_by_message

PROPERTIES_NEEDED_CIRRUS = ["SGAS", "DGAS", "DWAT"]
PROPERTIES_NEEDED_ECLIPSE = ["SGAS", "BGAS", "BWAT", "XMF2", "YMF2"]

RELEVANT_PROPERTIES = [
    "RPORV",
    "PORV",
    "SGAS",
    "DGAS",
    "BGAS",
    "SWAT",
    "DWAT",
    "BWAT",
    "SOIL",
    "DOIL",
    "BOIL",
    "AMFG",
    "YMFG",
    "XMFG",
    "AMFS",
    "YMFS",
    "XMFS",
    "AMFW",
    "YMFW",
    "XMFW",
    "XMFO",
    "YMFO",
]


@dataclass
class SourceData:
    """Dataclass holding all grid properties needed for CO2 calculations.

    The XMF/YMF/ZMF per-component mole fractions (Eclipse compositional) are stored
    in typed dicts keyed by component index (1-based), e.g. ``xmfs[2]`` for XMF2.
    """

    x_coord: np.ndarray
    y_coord: np.ndarray
    active_cells: np.ndarray  # 3D array with True where calculations are performed
    DATES: List[str]
    cell_size: Optional[float] = None
    VOL: Optional[Dict[str, np.ndarray]] = None
    SOIL: Optional[Dict[str, np.ndarray]] = None
    SWAT: Optional[Dict[str, np.ndarray]] = None
    SGAS: Optional[Dict[str, np.ndarray]] = None
    SGSTRAND: Optional[Dict[str, np.ndarray]] = None
    SGTRH: Optional[Dict[str, np.ndarray]] = None
    RPORV: Optional[Dict[str, np.ndarray]] = None
    PORV: Optional[Dict[str, np.ndarray]] = None
    AMFG: Optional[Dict[str, np.ndarray]] = None
    YMFG: Optional[Dict[str, np.ndarray]] = None
    XMFG: Optional[Dict[str, np.ndarray]] = None
    DWAT: Optional[Dict[str, np.ndarray]] = None
    DGAS: Optional[Dict[str, np.ndarray]] = None
    DOIL: Optional[Dict[str, np.ndarray]] = None
    BWAT: Optional[Dict[str, np.ndarray]] = None
    BGAS: Optional[Dict[str, np.ndarray]] = None
    BOIL: Optional[Dict[str, np.ndarray]] = None
    AMFS: Optional[Dict[str, np.ndarray]] = None
    YMFS: Optional[Dict[str, np.ndarray]] = None
    XMFS: Optional[Dict[str, np.ndarray]] = None
    AMFW: Optional[Dict[str, np.ndarray]] = None
    YMFW: Optional[Dict[str, np.ndarray]] = None
    XMFW: Optional[Dict[str, np.ndarray]] = None
    XMFO: Optional[Dict[str, np.ndarray]] = None
    YMFO: Optional[Dict[str, np.ndarray]] = None
    zone: Optional[np.ndarray] = None
    region: Optional[np.ndarray] = None
    # Per-component mole fractions for Eclipse compositional runs, keyed by
    # component index (1-based).  E.g. xmfs[2] corresponds to XMF2.
    xmfs: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    ymfs: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    zmfs: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)

    # Names of static property fields (excludes coordinates, DATES, zone, region,
    # and the indexed mf dicts — see active_property_names for a full list).
    _STATIC_PROP_FIELDS = [
        "VOL",
        "SOIL",
        "SWAT",
        "SGAS",
        "SGSTRAND",
        "SGTRH",
        "RPORV",
        "PORV",
        "AMFG",
        "YMFG",
        "XMFG",
        "DWAT",
        "DGAS",
        "DOIL",
        "BWAT",
        "BGAS",
        "BOIL",
        "AMFS",
        "YMFS",
        "XMFS",
        "AMFW",
        "YMFW",
        "XMFW",
        "XMFO",
        "YMFO",
    ]

    def active_property_names(self) -> List[str]:
        """Return names of all non-None properties, including indexed XMF/YMF/ZMF.

        The indexed component properties are expanded to their string names
        (e.g. ``"XMF1"``, ``"XMF2"``) so that functions like ``_n_components``
        and ``_find_source_and_scenario`` work unchanged.
        """
        names: List[str] = [
            name for name in self._STATIC_PROP_FIELDS if getattr(self, name) is not None
        ]
        names.extend(f"XMF{i}" for i in sorted(self.xmfs))
        names.extend(f"YMF{i}" for i in sorted(self.ymfs))
        names.extend(f"ZMF{i}" for i in sorted(self.zmfs))
        return names


class Scenario(Enum):
    """
    Which scenario is CO2 amounts calculated in
    """

    AQUIFER = 0
    DEPLETED_GAS_FIELD = 1
    DEPLETED_OIL_GAS_FIELD = 2


def _detect_eclipse_mole_fraction_props(
    unrst_file: str,
) -> tuple[list[str], list[int], bool]:
    """
    Detects which and how many components are there in Eclipse data.

    Args:
        unrst_file (str): Path to UNRST file

    Returns:
        Tuple of (mole_frac_props, component_indices, has_zmf) where
        component_indices is a list of 1-based int indices found (e.g. [1, 2, 3])
        and has_zmf indicates whether ZMF properties were present.
    """
    unrst_props = xtgeo.list_gridproperties(unrst_file)
    has_zmf = "ZMF1" in unrst_props
    component_indices: List[int] = []
    mole_frac_props = []
    for suffix_count in range(1, 51):
        tmp_x = f"XMF{suffix_count}" in unrst_props
        tmp_y = f"YMF{suffix_count}" in unrst_props
        tmp_z = f"ZMF{suffix_count}" in unrst_props
        if not tmp_x and not tmp_y:
            # Neither XMFi nor YMFi found, assume no more components
            break
        if has_zmf:
            if not tmp_x == tmp_y == tmp_z:
                error_text = (
                    "Error: Number of components with XMF property differ from "
                    "the number of components with YMF or ZMF"
                )
                raise ValueError(format_error(error_text))
            mole_frac_props.extend(
                [name + str(suffix_count) for name in ["XMF", "YMF", "ZMF"]]
            )
        else:
            if not tmp_x == tmp_y:
                error_text = (
                    "Error: Number of components with XMF property differ from "
                    "the number of components with YMF"
                )
                raise ValueError(format_error(error_text))
            mole_frac_props += [f"XMF{suffix_count}", f"YMF{suffix_count}"]
        component_indices.append(suffix_count)
    return mole_frac_props, component_indices, has_zmf


def _find_props_to_extract(
    unrst_file: str, residual_trapping: bool
) -> Tuple[List[str], List[int], bool]:
    """Return (props_to_extract, component_indices, has_zmf)."""
    props_to_extract = copy.deepcopy(RELEVANT_PROPERTIES)
    mole_frac_props, component_indices, has_zmf = _detect_eclipse_mole_fraction_props(
        unrst_file
    )
    props_to_extract.extend(mole_frac_props)
    if residual_trapping:
        props_to_extract.extend(["SGSTRAND", "SGTRH"])
    return props_to_extract, component_indices, has_zmf


# pylint: disable=too-many-arguments
def _extract_source_data_from_properties(
    grid_file: str,
    unrst_file: str,
    component_indices: List[int],
    has_zmf: bool,
    props_to_extract: List[str],
    zone_info: ZoneInfo,
    region_info: RegionInfo,
    init_file: Optional[str] = None,
) -> Tuple[SourceData, xtgeo.Grid]:
    # pylint: disable=too-many-locals, too-many-statements
    """Extracts the properties in props_to_extract from Grid files

    Args:
      grid_file (str): Path to EGRID-file
      unrst_file (str): Path to UNRST-file
      component_indices (List[int]): 1-based indices of XMF/YMF(/ZMF) components found
      has_zmf (bool): Whether ZMF properties were present in the UNRST file
      props_to_extract (List): Names of the properties to be extracted
      init_file (str): Path to INIT-file
      zone_info (ZoneInfo): Zone information
      region_info (RegionInfo): Region information

    Returns:
      Tuple[SourceData, xtgeo.Grid]

    """
    logging.info("Start extracting source data\n")
    grid_handler = GridHandler(Path(grid_file), Path(unrst_file))
    grid = grid_handler.grid
    unrst_names = [p for p in props_to_extract if p in grid_handler.property_names]

    init: xtgeo.GridProperties | None = None
    if init_file is not None:
        try:
            # Extract everything from the init file. This is (probably) small
            # amounts of data compared to the dynamic part
            with suppress_xtgeo_warning_by_message("Unknown simulator code"):
                init = xtgeo.gridproperties_from_file(
                    init_file, grid=grid, names="all"
                )
        except Exception:
            init = None
    if init is None:
        logging.info(format_warning("No INIT-file loaded"))

    # Determine reduced set of active cells based on actnum and gasless cells
    dissolved_props = [d for d in ["AMFS", "AMFG", "XMF2"] if d in unrst_names]
    if len(dissolved_props) == 0 or "SGAS" not in unrst_names:
        error_text = (
            "CO2 containment calculation failed. "
            "Cannot find required properties SGAS+AMFG, SGAS+XMF2 or SGAS+AMFS"
        )
        raise RuntimeError(format_error(error_text))

    unrst_props = grid_handler.read_properties(names=unrst_names, dates="all")
    gasless = identify_gas_less_cells_from_iterator(
        (p.values for p in unrst_props.props if p.name.startswith("SGAS")),
        (p.values for p in unrst_props.props if p.name.startswith(dissolved_props[0])),
    )
    # TODO: whenever active cells are used in general, make
    # sure that they are bool in type, otherwise, unexpected
    # bugs can occur due to numpy treating non-bool arrays as
    # indices. Add assertions everywhere?
    active_cells = grid.actnum_array.astype(bool) & ~gasless

    dates = list(
        dict.fromkeys(unrst_props.dates)
    )  # preserve order, but remove duplicates
    extracted_names = unrst_names
    # dict[property][date] with only active and non-gasless cells
    props_reduced: dict[str, dict[str, np.ndarray]] = {p: {} for p in extracted_names}
    for prop in unrst_props.props:
        parts = prop.name.split("--")
        if len(parts) == 1:
            pname = prop.name
            pdate = prop.date
        else:
            pname = parts[0]
            # Prefer prop.date, but fall back to parsing from the name if not present
            pdate = prop.date or parts[1]
        if pname not in props_reduced:
            continue
        # .values is a masked array. actnum should correspond to the mask, but
        # "active_cells" also include gas-less cells, so we'll use that instead
        props_reduced[pname][pdate] = prop.values[active_cells].data

    # Warn about missing data for any dates
    missing: list[tuple[str, str]] = []
    for d in dates:
        for prop in extracted_names:
            if d not in props_reduced[prop]:
                missing.append((prop, d))
    if missing:
        missing_str = ", ".join([f"{prop} ({d})" for prop, d in missing])
        logging.warning(
            format_warning(
                f"WARNING: The following date-property pairs are missing: {missing_str}"
            )
        )

    log_saturation_summaries(props_reduced)
    # Tuple with (x,y,z) for each cell:
    xp, yp, _ = grid.get_xyz()
    cells_x = xp.values[active_cells].data
    cells_y = yp.values[active_cells].data

    zone = _process_zones(zone_info, grid, active_cells)
    region = _process_regions(region_info, grid, init, active_cells)
    vol = grid.get_bulk_volume().values[active_cells]
    try:
        cell_size = np.median(vol)
        dx = grid.get_dx().values[active_cells].data
        dy = grid.get_dy().values[active_cells].data
        dz = grid.get_dz().values[active_cells].data
        _log_grid_cell_dimensions(vol, dx, dy, dz)
    except Exception as e:
        logging.info(format_warning(f"WARNING: Could not compute grid cell size: {e}"))
        cell_size = None

    props_reduced["VOL"] = {d: vol for d in dates}
    if init is not None:
        porv = init.get_prop_by_name("PORV")
        if not grid_handler.has_lgr:
            if porv is not None:
                props_reduced["PORV"] = {
                    d: porv.values[active_cells].data for d in dates
                }
        elif "RPORV" not in props_reduced:
            # Grids with LGRs will not have valid PORV values for the cells
            # affected by LGR. For Cirrus, these cells will have a value of 1.0
            # (we believe), but other simulators are unclear. In any case,
            # we override these values with PORV calculated from PORO and
            # bulk. The exception is if RPORV is already present, in which case
            # we don't need PORV.
            # TODO: inform users that LGR porv values have been inferred.
            poro = init.get_prop_by_name("PORO")
            porv = init.get_prop_by_name("PORV")
            if poro is not None and porv is not None:
                poro_vals = poro.values[active_cells].data
                porv_vals = porv.values[active_cells].data
                porv_proxy = np.where(porv_vals == 1.0, poro_vals * vol, porv_vals)
                props_reduced["PORV"] = {d: porv_proxy for d in dates}
    # Infer SOIL from SGAS and SWAT if not stored in the file.
    # Some simulators (e.g. Eclipse compositional with 3 phases) store SGAS and
    # SWAT but not SOIL. SOIL = 1 - SGAS - SWAT in those cases, and its presence
    # is needed to detect the DEPLETED_OIL_GAS_FIELD scenario.
    if (
        "SOIL" not in props_reduced
        and "SGAS" in props_reduced
        and "SWAT" in props_reduced
    ):
        tol = 1e-6
        soil: dict[str, np.ndarray] = {}
        for d in dates:
            if d in props_reduced["SGAS"] and d in props_reduced["SWAT"]:
                soil[d] = np.maximum(
                    0.0, 1.0 - props_reduced["SGAS"][d] - props_reduced["SWAT"][d]
                )
        max_soil = max((v.max() for v in soil.values()), default=0.0)
        if max_soil > tol:
            props_reduced["SOIL"] = soil
            logging.info(
                "Oil Saturation (SOIL) not found as property."
                "\nHowever, SGAS + SWAT != 1 somewhere, so SOIL has been inferred"
                " as 1 - SGAS - SWAT."
            )
        else:
            logging.info(
                "Oil Saturation is zero everywhere. Two-phase scenario is assumed."
            )

    # Separate the indexed mole-fraction props from the static ones
    xmfs = {
        i: props_reduced.pop(f"XMF{i}")
        for i in component_indices
        if f"XMF{i}" in props_reduced
    }
    ymfs = {
        i: props_reduced.pop(f"YMF{i}")
        for i in component_indices
        if f"YMF{i}" in props_reduced
    }
    zmfs = (
        {
            i: props_reduced.pop(f"ZMF{i}")
            for i in component_indices
            if f"ZMF{i}" in props_reduced
        }
        if has_zmf
        else {}
    )
    source_data = SourceData(
        cells_x,
        cells_y,
        active_cells,
        dates,
        cell_size,
        **dict(props_reduced.items()),
        zone=zone,
        region=region,
        xmfs=xmfs,
        ymfs=ymfs,
        zmfs=zmfs,
    )
    logging.info("\nDone extracting source data\n")
    return source_data, grid


def _log_grid_cell_dimensions(
    vol0: list, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray
) -> None:
    vol0_scaled = np.array(vol0) / 1000.0

    dimensions = [
        ("dx (m)", dx),
        ("dy (m)", dy),
        ("dz (m)", dz),
        ("vol (1000 m^3)", vol0_scaled),
    ]

    header = (
        f"\n{'Grid dimension':<15} {'Min':>12} {'P10':>12} "
        f"{'Median':>12} {'Mean':>12} {'P90':>12} {'Max':>12}"
    )
    logging.info(header)
    logging.info(f"{'-' * 93}")

    for label, values in dimensions:
        row = (
            f"{label:<15} "
            f"{values.min():>12.1f} "
            f"{np.percentile(values, 10):>12.1f} "
            f"{np.median(values):>12.1f} "
            f"{values.mean():>12.1f} "
            f"{np.percentile(values, 90):>12.1f} "
            f"{values.max():>12.1f}"
        )
        logging.info(row)


def _check_grid_dimensions(
    roff_file: str,
    grid: xtgeo.Grid,
) -> None:
    roff_grid = xtgeo.gridproperty_from_file(roff_file)
    roff_shape = roff_grid.values.shape
    if roff_shape != grid.dimensions:
        err = f"Inconsistent grid dimensions {roff_shape} from file {roff_file}"
        err += f" and {grid.dimensions} from file {grid.filesrc}."
        raise ValueError(format_error(err))


def _process_zones(
    zone_info: ZoneInfo,
    grid: xtgeo.Grid,
    active_cells: np.ndarray,
) -> Optional[np.ndarray]:
    zone = None
    if zone_info.source is None:
        logging.info("No zone info specified")
        return None
    logging.info("Using zone info")
    if zone_info.zranges is not None:
        zone_array = np.zeros(grid.dimensions, dtype=int)
        zonevals = [int(x) for x in range(len(zone_info.zranges))]
        zone_info.int_to_zone = [f"Zone_{x}" for x in range(len(zonevals))]
        for zv, zr, zn in zip(
            zonevals,
            list(zone_info.zranges.values()),
            zone_info.zranges.keys(),
        ):
            zone_array[:, :, zr[0] - 1 : zr[1]] = zv
            zone_info.int_to_zone[zv] = zn
        return zone_array[active_cells]
    _check_grid_dimensions(zone_info.source, grid)
    zone = xtgeo.gridproperty_from_file(zone_info.source, grid=grid)
    try:
        zone_name_dict = zone.codes
        zone_values = list(zone_name_dict.keys())
    except AttributeError:
        zone_name_dict = {}
        zone_values = []
    zonevals = list(np.unique(zone.values[~zone.values.mask]))
    intvals = np.array(zonevals, dtype=int)
    if np.sum(intvals == zonevals) != len(zonevals):
        warning_text = (
            "Warning: Grid provided in zone file contains non-integer values. "
            "This might cause problems with the calculations for "
            "containment in different zones."
        )
        logging.info(format_warning(warning_text))
    zone_info.int_to_zone = [None] * (np.max(intvals) + 1)
    for zv in intvals:
        if zv >= 0:
            if zv in zone_values:
                zone_info.int_to_zone[zv] = zone_name_dict[zv]
            else:
                zone_info.int_to_zone[zv] = f"Zone_{zv}"
                logging.info(
                    f"Value {zv} in roff-grid not found in Codes."
                    f" Using generic zone name Zone_{zv}."
                )
        else:
            logging.info("Ignoring negative value in grid from zone file.")
    return zone.values.data[active_cells]


def _process_regions(
    region_info: RegionInfo,
    grid: xtgeo.Grid,
    init: xtgeo.GridProperties | None,
    active: np.ndarray,
) -> Optional[np.ndarray]:
    region = None
    if region_info.source is not None:
        logging.info("Using regions info")
        _check_grid_dimensions(
            region_info.source,
            grid,
        )
        region = xtgeo.gridproperty_from_file(region_info.source, grid=grid)
        try:
            region_name_dict = region.codes
            region_values = list(region_name_dict.keys())
        except AttributeError:
            region_name_dict = {}
            region_values = []
        regvals = np.unique(region.values.data[~region.values.mask])
        intvals = np.array(regvals, dtype=int)
        if np.sum(intvals == regvals) != len(regvals):
            warning_text = (
                "Warning: Grid provided in region file contains non-integer values. "
                "This might cause problems with the calculations for "
                "containment in different regions."
            )
            logging.info(warning_text)
        region_info.int_to_region = [None] * (np.max(intvals) + 1)
        for rv in intvals:
            if rv >= 0:
                if rv in region_values:
                    region_info.int_to_region[rv] = region_name_dict[rv]
                else:
                    region_info.int_to_region[rv] = f"Region_{rv}"
                    logging.info(
                        f"Value {rv} in roff-grid not found in Codes."
                        f" Using generic region name Region_{rv}."
                    )
            else:
                logging.info("Ignoring negative value in grid from region file.")
        return np.array(region[active], dtype=int)
    if region_info.property_name is not None:
        if init is None:
            logging.info("No INIT-file to use for region information.")
            region = None
            region_info.int_to_region = None
        else:
            try:
                logging.info(
                    f"Try reading region information ({region_info.property_name}"
                    f" property) from INIT-file."
                )
                region_prop = init.get_prop_by_name(region_info.property_name)
                if region_prop is None or region_prop.dimensions != grid.dimensions:
                    logging.info(
                        "Warning: Failed to use region property in INIT-file has due"
                        " to either different dimensions or a missing property."
                    )
                    region_info.int_to_region = None
                    return None

                region = region_prop.values[active]
                regvals = np.unique(region)
                region_info.int_to_region = [None] * (np.max(regvals) + 1)
                for rv in regvals:
                    if rv >= 0:
                        region_info.int_to_region[rv] = f"Region_{rv}"
                    else:
                        logging.info(
                            f"Ignoring negative value in {region_info.property_name}."
                        )
                logging.info("Region information successfully read from INIT-file")
            except KeyError:
                logging.info(
                    format_warning("Region information not found in INIT-file.")
                )
                region = None
                region_info.int_to_region = None
    return region


def extract_source_data(
    grid_file: str,
    unrst_file: str,
    zone_info: ZoneInfo,
    region_info: RegionInfo,
    residual_trapping: bool = False,
    init_file: Optional[str] = None,
    return_grid: bool = False,
) -> Tuple[SourceData, Optional[xtgeo.Grid]]:
    """Extracts the properties needed for CO2 containment calculations from EGRID and UNRST files.

    Args:
        grid_file (str): Path to EGRID-file
        unrst_file (str): Path to UNRST-file
        zone_info (ZoneInfo): Zone information
        region_info (RegionInfo): Region information
        calc_type (CalculationType): Type of CO2 calculation
        residual_trapping (bool): Whether to consider residual trapping
        init_file (Optional[str]): Path to INIT-file

    Returns:
        Tuple[SourceData, Optional[xtgeo.Grid]]: Extracted source data and optionally the grid
    """
    timer = Timer()
    timer.start("extract_source_data")
    props_to_extract, component_indices, has_zmf = _find_props_to_extract(
        unrst_file, residual_trapping
    )
    source_data, grid = _extract_source_data_from_properties(
        grid_file,
        unrst_file,
        component_indices,
        has_zmf,
        props_to_extract,
        zone_info,
        region_info,
        init_file,
    )
    timer.stop("extract_source_data")
    if return_grid:
        return source_data, grid
    return source_data, None
