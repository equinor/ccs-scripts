import warnings
from pathlib import Path
from typing import Literal

import xtgeo

from ccs_scripts.utils.xtgeo_logging import suppress_xtgeo_warning_by_message


class GridHandler:
    def __init__(
        self,
        grid_file_path: Path,
        properties_file_path: Path,
        monkey_patch_xtgeo: bool = True,
    ):
        # The purpose of this class is to hide the implementation details of how the
        # grid and properties are read. A specific issue when using xtgeo directly is
        # that it is slow compared to e.g. resdata when reading a lot of properties
        # from a UNRST file. One of the main bottlenecks at the time of writing is that
        # actnum is extracted (and copied) twice per property. A hacky fix is to
        # monkey-patch these two methods on the grid.
        #
        # Another possibility we might explore in the future is lazy-reading properties
        # to improve memory usage, but this is not currently implemented.
        self._grid, self._has_lgr = _read_grid(grid_file_path)
        self._properties_file = properties_file_path
        self._available_properties = xtgeo.list_gridproperties(self._properties_file)
        if monkey_patch_xtgeo:
            # Create a copy of the grid so that the monkey-patched version is only
            # handled locally in this class. This is to avoid unintended consequences
            # in other parts of the code, making it simpler to remove the
            # monkey-patching later if needed.
            #
            # The memory and performance cost of duplicating the grid is negligible
            # compared to the cost of reading properties
            self._property_grid = self._grid.copy()
            _monkey_patch_xtgeo_grid(self._property_grid)
        else:
            self._property_grid = self._grid

    @property
    def grid(self) -> xtgeo.Grid:
        return self._grid

    @property
    def has_lgr(self) -> bool:
        return self._has_lgr

    @property
    def property_names(self) -> list[str]:
        return self._available_properties

    def read_properties(
        self,
        names: list[str] | Literal["all"],
        dates: list[str] | Literal["all"],
    ) -> xtgeo.GridProperties:
        with suppress_xtgeo_warning_by_message("Unknown simulator code"):
            result = xtgeo.gridproperties_from_file(
                self._properties_file,
                names=names,
                dates=dates,
                grid=self._property_grid,
                namestyle=1,
            )
        return result


def _read_grid(grid_file: Path) -> tuple[xtgeo.Grid, bool]:
    # Read grid file. Currently, the LGRs are not supported by xtgeo grids
    # and all LGR information seems to be discarded during reading. However,
    # a warning is raised, and we'll use this as an indication of whether LGRs
    # are present or not.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Temp suppress these warnings.
        # Can remove if input data or xtgeo behaviour changes.
        warnings.filterwarnings(
            "ignore", "EGrid file given with numres < 1", UserWarning
        )
        warnings.filterwarnings("ignore", "Unknown simulator code -1", UserWarning)
        grid = xtgeo.grid_from_file(grid_file)
    has_lgr = any(
        "egrid file contains local grid refinements (LGR)" in str(warn.message)
        for warn in w
    )
    return grid, has_lgr


def _monkey_patch_xtgeo_grid(grid: xtgeo.Grid) -> None:
    grid_actnum = grid.get_actnum(asmasked=True)
    grid_actnum_f = grid.get_actnum_indices(order="F")

    def _monkey_actnum(asmasked=True):
        return grid_actnum

    def _monkey_actnum_indices(order="F"):
        return grid_actnum_f

    grid.get_actnum = _monkey_actnum
    grid.get_actnum_indices = _monkey_actnum_indices
