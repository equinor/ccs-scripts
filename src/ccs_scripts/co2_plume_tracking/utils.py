import itertools
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from resdata.grid import Grid

MAX_STEPS_RESOLVE_CELLS = 20
MAX_NEAREST_GROUPS_SEARCH_DISTANCE = 3


@dataclass
class InjectionWellData:
    name: str
    x: float
    y: float
    z: Optional[
        List[float]
    ]  # Normally only 1 value, but we might keep multiple when z is not provided
    number: int


class Status(Enum):
    UNDETERMINED = 0
    NO_CO2 = 1
    HAS_CO2 = 2


class CellGroup:
    def __init__(self, groups: Optional[List[int]] = None):
        self.status: Status = Status.NO_CO2
        self.all_groups: List[int] = []
        if groups is not None:
            self.status = Status.HAS_CO2
            self.all_groups = groups.copy()

    def set_cell_groups(self, new_groups: List[int]):
        self.status = Status.HAS_CO2
        self.all_groups = new_groups.copy()

    def set_undetermined(self):
        self.status = Status.UNDETERMINED
        self.all_groups = []

    def has_co2(self):
        return self.status == Status.HAS_CO2

    def has_no_co2(self):
        return self.status == Status.NO_CO2

    def is_undetermined(self):
        return self.status == Status.UNDETERMINED


class PlumeGroups:
    def __init__(self, number_of_grid_cells: Optional[int] = None):
        self.cells: List[CellGroup] = []
        if number_of_grid_cells is not None:
            self.cells = [CellGroup() for _ in range(0, number_of_grid_cells)]

    def copy(self):
        out = PlumeGroups()
        out.cells = self.cells.copy()
        return out

    def resolve_undetermined_cells(self, grid: Grid) -> List:
        ind_to_resolve = [
            ind for ind, group in enumerate(self.cells) if group.is_undetermined()
        ]
        counter = 1
        groups_to_merge = []  # A list of list of groups to merge
        while len(ind_to_resolve) > 0 and counter <= MAX_STEPS_RESOLVE_CELLS:
            for ind in ind_to_resolve:
                ijk = grid.get_ijk(active_index=ind)
                groups_nearby = self._find_nearest_groups(ijk, grid)
                if [-1] in groups_nearby:
                    groups_nearby = [x for x in groups_nearby if x != [-1]]
                if len(groups_nearby) == 1:
                    self.cells[ind].set_cell_groups(groups_nearby[0])
                elif len(groups_nearby) >= 2:
                    if groups_nearby not in groups_to_merge:
                        groups_to_merge.append(groups_nearby)
                    # Set to first group, but will be overwritten by merge later
                    self.cells[ind].set_cell_groups(groups_nearby[0])

            updated_ind_to_resolve = [
                ind for ind, group in enumerate(self.cells) if group.is_undetermined()
            ]
            if len(updated_ind_to_resolve) == len(ind_to_resolve):
                updated = False
                for ind in ind_to_resolve:
                    ijk = grid.get_ijk(active_index=ind)
                    # Wider search radius when looking for nearby groups
                    for tolerance in range(2, MAX_NEAREST_GROUPS_SEARCH_DISTANCE + 1):
                        groups_nearby = self._find_nearest_groups(
                            ijk, grid, tol=tolerance
                        )
                        if len(groups_nearby) >= 1:
                            self.cells[ind].set_cell_groups(groups_nearby[0])
                            updated = True
                            break
                if updated:
                    updated_ind_to_resolve = [
                        ind
                        for ind, group in enumerate(self.cells)
                        if group.is_undetermined()
                    ]
                    ind_to_resolve = updated_ind_to_resolve
                    counter += 1
                    continue
                else:
                    break
            ind_to_resolve = updated_ind_to_resolve
            counter += 1

        # Any unresolved grid cells?
        for ind in ind_to_resolve:
            self.cells[ind].set_cell_groups([-1])

        # Resolve groups to merge:
        new_groups_to_merge: List = []
        for g in groups_to_merge:
            merged = False
            for c in g:
                if merged:
                    continue
                # Is group c in a group that is already somewhere new_groups_to_merge?
                for d in new_groups_to_merge:
                    if c in d:
                        merged = True
                        # List of groups (g) needs to be merged with d
                        for new_g in g:
                            if new_g not in d:
                                d.append(new_g)
                        break
            if not merged:
                new_groups_to_merge.append(g)

        return new_groups_to_merge

    def _find_nearest_groups(self, ijk, grid, tol: int = 1) -> List[List[int]]:
        out = []
        (i1, j1, k1) = ijk
        neigs = list(
            itertools.product(
                range(max((i1 - tol), 0), min((i1 + tol), grid.get_nx() - 1) + 1),
                range(max((j1 - tol), 0), min((j1 + tol), grid.get_ny() - 1) + 1),
                range(max((k1 - tol), 0), min((k1 + tol), grid.get_nz() - 1) + 1),
            )
        )

        for ijk in neigs:
            ind = grid.get_active_index(ijk=ijk)
            if ind != -1 and self.cells[ind].has_co2():
                all_groups = self.cells[ind].all_groups
                if all_groups not in out:
                    out.append(all_groups.copy())
        return out

    def find_unique_groups(self):
        unique_groups = []
        for cell in self.cells:
            if cell.has_co2():
                if cell.all_groups not in unique_groups:
                    unique_groups.append(cell.all_groups)
            elif cell.is_undetermined() and [-1] not in unique_groups:
                unique_groups.append([-1])
        return unique_groups

    def check_if_well_is_part_of_larger_group(
        self, well_number: int
    ) -> Optional[List[int]]:
        for group in self.find_unique_groups():
            if len(group) > 1 and well_number in group:
                return group
        return None

    def debug_print(self):
        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.DEBUG):
            unique_groups = self.find_unique_groups()
            unique_groups.sort()
            logging.debug(
                f"Count '-'              : "
                f"{len([c for c in self.cells if c.has_no_co2()])}"
            )
            logging.debug(
                f"Count 'undetermined'   : "
                f"{len([c for c in self.cells if c.is_undetermined()])}"
            )
            for unique_group in unique_groups:
                n = len(
                    [
                        c
                        for c in self.cells
                        if c.has_co2() and c.all_groups == unique_group
                    ]
                )
                spaces = 10 - len(str(unique_group))
                logging.debug(f"Count '{unique_group}' {' ' * spaces}    : {n}")


def assemble_plume_groups_into_dict(plume_groups: List[str]) -> Dict[str, List[int]]:
    pg_dict: Dict[str, List[int]] = {}
    for ind, group in enumerate(plume_groups):
        if group != "":
            if group in pg_dict:
                pg_dict[group].append(ind)
            else:
                pg_dict[group] = [ind]
    return pg_dict


def _sort_well_names_in_merged_groups(name: str, inj_wells: List[InjectionWellData]):
    wells = name.split("+")
    if len(wells) > 1:
        sorted_wells = [well.name for well in inj_wells if well.name in wells]
        return "+".join(sorted_wells)
    return name


def sort_well_names(input_dict: Dict, inj_wells: List[InjectionWellData]):
    modified_dict = {
        _sort_well_names_in_merged_groups(name, inj_wells): value
        for name, value in input_dict.items()
    }
    cols = [c for c in modified_dict]
    sorted_cols = [well.name for well in inj_wells if well.name in cols]
    for col in cols:
        if col not in sorted_cols:
            sorted_cols.append(col)
    dict_sorted = {}
    for col in sorted_cols:
        dict_sorted[col] = input_dict[col]
    return dict_sorted
