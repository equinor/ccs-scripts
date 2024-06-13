from enum import Enum
from typing import Optional

from resdata.grid import Grid


class Status(Enum):
    UNDETERMINED = 0
    NO_CO2 = 1
    HAS_CO2 = 2


class CellGroup:
    def __init__(self, groups: Optional[list[int]] = None):
        if groups is not None:
            self.status: Status = Status.HAS_CO2
            self.all_groups: list[int] = groups.copy()
        else:
            self.status: Status = Status.NO_CO2
            self.all_groups: list[int] = []

    def set_cell_groups(self, new_groups: list[int]):
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
    def __init__(self, number_of_grid_cells: int):
        self.cells: list[CellGroup] = [
            CellGroup() for _ in range(0, number_of_grid_cells)
        ]

    def copy(self):
        out = PlumeGroups(len(self.cells))
        out.cells = self.cells.copy()
        return out

    def resolve_undetermined_cells(self, grid: Grid) -> list:
        ind_to_resolve = [
            ind for ind, group in enumerate(self.cells) if group.is_undetermined()
        ]
        counter = 1
        groups_to_merge = []
        while len(ind_to_resolve) > 0 and counter <= 20:
            # print(f"counter        : {counter}")
            # print(f"left to resolve: {len(ind_to_resolve)}")
            for ind in ind_to_resolve:
                ijk = grid.get_ijk(active_index=ind)
                groups_nearby = self._find_nearest_groups(ijk, grid)
                if -1 in groups_nearby:
                    groups_nearby.remove(-1)
                    print("----------------> SKIP MERGE WITH UNKNOWN GROUP")
                if len(groups_nearby) == 1:
                    self.cells[ind].set_cell_groups([groups_nearby[0]])
                elif len(groups_nearby) >= 2:
                    if len(groups_nearby) >= 2:
                        print("----------------> NEED TO MERGE")
                        print(groups_nearby)
                        if groups_nearby not in groups_to_merge:
                            groups_to_merge.append(groups_nearby)
                        self.cells[ind].set_cell_groups(groups_nearby)

            updated_ind_to_resolve = [
                ind for ind, group in enumerate(self.cells) if group.is_undetermined()
            ]
            if len(updated_ind_to_resolve) == len(ind_to_resolve):
                print("BREAK")
                break
            ind_to_resolve = updated_ind_to_resolve
            counter += 1

        # Any unresolved grid cells?
        for ind in ind_to_resolve:
            self.cells[ind].set_cell_groups([-1])

        # Resolve groups to merge:
        # print("\n\n--------")
        new_groups_to_merge = []
        # print(groups_to_merge)
        # groups_to_merge = [ [ [1,2],[3,4],[5] ]  , [[10],[11],[12]] , [ [3,4],[6],[7] ], [[8], [9]] ]
        # groups_to_merge = [ [ [1],[2],[3,4] ] ]
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

    def _find_nearest_groups(self, ijk, grid) -> list[int]:
        out = set()
        (i1, j1, k1) = ijk
        cells_with_co2 = [i for i in range(len(self.cells)) if self.cells[i].has_co2()]
        for ind in cells_with_co2:
            group = self.cells[ind]
            (i2, j2, k2) = grid.get_ijk(active_index=ind)
            if abs(i2 - i1) <= 1 and abs(j2 - j1) <= 1 and abs(k2 - k2) <= 1:
                out = out.union(set(group.all_groups))
        return list(out)

    def _find_unique_groups(self):
        unique_groups = []
        for cell in self.cells:
            if cell.has_co2():
                if cell.all_groups not in unique_groups:
                    unique_groups.append(cell.all_groups)
        return unique_groups

    def _temp_print(self):
        # Find the groups:
        unique_groups = []
        for cell in self.cells:
            if cell.has_co2():
                if cell.all_groups not in unique_groups:
                    unique_groups.append(cell.all_groups)
        unique_groups.sort()

        print(
            f"Count '-'              : {len([c for c in self.cells if c.has_no_co2()])}"
        )
        print(
            f"Count '?'              : {len([c for c in self.cells if c.is_undetermined()])}"
        )
        for unique_group in unique_groups:
            print(
                f"Count '{unique_group}' {' '*(10-len(str(unique_group)))}    : {len([c for c in self.cells if c.has_co2() and c.all_groups == unique_group])}"
            )
