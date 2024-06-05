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
        self.all_groups = new_groups.copy()  # NBNB-AS: Problem with copying list?

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
        self.cells: list[CellGroup] = [CellGroup() for _ in range(0, number_of_grid_cells)]

    def copy(self):
        out = PlumeGroups(len(self.cells))
        out.cells = self.cells.copy()
        return out

    def resolve_undetermined_cells(self, grid: Grid):
        ind_to_resolve = [ind for ind, group in enumerate(self.cells) if group.is_undetermined()]
        counter = 1
        while len(ind_to_resolve) > 0 and counter <= 20:
            print(f"counter        : {counter}")
            print(f"left to resolve: {len(ind_to_resolve)}")
            for ind in ind_to_resolve:
                ijk = grid.get_ijk(active_index=ind)
                groups_nearby = self._find_nearest_groups(ijk, grid)
                if len(groups_nearby) == 1:
                    self.cells[ind].set_cell_groups([groups_nearby[0]])
                elif len(groups_nearby) >= 2:
                    if -1 in groups_nearby:
                        groups_nearby.remove(-1)
                        print("SKIP MERGE WITH U")
                    if len(groups_nearby) >= 2:
                        print("NEED TO MERGE")
                        print(groups_nearby)
                        exit()

            updated_ind_to_resolve = [
                ind for ind, group in enumerate(self.cells) if group.is_undetermined()
            ]
            if len(updated_ind_to_resolve) == len(ind_to_resolve):
                print("BREAK")
                break
            ind_to_resolve = updated_ind_to_resolve
            counter += 1

        # # Any unresolved grid cells?
        for ind in ind_to_resolve:
            self.cells[ind].set_cell_groups([-1])

    def _find_nearest_groups(self, ijk, grid) -> list[int]:
        out = set()
        (i1, j1, k1) = ijk
        for ind, group in enumerate(self.cells, 0):
            (i2, j2, k2) = grid.get_ijk(active_index=ind)
            if abs(i2 - i1) <= 1 and abs(j2 - j1) <= 1 and abs(k2 - k2) <= 1:
                # if not group.is_undetermined() and not group.has_no_co2():
                if group.has_co2():
                    out.union(set(group.all_groups))
        return list(out)
