import shutil
from pathlib import Path

import pytest
import xtgeo

from ccs_scripts.aggregate import grid3d_migration_time


def test_migration_time1():
    result = Path(__file__).absolute().parent / "migration_time1_folder"
    result.mkdir(parents=True)
    grid3d_migration_time.main(
        [
            "--config",
            "tests/yaml/config_migration_time1.yml",
            "--mapfolder",
            str(result),
        ]
    )

    swat = xtgeo.surface_from_file(result / "all--migrationtime_swat.gri")
    assert swat.values.max() == pytest.approx(3.08767, abs=0.001)
    shutil.rmtree(str(Path(result)))


def test_migration_time2():
    result = Path(__file__).absolute().parent / "migration_time2_folder"
    result.mkdir(parents=True)
    grid3d_migration_time.main(
        [
            "--config",
            "tests/yaml/config_migration_time2.yml",
            "--mapfolder",
            str(result),
        ]
    )
    assert (result / "lower_zone--migrationtime_swat.gri").is_file()
    assert not (result / "all--migrationtime_swat.gri").is_file()
    shutil.rmtree(str(Path(result)))
