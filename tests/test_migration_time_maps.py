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
            "--config_migtime",
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
            "--config_migtime",
            "tests/yaml/config_migration_time2.yml",
            "--mapfolder",
            str(result),
        ]
    )
    assert (result / "lower_zone--migrationtime_swat.gri").is_file()
    assert not (result / "all--migrationtime_swat.gri").is_file()
    shutil.rmtree(str(Path(result)))


def test_migration_time_first_injection_year():
    test_dir = Path(__file__).absolute().parent
    default_result = test_dir / "migration_time_default_folder"
    shifted_result = test_dir / "migration_time_first_injection_year_folder"
    shifted_config = test_dir / "config_migration_time_first_injection_year.yml"
    if default_result.exists():
        shutil.rmtree(str(Path(default_result)))
    if shifted_result.exists():
        shutil.rmtree(str(Path(shifted_result)))
    if shifted_config.exists():
        shifted_config.unlink()
    default_result.mkdir(parents=True)
    shifted_result.mkdir(parents=True)
    base_config = Path("tests/yaml/config_migration_time1.yml")
    shifted_config.write_text(
        base_config.read_text(encoding="utf-8")
        + "\n"
        + "migration_time_settings:\n"
        + "  first_injection_year: 2001\n",
        encoding="utf-8",
    )
    try:
        grid3d_migration_time.main(
            [
                "--config_migtime",
                str(base_config),
                "--mapfolder",
                str(default_result),
            ]
        )
        grid3d_migration_time.main(
            [
                "--config_migtime",
                str(shifted_config),
                "--mapfolder",
                str(shifted_result),
            ]
        )
        default_swat = xtgeo.surface_from_file(
            default_result / "all--migrationtime_swat.gri"
        )
        shifted_swat = xtgeo.surface_from_file(
            shifted_result / "all--migrationtime_swat.gri"
        )
        assert default_swat.values.max() == pytest.approx(3.08767, abs=0.001)
        assert shifted_swat.values.max() == pytest.approx(2.0, abs=0.001)
        assert shifted_swat.values.max() < default_swat.values.max()
    finally:
        if default_result.exists():
            shutil.rmtree(str(Path(default_result)))
        if shifted_result.exists():
            shutil.rmtree(str(Path(shifted_result)))
        if shifted_config.exists():
            shifted_config.unlink()
