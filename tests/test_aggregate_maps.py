import os
import shutil
from pathlib import Path

import pytest
import xtgeo

from ccs_scripts.aggregate import grid3d_aggregate_map


def test_aggregated_map1():
    result = Path(__file__).absolute().parent / "aggregate1_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    cfg = "tests/yaml/config_aggregate1.yml"

    grid3d_aggregate_map.main(
        [
            "--config",
            cfg,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    swat = xtgeo.surface_from_file(result / "all--max_swat--20030101.gri")
    assert swat.values.min() == pytest.approx(0.14292679727077484, abs=1e-8)
    shutil.rmtree(str(Path(result)))


def test_aggregated_map2():
    result = Path(__file__).absolute().parent / "aggregate2_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    cfg = "tests/yaml/config_aggregate2.yml"

    grid3d_aggregate_map.main(
        [
            "--config",
            cfg,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    swat = xtgeo.surface_from_file(result / "all--min_swat--20030101.gri")
    assert swat.values.mean() == pytest.approx(0.7908786104444353, abs=1e-8)
    shutil.rmtree(str(Path(result)))


def test_aggregated_map3():
    result = Path(__file__).absolute().parent / "aggregate3_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    cfg = "tests/yaml/config_aggregate3.yml"

    grid3d_aggregate_map.main(
        [
            "--config",
            cfg,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    poro = xtgeo.surface_from_file(result / "all--mean_poro.gri")
    assert poro.values.mean() == pytest.approx(0.1677586422488292, abs=1e-8)
    shutil.rmtree(str(Path(result)))


def test_aggregated_map4():
    result = Path(__file__).absolute().parent / "aggregate4_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    yml = "tests/yaml/config_aggregate4.yml"

    grid3d_aggregate_map.main(
        [
            "--config",
            yml,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    swat = xtgeo.surface_from_file(result / "zone1--max_swat--20030101.gri")
    assert swat.values.max() == pytest.approx(1.0000962018966675, abs=1e-8)
    assert (result / "all--max_swat--20030101.gri").is_file()
    assert (result / "zone2--max_swat--20030101.gri").is_file()
    assert (result / "zone3--max_swat--20030101.gri").is_file()
    shutil.rmtree(str(Path(result)))


def test_aggregated_map5():
    result = Path(__file__).absolute().parent / "aggregate5_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    cfg = "tests/yaml/config_aggregate5.yml"

    grid3d_aggregate_map.main(
        [
            "--config",
            cfg,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    poro = xtgeo.surface_from_file(result / "all--mean_poro.gri")
    assert poro.values.mean() == pytest.approx(0.1648792893163274, abs=1e-5)
    shutil.rmtree(str(Path(result)))


def test_aggregated_map6():
    result = Path(__file__).absolute().parent / "aggregate6_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    cfg = "tests/yaml/config_aggregate6.yml"

    grid3d_aggregate_map.main(
        [
            "--config",
            cfg,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    gri_files = [p.stem for p in Path(result).glob("*.gri")]
    assert sorted(gri_files) == sorted(
        [
            "all--max_swat--19991201",
            "all--max_swat--20030101",
            "firstzone--max_swat--19991201",
            "firstzone--max_swat--20030101",
            "secondzone--max_swat--19991201",
            "secondzone--max_swat--20030101",
            "thirdzone--max_swat--19991201",
            "thirdzone--max_swat--20030101",
        ]
    )
    shutil.rmtree(str(Path(result)))


def test_aggregated_map7():
    result = Path(__file__).absolute().parent / "aggregate7_folder"
    if os.path.exists(result):
        shutil.rmtree(str(Path(result)))
    result.mkdir(parents=True)
    cfg = "tests/yaml/config_aggregate7.yml"

    grid3d_aggregate_map.main(
        [
            "--config",
            cfg,
            "--mapfolder",
            str(result),
            "--plotfolder",
            str(result),
        ]
    )
    gri_files = [p.stem for p in Path(result).glob("*.gri")]
    assert sorted(gri_files) == sorted(
        [
            "all--max_sgstrand--24000101",
            "all--max_sgstrand--25000101",
        ]
    )
    shutil.rmtree(str(Path(result)))
