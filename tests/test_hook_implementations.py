import shutil

import pytest
from ert.plugins.plugin_manager import ErtPluginManager

from ccs_scripts.hook_implementations import forward_model_steps


@pytest.fixture
def expected_jobs() -> list[str]:
    return [
        "CO2_CONTAINMENT",
        "CO2_PLUME_AREA",
        "CO2_PLUME_EXTENT",
        "CO2_CSV_ARROW_CONVERTER",
        "GRID3D_CO2_MASS_MAP",
        "GRID3D_AGGREGATE_MAP",
        "GRID3D_MIGRATION_TIME",
    ]


# Avoid category inflation. Add to this list when it makes sense:
ACCEPTED_JOB_CATEGORIES = ["modelling", "utility"]


def test_hooks_are_installed_in_erts_plugin_manager(expected_jobs):
    """Test that we have the correct set of jobs installed,
    nothing more, nothing less"""
    plugin_m = ErtPluginManager(plugins=[forward_model_steps])
    available_fm_steps = [step().name for step in plugin_m.forward_model_steps]
    assert set(available_fm_steps) == set(expected_jobs)


def test_executables(expected_jobs):
    """Test executables listed in job configurations exist in $PATH"""
    for job in expected_jobs:
        assert shutil.which(job.lower())


def test_hook_implementations_job_docs():
    """For each installed job, we require the associated
    description string to be nonempty, and valid RST markup"""

    plugin_m = ErtPluginManager(plugins=[forward_model_steps])
    for step_doc in [step().documentation() for step in plugin_m.forward_model_steps]:
        assert step_doc.description
        assert step_doc.category
        assert step_doc.examples
