#!/usr/bin/env python
"""
Calculations for tracking the CO2 plumes from different injection wells,
using SGAS and the dissolved property (AMFG/XMF2).
Keeps track of which grid cells belong to which
plume group at each time step, and merges plumes if they meet.
"""

import logging
import os
import sys
import time

from ccs_scripts.co2_plume_tracking.compute import (
    load_data_and_calculate_plume_groups,
)
from ccs_scripts.co2_plume_tracking.input import process_input
from ccs_scripts.co2_plume_tracking.output import export_results
from ccs_scripts.utils.xtgeo_logging import setup_xtgeo_logging

setup_xtgeo_logging()

DESCRIPTION = """
Calculations for tracking the CO2 plumes from different injection wells,
using SGAS and the dissolved property (AMFS/AMFG/XMF2). Keeps track of
which grid cells belong to which plume group at each time step, and
merges plumes if they meet.

Output is a table on CSV format, counting the number of grid cells in
each group at each time step. The functionality is also used by the plume
extent script, to separate the results into different plume groups.
"""

CATEGORY = "modelling.reservoir"


def main():
    """
    Calculations for tracking plume groups.
    The method calculate_plume_groups() can be used by other scripts
    that want this functionality.
    Output from this script is a simple CSV-file counting the number of
    grid cells in each plume group for each time step.
    """
    time_start = time.time()
    args, config = process_input()

    (
        pg_prop_gas,
        pg_prop_dissolved,
        dissolved_prop_key,
        dates,
    ) = load_data_and_calculate_plume_groups(
        args.case,
        config.injection_wells,
        args.threshold_gas,
        args.threshold_dissolved,
    )

    export_results(
        args.output_csv,
        args.case,
        dates,
        pg_prop_gas,
        pg_prop_dissolved,
        dissolved_prop_key,
        config.injection_wells,
    )

    dt = time.time() - time_start
    logging.info(f"Total execution time for plume tracking script: {dt:.1f} s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
