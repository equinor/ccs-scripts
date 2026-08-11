#!/usr/bin/env python
"""
Calculates the plume extent from a given coordinate, or well point,
using SGAS and the dissolved property (AMFG/XMF2).
"""

import sys

from ccs_scripts.co2_plume_extent.compute import calculate_distances
from ccs_scripts.co2_plume_extent.config import (
    find_input_line,
    find_input_point,
)
from ccs_scripts.co2_plume_extent.input import (
    init_timer,
    process_input,
)
from ccs_scripts.co2_plume_extent.output import export_results
from ccs_scripts.utils.timer import Timer


def _find_input_point(*args, **kwargs):
    """Unused legacy code"""
    return find_input_point(*args, **kwargs)


def _find_input_line(*args, **kwargs):
    """Unused legacy code"""
    return find_input_line(*args, **kwargs)


def main():
    """
    Calculate plume extent or distance to point/line using EGRID and
    UNRST-files. Calculated for SGAS and AMFG/XMF2. Output is distance per
    date written to a CSV file.
    """
    init_timer()
    timer = Timer()
    timer.start("total")

    args, config = process_input()

    all_results = calculate_distances(
        args.case,
        config.distance_calculations,
        config.injection_wells,
        config.do_plume_tracking,
        args.threshold_gas,
        args.threshold_dissolved,
    )

    export_results(all_results, config, args.output_csv, args.case)

    timer.stop("total")
    timer.report()

    return 0


if __name__ == "__main__":
    sys.exit(main())
