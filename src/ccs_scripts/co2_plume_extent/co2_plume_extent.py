#!/usr/bin/env python
"""
Calculates the plume extent from a given coordinate, or well point,
using SGAS and the dissolved property (AMFG/XMF2).
"""

import logging
import os
import sys

from ccs_scripts.co2_plume_extent.input import (
    init_timer,
    log_distance_calculation_configurations,
    log_input_configuration,
    make_parser,
    replace_default_dummies_from_ert,
    setup_log_configuration,
)
from ccs_scripts.co2_plume_extent.compute import calculate_distances
from ccs_scripts.co2_plume_extent.config import (
    Configuration,
    find_input_line,
    find_input_point,
)
from ccs_scripts.co2_plume_extent.output import (
    collect_results_into_dataframe,
    find_output_file,
    log_results,
    log_results_detailed,
)
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

    args = make_parser().parse_args()
    replace_default_dummies_from_ert(args)
    args.column_name = (
        args.column_name.upper() if args.column_name is not None else None
    )
    setup_log_configuration(args)
    log_input_configuration(args)

    config = Configuration(
        args.config_plume_extent,
        args.calc_type,
        args.inj_point,
        args.column_name,
        args.case,
    )
    log_distance_calculation_configurations(config)

    all_results = calculate_distances(
        args.case,
        config.distance_calculations,
        config.injection_wells,
        config.do_plume_tracking,
        args.threshold_gas,
        args.threshold_dissolved,
    )

    df = collect_results_into_dataframe(
        all_results,
        config,
        config.injection_wells,
    )
    log_results(df)
    log_results_detailed(df)

    timer.start("export_results")
    output_file = find_output_file(args.output_csv, args.case)
    logging.info("\nExport results to CSV file")
    logging.info(f"    - File path: {output_file}")
    if os.path.isfile(output_file):
        logging.info("Output CSV file already exists => Will overwrite existing file")
    df.to_csv(output_file, index=False, na_rep="0.0")
    timer.stop("export_results")

    timer.stop("total")
    timer.report()
    timer.stop("total")
    timer.report()

    return 0


if __name__ == "__main__":
    sys.exit(main())
