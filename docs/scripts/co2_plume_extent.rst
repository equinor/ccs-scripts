
CO2_PLUME_EXTENT
================

.. argparse::
   :module: ccs_scripts.co2_plume_extent.co2_plume_extent
   :func: get_parser
   :prog: plume_extent

Calculates the maximum lateral distance of the CO\ :sub:`2` plume from a given location, for instance an injection point. The distance is calculated for each time step, for both SGAS and AMFG (Pflotran) / YMF2 (Eclipse).

Output is a table on CSV format.

CSV file example - plume extent
-------------------------------
Example of how the plume extent output CSV file is structured:

.. list-table:: CSV file of CO2 plume extent (m)
   :widths: 25 25 25
   :header-rows: 1

   * - DATE
     - MAX_DISTANCE_SGAS
     - MAX_DISTANCE_AMFG
   * - 2020-01-01
     - 0.0
     - 0.0
   * - 2060-01-01
     - 703.4
     - 761.5
   * - 2100-01-01
     - 1305.2
     - 1521.0
