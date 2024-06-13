
CO2_PLUME_AREA
==============

.. argparse::
   :module: ccs_scripts.co2_plume_area.co2_plume_area
   :func: get_parser
   :prog: plume_area

Calculates the area of the CO\ :sub:`2` plume for each formation and time step, for both SGAS and AMFG (Pflotran) / YMF2 (Eclipse).

Output is a table on CSV format.


CSV file example - plume area
-----------------------------
Example of how the plume area output CSV file is structured:

.. list-table:: CSV file of CO2 plume area (m^2)
   :widths: 25 25 25 25 25 25 25
   :header-rows: 1

   * - DATE
     - toptherys_SGAS
     - topvolantis_SGAS
     - topvolon_SGAS
     - toptherys_AMFG
     - topvolantis_AMFG
     - topvolon_AMFG
   * - 2020-01-01
     - 0.0
     - 0.0
     - 0.0
     - 0.0
     - 0.0
     - 0.0
   * - 2060-01-01
     - 1200000.0
     - 300000.0
     - 100000.0
     - 1600000.0
     - 320000.0
     - 105000.0
   * - 2100-01-01
     - 2100000.0
     - 400000.0
     - 300000.0
     - 2900000.0
     - 510000.0
     - 360000.0
