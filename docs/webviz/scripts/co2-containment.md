# CO2 containment

## Overview

Calculates the amount of CO2 inside or outside a boundary (polygons, zones or regions) and returns it as a volume or mass. Multiple options are available.


## How to set it up?

### ERT

✅ Available on Komodo

**Note:** Remember to change the sections marked with <span style="background-color: #DFF5FF">[ ]</span> below:





### Examples

``` yaml title="Calculates actual CO2 volume inside & outside your 2 polygons"
FORWARD_MODEL CO2_CONTAINMENT(<GRID>=<RUNPATH><ECLBASE>.EGRID, <OUTFILE>="share/results/tables/plume_actual_volume_test.csv", <CALC_TYPE_INPUT>="actual_volume", <XARG1>="--containment_polygon", <XARG2>=share/results/polygons/[1st_boundary_name]--boundary.csv, <XARG3>="--hazardous_polygon", <XARG4>=share/results/polygons/[2nd_boundary_name]--boundary.csv, <XARG5>="--zonefile", <XARG6>=<RUNPATH>rms/output/zone/layer_zone_table.csv)
```

``` yaml title="Calculates cell volume inside & outside your 1 polygon"
FORWARD_MODEL CO2_CONTAINMENT(<GRID>=<RUNPATH><ECLBASE>.EGRID, <OUTFILE>="share/results/tables/plume_cell_volume.csv", <CALC_TYPE_INPUT>="cell_volume",<XARG1>="--containment_polygon", <XARG2>=share/results/polygons/[1st_boundary_name]--boundary.csv)
```

``` yaml title="Calculates mass inside & outside your 2 polygons and per zone"
FORWARD_MODEL CO2_CONTAINMENT(<GRID>=<RUNPATH><ECLBASE>.EGRID, <OUTFILE>="share/results/tables/plume_mass.csv", <CALC_TYPE_INPUT>="mass", <XARG1>="--containment_polygon", <XARG2>="share/results/polygons/[1st_boundary_name]--boundary.csv", <XARG3>="--hazardous_polygon", <XARG4>="share/results/polygons/[2nd_boundary_name]--boundary.csv", <XARG5>="--zonefile", <XARG6>="<RUNPATH>rms/output/zone/layer_zone_table.csv" )
```




## Versions

**Future development**

<br />
<br />

**Updates**

- Date - Insert changes
