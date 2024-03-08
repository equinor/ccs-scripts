# CO2 containment

## 🎯 Overview

Calculates the amount of CO2 inside or outside a boundary (polygons, zones or regions) and returns it as a volume or mass. Multiple options are available.


## 📝 How to set it up?

### ERT

✅ Available on Komodo

⚪ Optional parameters: Boundary polygons / zone output file / regions - depending on which calculation method has been selected. For more information on how to output your boundary polygons or zone output file, read the "RMS" section below.

``` yaml title="Calculates actual volume of CO2 in the model"
FORWARD_MODEL CO2_CONTAINMENT(<GRID>=<RUNPATH><ECLBASE>.EGRID, <OUTFILE>="share/results/tables/plume_actual_volume.csv", <CALC_TYPE_INPUT>="actual_volume")
```
``` yaml title="Calculates cell volume with CO2 in the model"
FORWARD_MODEL CO2_CONTAINMENT(<GRID>=<RUNPATH><ECLBASE>.EGRID, <OUTFILE>="share/results/tables/plume_cell_volume.csv", <CALC_TYPE_INPUT>="cell_volume")
```
``` yaml title="Calculates mass of CO2 in the model"
FORWARD_MODEL CO2_CONTAINMENT(<GRID>=<RUNPATH><ECLBASE>.EGRID, <OUTFILE>="share/results/tables/plume_mass.csv", <CALC_TYPE_INPUT>="mass")
```

### RMS

- Boundary polygons

*More to come*

- Zone output file

*More to come*

## 📚 Other examples

``` yaml title="Calculates actual volume of CO2 inside & outside 2 polygons"
FORWARD_MODEL CO2_CONTAINMENT(<GRID>=<RUNPATH><ECLBASE>.EGRID, <OUTFILE>="share/results/tables/plume_actual_volume.csv", <CALC_TYPE_INPUT>="actual_volume", <XARG1>="--containment_polygon", <XARG2>=share/results/polygons/[1st_boundary_name]--boundary.csv, <XARG3>="--hazardous_polygon", <XARG4>=share/results/polygons/[2nd_boundary_name]--boundary.csv, <XARG5>="--zonefile", <XARG6>=<RUNPATH>rms/output/zone/layer_zone_table.csv)
```

``` yaml title="Calculates cell volume inside & outside your 1 polygon"
FORWARD_MODEL CO2_CONTAINMENT(<GRID>=<RUNPATH><ECLBASE>.EGRID, <OUTFILE>="share/results/tables/plume_cell_volume.csv", <CALC_TYPE_INPUT>="cell_volume",<XARG1>="--containment_polygon", <XARG2>=share/results/polygons/[1st_boundary_name]--boundary.csv)
```

``` yaml title="Calculates mass of CO2 inside & outside your 2 polygons and per zone"
FORWARD_MODEL CO2_CONTAINMENT(<GRID>=<RUNPATH><ECLBASE>.EGRID, <OUTFILE>="share/results/tables/plume_mass.csv", <CALC_TYPE_INPUT>="mass", <XARG1>="--containment_polygon", <XARG2>="share/results/polygons/[1st_boundary_name]--boundary.csv", <XARG3>="--hazardous_polygon", <XARG4>="share/results/polygons/[2nd_boundary_name]--boundary.csv", <XARG5>="--zonefile", <XARG6>="<RUNPATH>rms/output/zone/layer_zone_table.csv" )
```




## 🔧 Versions & Updates

**Future development**
<br />
<br />

**Updates**

- Date - Insert changes
