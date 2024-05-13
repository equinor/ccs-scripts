# CO2 containment

## 🎯 Overview

Calculates the amount of CO2 inside or outside a boundary (polygons) per  zones or regions and returns it as a volume or mass. Multiple options are available.


## 📝 How to set it up?

### ERT

✅ Available on Komodo

⚪ Optional parameters: Boundary polygons / zone output file / regions - depending on which calculation method has been selected. For more information on how to output your boundary polygons , zone or region output file, read the "RMS" section below.

``` yaml title="Mandatory arguments"
<CASE>: Path to Eclipse case, including base name, but excluding the file extension (.EGRID, .INIT, .UNRST)

<CALC_TYPE_INPUT>: mass / cell_volume / actual_volume.
```
``` yaml title="Optional arguments"
<ROOT_DIR>: Path to root directory.

<OUT_DIR>: Path to output directory. By default, the filename is set to "plume_<calculation_type>.csv" and stored in <root_dir>/share/results/tables.

<CONTAINMENT_POLYGON>: Path to the containment polygon. Counts all CO2 as contains if not provided. 

<HAZARDOUS_POLYGON>: Path to the hazardous polygon.

<EGRID>: Path to EGRID file. Overwrites <case> if provided.

<UNRST>:  Path to UNRST file. Overwrites <case> if provided.

<INIT>:  Path to INIT file. Overwrites <case> if provided.

<ZONEFILE>: Path to file containing zone information. By default, does not calculates Co2 containment per zone.

<REGIONFILE>: Path to file containing Region information. By default, does not calculates Co2 containment per Region.

<REGION_PROPERTY>: Grid property FIPXXX used in simulation to define regions. By default, does not calculates Co2 containment per Region.

--verbose:  option outputs information about the calculations being done.
--debug: option outputs more information at each step of the calculation to help with debugging possible error messages. 
```

Example:

``` yaml title="Calculates actual volume of CO2 in the model"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <CALC_TYPE_INPUT>="actual_volume")
```


### RMS

RMS outputs some key information that the co2_containment script needs in order to function. Necesary jobs and output examples are provided below:

**Boundary polygons**

1. Create your polygons in the clipboard, under a `boundary_polygon` folder. 

2. To export your boundary via RMS, use the following python job:
``` yaml title="export_boundary_polygons.py"
###############################################################################
#export boundary polygons(csv-file) for use with webviz co2leakage plugin.
#
#Polygons have to be manually created by the user in a clipboard folder.
#Folder name and boundary name needs to be defined by the user.
#
#modified aug 2022 by sbern based on export_fault_polygons.py script made by rnyb (aug 2020) 
###############################################################################

import xtgeo
from pathlib import Path

# ------------------
# user input
# ------------------

# define polygon to export (in clipboard folder)
BOUNDARY_FOLDER = "boundary_polygon"
BOUNDARY_NAME = ["containment","hazardous"]

# ------------------
# functions
# ------------------
def _xtract_boundary_polygon(boundary_name, boundary_folder):
    mypoly = xtgeo.polygons_from_roxar(project, boundary_name, boundary_folder, 'clipboard')
    return mypoly.dataframe, mypoly


def export_boundary_polygon(boundary_name, boundary_folder):
    path = Path("../../share/results/polygons")
    csvname = ""+boundary_name+"--boundary.csv"

    path.mkdir(parents=True, exist_ok=True)
    csvfile = path / csvname

    # get polygon
    df_pol, pol = _xtract_boundary_polygon(boundary_name, boundary_folder)
    # export polygon
    df_pol.to_csv(csvfile, index=False)
    print("Created file {name}".format(name=csvfile))


# ------------------
# main
# ------------------

if __name__ == "__main__":
    for boundname in BOUNDARY_NAME:
        export_boundary_polygon(boundname, BOUNDARY_FOLDER)
    print("Done")

```

3. The output file should look like:
``` yaml title="Example of a boundary polygon exported by RMS"
X_UTME,Y_UTMN,Z_TVDSS,POLY_ID
565261.4599609375,6711611.201171875,1730.358154296875,0
557894.8095703125,6705717.87890625,1901.104248046875,0
564719.4873046875,6698037.1328125,1794.671630859375,0
572294.3525390625,6704132.8828125,1602.563232421875,0
565243.28515625,6711619.79296875,1730.311279296875,0
565261.4599609375,6711611.201171875,1730.358154296875,0
```

**Zone output file**

The following RMS job is used to export the zone information:

``` yaml title="export_geogrid_zone_layer_yml.py"
"""
    Create yml file with zone and layer info to be used with xtgeo qc scripts (GRID3D_HC_THICKNESS and GRID3D_AVG_MAP in ert model file)

    rnyb, Aug 20
"""
from pathlib import Path

GNAME = "Main_grid"
ZONE_YML_FILE = Path ("../../rms/output/zone/zonation_geo_map.yml")

# ------------------------------------------------------------------

def main ():
    """create yml file with zone-layer info """

    # Retrieve the grid to process:
    grid_model = project.grid_models [GNAME]
    grid = grid_model.get_grid()

    #Retrieve the zones to process:
    zone_names = grid.zone_names

    ZONE_YML_FILE.parent.mkdir(parents*=True, exist_ok=True)
    f = open(ZONE_YML_FILE, "w+)
    f.write("zranges:\n")

    # Loop over all zones in the grid
    for zone_index in range (len(zone_names)):

        # Get layers for zone_index
        layer_ranges = grid.grid_indexer.zonation[zone_index]
        # Convert range to list and add 1 (range starts at 0, we want to start at 1)

        layer_list = [1 + layer_ranges[0][0], 1 + layer_ranges[0][-1]]
        #Write result to file:
        f.write(" - " + zone_names[zone_index] + ": " + str(layer_list) + "\n")
    f.close()

if __name__ == "__main__":
    main()
    print("Done. Created xtgeo compatible yml file". ZONE_YML_FILE)
```

The output file should look like:

``` yaml title="Example of `zonation_geo_map.yml` exported by RMS"
zranges:
  - Zone1: [1, 24]
  - Zone2: [25, 54]
  - Zone3: [55, 60]
  - Zone4: [61, 61]


```

**Region output file**

The following RMS job is used to export the region information:

``` yaml title="export_geogrid_parameter.py" 
"""
    Export geogrid and geogrid parameters (roff format)
    Used for both visualization and as input to seismic project

    Parameters required/used by seismic projects is set in PROPS_SEISMIC
    Additional parameters can be set with PROPS_OTHER
    All parameters can be used for visualization (coviz)

    Authors:
    JRIV
    rnyb, Sep 20
"""
import xtgeo
from pathlib import Path, PurePath

PRJ = project

GNAME = "Main_grid"
PROPS_SEISMIC = ["PORO", "VSH", "SW"]
PROPS_OTHER =["PERMX", "Zone", "FIPSEG"] ##Any other region property of the form FIPXXX can be used.
TARGET = "../../share/results/grids"

def export_geogrid_parameters():
    
    """export geogrid and associated parameters based on user defined lists"""
    
    Path(TARGET).mkdir(parents=True, exists_ok=True)

    props = PROPS_SESIMIC + PROPS_OTHER ###can only have PROPS_OTHER

    print("Write grid to ", TARGET)
    grd = xtgeo.grid_from_roxar(PRJ,GNAME)
    grd.to_file(Path(TARGET).joinpath(GNAME.lower() + ".roff"))

    print("Write grid properties to ", TARGET)
    for propname in props:
        print(propname)
        prop = xtgeo.gridproperty_from_roxar(PRJ, GNAME, propname)
        filename = GNAME.lower() + "--" + propname.lower() + ".roff"
        prop.to_file(Path(TARGET).joinpath(filename))

if __name__ == "__main__":
    export_geogrid_parameters()
    print("Done.")
```



## 📚 Other examples

Note: rename `[1st_boundary_name]` and `[2nd_boundary_name]` according to your project's naming standards.

``` yaml title="Calculates cell volume with CO2 in the model"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <CALC_TYPE_INPUT>="cell_volume")
```
``` yaml title="Calculates mass of CO2 in the model"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <CALC_TYPE_INPUT>="mass")
```

``` yaml title="Calculates actual volume of CO2 inside & outside 2 polygons per zone"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <OUT_DIR>="share/results/tables/plume_actual_volume.csv", <CALC_TYPE_INPUT>="actual_volume", <CONTAINMENT_POLYGON>=share/results/polygons/[1st_boundary_name]--boundary.csv, <HAZARDOUS_POLYGON>=share/results/polygons/[2nd_boundary_name]--boundary.csv, <ZONEFILE>=<RUNPATH>rms/output/zone/zonation_geo_map.yml)
```


``` yaml title="Calculates cell volume inside & outside your 1 polygon with verbose and debug option"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <OUT_DIR>="share/results/tables/plume_cell_volume.csv", <CALC_TYPE_INPUT>="cell_volume", <CONTAINMENT_POLYGON>=share/results/polygons/[1st_boundary_name]--boundary.csv, <XARG1>= "--verbose", <XARG2>= "--debug")
```

``` yaml title="Calculates mass of CO2 inside & outside your 2 polygons and per zone"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <OUT_DIR>="share/results/tables/plume_mass.csv", <CALC_TYPE_INPUT>="mass", <CONTAINMENT_POLYGON>="share/results/polygons/[1st_boundary_name]--boundary.csv",  <HAZARDOUS_POLYGON>="share/results/polygons/[2nd_boundary_name]--boundary.csv", <ZONEFILE>="<RUNPATH>rms/output/zone/layer_zone_table.csv" )
```

``` yaml title="Calculates mass of CO2 inside & outside your 2 polygons and per region using region file"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <OUT_DIR>="share/results/tables/plume_mass.csv", <CALC_TYPE_INPUT>="mass", <CONTAINMENT_POLYGON>="share/results/polygons/[1st_boundary_name]--boundary.csv",  <HAZARDOUS_POLYGON>="share/results/polygons/[2nd_boundary_name]--boundary.csv", <REGIONFILE>="<RUNPATH>share/results/grids/main_grid--fipseg.roff" )
```

``` yaml title="Calculates cell volume inside & outside your 1 polygon per region using region property"
FORWARD_MODEL CO2_CONTAINMENT(<CASE>=<RUNPATH><ECLBASE>, <OUT_DIR>="share/results/tables/plume_cell_volume.csv", <CALC_TYPE_INPUT>="cell_volume", <CONTAINMENT_POLYGON>=share/results/polygons/[1st_boundary_name]--boundary.csv, <REGION_PROPERTY>="FIPSEG")
```


## 🔧 Versions & Updates

**Future development**

In progress:

- Include residual trapping analysis in containment calculations.
<br />
<br />

**Updates**
**May 2024:**

- This script now returns volume and mass of CO2 inside and outside  polygons per zone and region. 

- Added a "--verbose" option. Outputs all the calculation steps during the ERT run in the .STDERR file.

- Added a "--debug" option. Outputs all the calculation steps and extra information during the ERT run in the .STDERR file.