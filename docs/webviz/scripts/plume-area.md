# Plume area

## 🎯 Overview

Calculates the area of the CO2 plume (as gas or CO2 dissolved in water).

**Limitations**

Only calculates the total area of the plume(s). 



## 📝 How to set it up?

### ERT

✅ Available on Komodo

🔺 Pre-requisite: Run `GRID3D_AGGREGATE_MAP` - [More information](https://fmu-for-ccs.radix.equinor.com/webviz/maps/agg-map/).

``` yaml
FORWARD_MODEL CO2_PLUME_AREA(<INPUT>=<RUNPATH>share/results/maps/, <XARG1>= "--verbose")
```

## 📚 Other examples

``` yaml
FORWARD_MODEL CO2_PLUME_AREA(<INPUT>=<RUNPATH>share/results/maps/, <XARG1>= "--debug")
```

## 🔧 Versions & Updates

**Future development**

- Ensure area distinction between the different injection sites, in addition to the total area. 

<br />
<br />

**Updates**

**May 2024:** 

- "--verbose" option. Outputs all the calculation steps during the ERT run in the .STDERR file.
- "--debug" option. Outputs all the calculation steps and extra information during the ERT run in the .STDERR file. 