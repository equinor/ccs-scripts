# Migration time maps

## 🎯 Overview

![image alt ><](mig-time.jpg)

## 📝 How to set it up?

### ERT


✅ Available on Komodo

```yaml
FORWARD_MODEL GRID3D_MIGRATION_TIME(<ECLROOT>=<ECLBASE>, <CONFIG_MIGTIME>=<CONFIG_PATH>/../input/config/grid3d_migration_time.yml)
```

### Config file

Config file are to be placed in `ert/input/config`

Template example:

~~~ yaml title="grid3d_migration_time.yml"
{% include "./config-file-examples/mig-time-template.yml" %}
~~~

## 📚 Other config file example



## 🔧 Versions & updates


**Future development**

- Extend the script to produce Migration time map for CO2 dissolved in water, in addition to SGAS. 

<br />
<br />

**Updates**