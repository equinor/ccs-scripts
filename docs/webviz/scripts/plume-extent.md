# Plume extent

## 🎯 Overview

Calculates the maximum distance of the CO2 plume to the injector or another point (as gas or CO2 dissolved in water).

**Limitations**

Only calculates the maximum distance from 1 injector. 


## 📝 How to set it up?

### ERT

✅ Available on Komodo


``` yaml
FORWARD_MODEL PLUME_EXTENT(<CASE>=<ECLBASE>, <XARG1>="--well_name", <XARG2>=[WELL_NAME])
```

## 📚 Other examples

``` yaml title="Calculates max extent for a well called S-J"
FORWARD_MODEL PLUME_EXTENT(<CASE>=<ECLBASE>, <XARG1>="--well_name", <XARG2>=S-J)
```

## 🔧 Versions & Updates

**Future development**

- Ensure 2+ injectors can be used. 

- Evaluate adding script extentsion from Smeaheia in this code.

![image alt ><](./img/plume_extent2.png)
<br />
<br />

**Updates**

- Date - Insert changes
