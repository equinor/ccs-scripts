"""Thin wrapper around fmu-dataio ExportData for ccs-scripts outputs.

No fmu-dataio "Standard Result" schema exists for CCS/CO2 content yet, so this is
deliberately freeform ExportData usage (content="property"), not a custom
serializer - fmu-dataio itself already knows how to handle xtgeo.RegularSurface
(and, for future use, pandas.DataFrame and xtgeo.Grid/GridProperty) objects.
Extend _CONTENT_MAP when a new property is added; keep the content decision here
instead of scattering string checks elsewhere.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import xtgeo
from fmu.dataio import ExportData

WORKFLOW = "ccs-scripts"


@dataclass(frozen=True)
class ContentTag:
    content: str
    unit: str = ""


# Keyed by the property name ccs-scripts already uses internally (see
# aggregate._co2_mass.MapName for the co2_mass_* values). Matched by suffix
# against the property part of a surface name, since aggregation-method
# prefixes (e.g. "mean_", "max_") may precede it.
_CONTENT_MAP: Dict[str, ContentTag] = {
    "co2_mass_total": ContentTag("property", "kg"),
    "co2_mass_dissolved_water_phase": ContentTag("property", "kg"),
    "co2_mass_dissolved_oil_phase": ContentTag("property", "kg"),
    "co2_mass_gas_phase": ContentTag("property", "kg"),
    "co2_mass_trapped_gas_phase": ContentTag("property", "kg"),
    "co2_mass_free_gas_phase": ContentTag("property", "kg"),
    "co2_mass_migration_time_total": ContentTag("property"),
}


def _tag_for(property_key: str) -> ContentTag:
    for known_key, tag in _CONTENT_MAP.items():
        if property_key.endswith(known_key):
            return tag
    # Fall back to a generic property tag rather than hard-failing: aggregate
    # maps cover an open-ended set of grid properties (SWAT, PORO, ...) that
    # can't all be enumerated up front.
    return ContentTag("property")


def filter_property_and_date_from_surface_name(
    name: str,
) -> Tuple[Optional[str], str, Optional[str]]:
    """Split a "{filter}--{property}[--{date}]" surface name (see
    grid3d_aggregate_map._deduce_surface_name / _name_with_date) into the
    filter name (e.g. "all", or a zone name), the property key used for
    content lookup, and the date if present.
    """
    parts = name.split("--")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], None
    return None, parts[0], None


def export_surface(
    surface: xtgeo.RegularSurface,
    property_key: str,
    date: Optional[str] = None,
    filter_name: Optional[str] = None,
) -> str:
    tag = _tag_for(property_key)
    # "all" (no zone/region breakdown) isn't a meaningful tagname qualifier.
    tagname = filter_name if filter_name and filter_name != "all" else ""
    exporter = ExportData(
        content=tag.content,
        content_metadata={"attribute": property_key, "is_discrete": False},
        name=property_key.replace("_", "-"),
        tagname=tagname,
        unit=tag.unit,
        vertical_domain="depth",
        timedata=[[date]] if date else None,
        workflow=WORKFLOW,
    )
    return str(exporter.export(surface))
