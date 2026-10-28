import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from ccs_scripts.utils.utils import format_error, format_warning

_NUMBERED_XMF = re.compile(r"^XMF(\d+)$")
_NUMBERED_YMF = re.compile(r"^YMF(\d+)$")

# Used when the position of CO2 among numbered components is not known
DEFAULT_CO2_COMPONENT = 2
# Same tolerance as used by CO2 containment when inferring SOIL
OIL_SATURATION_TOLERANCE = 1e-6
# Dissolved properties tried, in this order, when the case type is not known.
DISSOLVED_FALLBACK_ORDER = ("AMFS", "AMFG", "XMF2")


class Scenario(Enum):
    """
    Which scenario is CO2 amounts calculated in
    """

    AQUIFER = 0
    DEPLETED_GAS_FIELD = 1
    DEPLETED_OIL_GAS_FIELD = 2


class Source(Enum):
    """
    Data source of the simulation case.
    """

    CIRRUS = "Cirrus"
    CIRRUS_COMP = "Cirrus COMP"
    ECLIPSE = "Eclipse"


@dataclass(frozen=True)
class CaseType:
    source: Source
    scenario: Scenario
    n_components: Optional[int] = None  # Only set for numbered components


def component_count(property_names: Iterable[str]) -> Optional[int]:
    """
    Detects how many components there are, from the numbered XMF and YMF
    properties (XMFn, YMFn). Named properties such as
    XMFS or YMFG are ignored.

    Args:
        property_names: Properties in the restart file. Duplicates are fine.

    Returns:
        The number of components, or None if there are no numbered XMF or no
        numbered YMF properties.
    """
    names = set(property_names)
    xmf = [int(m.group(1)) for name in names if (m := _NUMBERED_XMF.match(name))]
    ymf = [int(m.group(1)) for name in names if (m := _NUMBERED_YMF.match(name))]
    if not xmf or not ymf:
        return None
    if max(xmf) != max(ymf):
        error_text = (
            "Error: Number of components with XMF property differ from "
            "the number of components with YMF"
        )
        raise ValueError(format_error(error_text))
    return max(xmf)


def scenario_from_components(n_components: int, has_oil: bool) -> Scenario:
    """
    Scenario for a case with numbered components (Cirrus COMP or Eclipse).
    Three components or fewer is taken to be an aquifer.
    """
    if n_components <= 3:
        return Scenario.AQUIFER
    if has_oil:
        return Scenario.DEPLETED_OIL_GAS_FIELD
    return Scenario.DEPLETED_GAS_FIELD


def classify_case(
    property_names: Iterable[str],
    has_oil: Optional[bool] = None,
) -> Optional[CaseType]:
    """
    Classify a case by data source and scenario from its property names.

    Args:
        property_names: Properties in the restart file. Duplicates are fine.
        has_oil: Whether an oil phase is present. None if the decision is
            expected to be made based on the presence of SOIL. Explicit
            when SOIL has to be inferred from SGAS and SWAT.

    Returns:
        The case type, or None if the case cannot be classified.
    """
    names = set(property_names)
    if has_oil is None:
        has_oil = "SOIL" in names

    # Cirrus realizations: AMFS must be checked before AMFG.
    if "AMFS" in names:
        if "YMFO" in names:
            return CaseType(Source.CIRRUS, Scenario.DEPLETED_OIL_GAS_FIELD)
        return CaseType(Source.CIRRUS, Scenario.DEPLETED_GAS_FIELD)
    if {"AMFG", "YMFG"}.issubset(names):
        return CaseType(Source.CIRRUS, Scenario.AQUIFER)

    # Numbered components: Cirrus COMP or Eclipse
    n_components = component_count(names)
    if n_components is None:
        return None
    if "DGAS" in names:
        source = Source.CIRRUS_COMP
    elif "BGAS" in names:
        source = Source.ECLIPSE
    else:
        return None
    return CaseType(
        source, scenario_from_components(n_components, has_oil), n_components
    )


def oil_is_present(
    sgas: Iterable[np.ndarray],
    swat: Iterable[np.ndarray],
    tolerance: float = OIL_SATURATION_TOLERANCE,
) -> bool:
    """
    Whether SGAS + SWAT is below 1 somewhere, meaning there is an oil saturation.
    """
    return any(float(np.max(1.0 - g - w)) > tolerance for g, w in zip(sgas, swat))


@dataclass(frozen=True)
class Co2Proxies:
    """
    Properties used as proxies for CO2 in the gas phase and dissolved in water,
    with the reasons for the choice.
    """

    gas: str
    dissolved: Optional[str]  # CO2 in the aqueous phase
    gas_reason: str
    dissolved_reason: str
    case_type: Optional[CaseType] = None
    oil: Optional[str] = None  # CO2 in the oil phase, only when there is oil
    oil_reason: str = ""
    # Set when the aqueous proxy is not written, but can be derived by a mole
    # balance from this property (ZMF). See ccs_scripts.utils.aqueous_proxy.
    dissolved_derived_from: Optional[str] = None

    @property
    def gas_is_mole_fraction(self) -> bool:
        """True when the gas proxy is a mole fraction rather than SGAS"""
        return self.gas != "SGAS"

    def log(self) -> None:
        col = 24
        if self.case_type is None:
            logging.warning(
                format_warning(
                    f"{'Case type':<{col}}: could not be determined from the "
                    "properties, the proxies are chosen as before"
                )
            )
        else:
            case_text = f"{self.case_type.source.value}, {self.case_type.scenario.name}"
            if self.case_type.n_components is not None:
                case_text += f" ({self.case_type.n_components} components)"
            logging.info(f"{'Case type':<{col}}: {case_text}")
        logging.info(f"{'CO2 in gas phase':<{col}}: {self.gas} ({self.gas_reason})")
        label = "CO2 in aqueous phase"
        if self.dissolved is not None:
            logging.info(f"{label:<{col}}: {self.dissolved} ({self.dissolved_reason})")
        else:
            logging.warning(
                format_warning(
                    f"{label:<{col}}: not available ({self.dissolved_reason})"
                )
            )
        label = "CO2 in oil phase"
        if self.oil is not None:
            logging.info(f"{label:<{col}}: {self.oil} ({self.oil_reason})")
        elif (
            self.case_type is not None
            and self.case_type.scenario == Scenario.DEPLETED_OIL_GAS_FIELD
        ):
            logging.warning(
                format_warning(f"{label:<{col}}: not available ({self.oil_reason})")
            )
        else:
            state = "not determined" if self.case_type is None else "not applicable"
            logging.info(f"{label:<{col}}: {state} ({self.oil_reason})")


def select_co2_proxies(
    property_names: Iterable[str],
    has_oil: Optional[bool] = None,
    co2_component: Optional[int] = None,
) -> Co2Proxies:
    """
    Choose the properties that represent CO2 in the gas phase, in the aqueous
    phase (dissolved in water) and in the oil phase, based on the source and
    scenario of the case:

                       Aquifer        Depleted gas   Depleted oil & gas
        Cirrus         SGAS / AMFG    YMFS / AMFS    YMFS / AMFS / XMFS
        Cirrus COMP    SGAS / XMF2    YMF2 / AMF2    YMF2 / AMF2 / XMF2
        Eclipse        SGAS / XMF2    YMF2 / XMF2    YMF2 / AMF2 / XMF2

    With oil, XMF is the oil phase, so the aqueous phase is AMF. In Cirrus COMP it is
    AMF also without oil: water is not part of the flash, and XMF repeats YMF where
    there is only one hydrocarbon phase, so XMF is not the aqueous phase. If AMF is
    not written (Cirrus writes it in MODE COMP3P), it is derived from the overall
    composition ZMF by a mole balance, when the properties for that are there
    (dissolved_derived_from is then set). The derivation itself needs values, see
    ccs_scripts.utils.aqueous_proxy.

    Args:
        property_names: Properties in the restart file.
        has_oil: See classify_case().
        co2_component: Position of CO2 among numbered components. Defaults to 2.
    """
    names = set(property_names)
    case = classify_case(names, has_oil=has_oil)
    if case is None:
        legacy = next((p for p in DISSOLVED_FALLBACK_ORDER if p in names), None)
        return Co2Proxies(
            gas="SGAS",
            dissolved=legacy,
            gas_reason="case type unknown, SGAS used as before",
            dissolved_reason=(
                "case type unknown, first found of AMFS, AMFG and XMF2"
                if legacy is not None
                else "none of AMFS, AMFG or XMF2 found"
            ),
            oil_reason="case type unknown",
        )

    aquifer = case.scenario == Scenario.AQUIFER
    with_oil = case.scenario == Scenario.DEPLETED_OIL_GAS_FIELD
    gas_reason = (
        "aquifer: the gas phase is taken to be CO2"
        if aquifer
        else "SGAS not used: the gas phase also contains hydrocarbons"
    )
    dissolved_reason = "CO2 dissolved in water"
    oil: Optional[str] = None
    oil_reason = "no oil phase"
    if case.source == Source.CIRRUS:
        gas = "SGAS" if aquifer else "YMFS"
        dissolved: Optional[str] = "AMFG" if aquifer else "AMFS"
        if with_oil:
            oil = "XMFS"
            oil_reason = "mole fraction of CO2 in the oil phase"
    else:
        component = co2_component or DEFAULT_CO2_COMPONENT
        gas = "SGAS" if aquifer else f"YMF{component}"
        if with_oil:
            # With oil, XMF is the oil phase, and the aqueous phase has its own
            # property, AMF
            dissolved = f"AMF{component}"
            dissolved_reason = "mole fraction of CO2 in the aqueous phase"
            oil = f"XMF{component}"
            oil_reason = "mole fraction of CO2 in the oil phase"
        elif case.source == Source.CIRRUS_COMP and not aquifer:
            # Depleted gas in MODE COMP: XMF is not the aqueous phase, see above
            dissolved = f"AMF{component}"
            dissolved_reason = "mole fraction of CO2 in the aqueous phase"
        else:
            dissolved = f"XMF{component}"
        if case.source == Source.CIRRUS_COMP and co2_component is None:
            note = f"; CO2 assumed to be component {component}, no Cirrus info file"
            dissolved_reason += note
            if not aquifer:
                gas_reason += note
            if oil is not None:
                oil_reason += note

    derived_from: Optional[str] = None
    if gas not in names:
        gas_reason = f"{gas} not found, SGAS used instead (includes hydrocarbon gas)"
        gas = "SGAS"
    # Depleted gas in Cirrus COMP: as with oil, the aqueous phase has no property of
    # its own that can be used, XMF only repeats YMF
    comp_gas = (
        case.source == Source.CIRRUS_COMP
        and case.scenario == Scenario.DEPLETED_GAS_FIELD
    )
    if dissolved is not None and dissolved not in names:
        if (with_oil and case.source != Source.CIRRUS) or comp_gas:
            component = co2_component or DEFAULT_CO2_COMPONENT
            from_mass = case.source == Source.CIRRUS_COMP
            # Cirrus COMP writes mass densities: the molecular weights in the
            # Cirrus info file are needed to get to moles. The info file is also
            # what gives co2_component
            if not from_mass:
                densities = ["BGAS", "BWAT", "BOIL"]
            elif comp_gas:
                densities = ["DGAS", "DWAT"]  # No oil phase
            else:
                densities = ["DGAS", "DWAT", "DOIL"]
            needed = [f"ZMF{component}", f"YMF{component}"]
            if not comp_gas:  # Without oil, the balance does not need XMF
                needed.append(f"XMF{component}")
            needed.extend(["SGAS", "SWAT", *densities])
            missing = [name for name in needed if name not in names]
            needs_info = from_mass and co2_component is None
            if needs_info:
                missing.append("the Cirrus info file (molecular weights)")
            if not missing:
                # The name is kept: the values are derived when the data is read
                derived_from = f"ZMF{component}"
                dissolved_reason = (
                    f"{dissolved} is not written, derived from {derived_from} "
                    "by a mole balance"
                )
            else:
                if comp_gas:
                    why = (
                        f"{dissolved} is not written, and XMF is not the aqueous "
                        f"phase in MODE COMP"
                    )
                    hint = "ZMF and the Cirrus info file are needed"
                else:
                    why = f"{dissolved} is not written, and XMF is the oil phase here"
                    hint = "Add AMF to RPTRST"
                dissolved_reason = (
                    f"{why}. It cannot be derived either, {', '.join(missing)} "
                    f"not found. {hint}"
                )
                dissolved = None
        else:
            dissolved_reason = f"{dissolved} not found"
            dissolved = None
    if oil is not None and oil not in names:
        oil_reason = f"{oil} not found"
        oil = None
    return Co2Proxies(
        gas,
        dissolved,
        gas_reason,
        dissolved_reason,
        case,
        oil,
        oil_reason,
        derived_from,
    )
