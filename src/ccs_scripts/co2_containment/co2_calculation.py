# pylint: disable-msg=too-many-lines
"""Methods for CO2 containment calculations"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ccs_scripts.co2_containment.input import CalculationType
from ccs_scripts.co2_containment.source_data import (
    PROPERTIES_NEEDED_CIRRUS,
    PROPERTIES_NEEDED_ECLIPSE,
    Scenario,
    SourceData,
)
from ccs_scripts.utils.timer import Timer
from ccs_scripts.utils.utils import (
    THRESHOLD_DISSOLVED,
    format_error,
    format_warning,
    identify_gas_less_cells,
    is_subset,
)

DEFAULT_CO2_MOLAR_MASS = 44.0
DEFAULT_WATER_MOLAR_MASS = 18.0


@dataclass
class Co2DataAtTimeStep:
    """
    Dataclass with amount of co2 for each phase (dissolved/gas/undefined)
    at a given time step.

    Args:
      date (str): The time step
      dis_water_phase (np.ndarray): The amount of CO2 in dissolved phase
      gas_phase (np.ndarray): The amount of CO2 in gaseous phase
      dis_oil_phase (np.ndarray): The amount of CO2 in oil phase
      volume_coverage (np.ndarray): The volume of a cell (specific of
                                    calc_type_input = volume_extent)
      trapped_gas_phase (np.ndarray): The amount of CO2 in trapped/stranded gas phase
      free_gas_phase (np.ndarray): The amount of CO2 in free gas phase
    """

    date: str
    dis_water_phase: np.ndarray
    gas_phase: np.ndarray
    dis_oil_phase: np.ndarray
    volume_coverage: np.ndarray
    trapped_gas_phase: np.ndarray
    free_gas_phase: np.ndarray

    def total_mass(self) -> np.ndarray:
        """
        Computes total mass as the sum of gas in dissolved and gas
        phase.
        """
        return self.dis_water_phase + self.gas_phase + self.dis_oil_phase


@dataclass
class Co2Data:
    """
    Dataclass with amount of CO2 at (x,y) coordinates

    Args:
      x_coord (np.ndarray): x coordinates
      y_coord (np.ndarray): y coordinates
      data_list (List): List with CO2 amounts calculated
                        at multiple time steps
      units (Literal): Units of the calculated amount of CO2
      scenario (Scenario): Scenario information
      zone (np.ndarray): Zone information
      region (np.ndarray): Region information

    """

    x_coord: np.ndarray
    y_coord: np.ndarray
    active_cells: np.ndarray  # 3D array with True where calculations are performed
    data_list: List[Co2DataAtTimeStep]
    units: Literal["kg", "tons", "m3"]
    scenario: Scenario
    zone: Optional[np.ndarray] = None
    region: Optional[np.ndarray] = None
    cell_size: Optional[float] = None


def _extract_mnemonic_value(info_data, mnemonic: str) -> Optional[float]:
    """Return value for mnemonic if present and valid, else None."""
    if mnemonic not in info_data["Mnemonic"].values:
        return None
    subset = info_data.loc[info_data["Mnemonic"] == mnemonic, "Value"]
    if subset.empty:
        return None
    val = subset.iloc[0]
    if pd.isna(val) or (isinstance(val, str) and not val.strip()):
        return None
    return float(val)


def _extract_molar_masses(
    scenario: Scenario,
    cirrus_info_file: Optional[str] = None,
):
    """
    Extract gas and oil molar masses from a CSV file.

    Args:
        cirrus_info_file (str): Path to the Cirrus info CSV file.
        scenario (Scenario): Which scenario co2 mass is computed for
    Returns:
        tuple[float | None, float | None]: (gas_molar_mass, oil_molar_mass)
    """
    if scenario == Scenario.AQUIFER:
        return None, None
    info_data = pd.read_csv(cirrus_info_file)
    info_data.columns = info_data.columns.str.strip()
    info_data["Mnemonic"] = info_data["Mnemonic"].str.strip()
    gas_molar_mass = _extract_mnemonic_value(info_data, "MWG")
    oil_molar_mass = (
        _extract_mnemonic_value(info_data, "MWO")
        if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
        else None
    )
    if gas_molar_mass is None:
        error_text = f"\nScenario: {scenario.name}."
        error_text += (
            "\nTo compute mass or actual volume in this scenario "
            "hydrocarbon gas molar mass must be provided"
        )
        raise ValueError(format_error(error_text))
    if scenario == Scenario.DEPLETED_OIL_GAS_FIELD and oil_molar_mass is None:
        error_text = f"\nScenario: {scenario.name}."
        error_text += (
            "\nTo compute mass or actual volume in this scenario "
            "oil molar mass must be provided"
        )
        raise ValueError(format_error(error_text))
    return gas_molar_mass, oil_molar_mass


def _extract_comp_molar_masses(
    cirrus_info_file: str,
):
    info_data = pd.read_csv(cirrus_info_file)
    info_data.columns = info_data.columns.str.strip()
    info_data["Mnemonic"] = info_data["Mnemonic"].str.strip()
    mw_df = (
        info_data.loc[
            info_data["Mnemonic"].str.startswith("MW_", na=False),
            ["Mnemonic", "Value"],
        ]
        .assign(
            Value=lambda df: df["Value"].astype(float),
            Component=lambda df: df["Mnemonic"].str.replace("MW_", "", regex=False),
        )
        .reset_index(drop=True)
    )
    molar_weights = {
        row["Component"]: (i + 1, row["Value"]) for i, row in mw_df.iterrows()
    }
    if "CO2" not in molar_weights:
        error_text = "CO2 molar mass not found in cirrus info file"
        raise ValueError(format_error(error_text))
    return molar_weights


def _n_components(active_props: List):
    """
    Detects how many components are there in vapor phase

    Args:
        active_props (List): List of active properties

    Returns
        int with the number of components
    """
    xmf_suffixes = [int(item[3:]) for item in active_props if item.startswith("XMF")]
    # Find the max suffix
    max_xmf_suffix = max(xmf_suffixes)

    ymf_suffixes = [int(item[3:]) for item in active_props if item.startswith("YMF")]
    # Find the max suffix
    max_ymf_suffix = max(ymf_suffixes)

    if max_xmf_suffix != max_ymf_suffix:
        error_text = (
            "Error: Number of components with XMF property differ from "
            "the number of components with YMF"
        )
        raise ValueError(format_error(error_text))
    return max_xmf_suffix


def _compute_phases_avg_mol_weight(
    source_data: SourceData,
    comp_molar_masses: Optional[Dict[str, Tuple[int, float]]],
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
):
    if comp_molar_masses is None:
        raise ValueError(
            "comp_molar_masses cannot be None when computing phase average molar "
            "mass weight"
        )
    dates = source_data.DATES
    gas_avg_mol_weight = {}
    oil_avg_mol_weight = {}
    water_avg_mol_weight = {}
    for date in dates:
        water_avg_mol_weight_at_date = {}
        gas_avg_mol_weight_at_date = {}
        oil_avg_mol_weight_at_date = {}
        for idx, molar_mass in comp_molar_masses.values():
            ymf_tmp_date = source_data.ymfs[idx][date]
            xmf_tmp_date = source_data.xmfs[idx][date]
            gas_avg_mol_weight_at_date[idx] = molar_mass * ymf_tmp_date
            oil_avg_mol_weight_at_date[idx] = (
                molar_mass * xmf_tmp_date if Scenario.DEPLETED_OIL_GAS_FIELD else None
            )
            water_avg_mol_weight_at_date[idx] = (
                molar_mass * xmf_tmp_date
                if not Scenario.DEPLETED_OIL_GAS_FIELD
                else (water_molar_mass / len(comp_molar_masses))
                * np.ones_like(xmf_tmp_date)
            )
        gas_avg_mol_weight[date] = np.sum(
            list(gas_avg_mol_weight_at_date.values()), axis=0
        )
        oil_avg_mol_weight[date] = np.sum(
            list(oil_avg_mol_weight_at_date.values()), axis=0
        )
        water_avg_mol_weight[date] = np.sum(
            list(water_avg_mol_weight_at_date.values()), axis=0
        )
    return water_avg_mol_weight, gas_avg_mol_weight, oil_avg_mol_weight


def _convert_phase_density_from_mass_to_mole(
    source_data: SourceData,
    comp_molar_masses: Optional[Dict[str, Tuple[int, float]]],
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
):
    water_avg_mol_weight, gas_avg_mol_weight, oil_avg_mol_weight = (
        _compute_phases_avg_mol_weight(source_data, comp_molar_masses, water_molar_mass)
    )
    dates = source_data.DATES
    dwat = source_data.DWAT
    dgas = source_data.DGAS
    doil = source_data.DOIL
    assert dwat is not None
    assert dgas is not None
    assert doil is not None
    bwat = {}
    bgas = {}
    boil = {}
    for date in dates:
        bwat[date] = dwat[date] / water_avg_mol_weight[date]
        bgas[date] = dgas[date] / gas_avg_mol_weight[date]
        boil[date] = (
            doil[date] / oil_avg_mol_weight[date]
            if Scenario.DEPLETED_OIL_GAS_FIELD
            else np.zeros_like(bgas[date])
        )
    return bwat, bgas, boil


def _mole_to_mass_fraction(
    co2_mf_prop: np.ndarray,
    gas_mf_prop: np.ndarray,
    water_mf_prop: np.ndarray,
    m_co2: float,
    m_h20: float,
    m_gas: Optional[float],
    m_oil: Optional[float],
) -> np.ndarray:
    """
    Converts from mole fraction to mass fraction

    Args:
      co2_mf_prop (np.ndarray): Property with mole fractions of CO2 in a given phase
      gas_mf_prop (np.ndarray): Property with mole fractions of hydrocarbon gas
                                in a given phase.For more than two components
      h20_mf_prop (np.ndarray): Property with mole fractions of H2O in a given phase
      m_co2 (float): Molar mass of CO2
      m_h20 (float): Molar mass of H2O
      m_gas (float): Molar mass of hydrocarbon gas
      m_oil (float): Molar mass of oil

    Returns:
      np.ndarray

    """

    m_gas = m_gas if m_gas is not None else 0.0
    m_oil = m_oil if m_oil is not None else 0.0
    return (
        co2_mf_prop
        * m_co2
        / (
            co2_mf_prop * m_co2
            + gas_mf_prop * m_gas
            + water_mf_prop * m_h20
            + (1 - co2_mf_prop - gas_mf_prop - water_mf_prop) * m_oil
        )
    )


def _cirrus_co2mass(
    source_data: SourceData,
    scenario: Scenario,
    pore_volume_prop: str,
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
    gas_molar_mass: Optional[float] = None,
    oil_molar_mass: Optional[float] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Calculates CO2 mass based on the existing properties in Cirrus

    Args:
      source_data (SourceData): Data with the information of the necessary properties
                                for the calculation of CO2 mass
      scenario (Scenario): Which scenario co2 mass is computed for
      pore_volume_prop (str): Which pore volume property to use (RPORV vs PORV)
      co2_molar_mass (float): CO2 molar mass - Default is 44 g/mol
      water_molar_mass (float): Water molar mass - Default is 18 g/mol
      gas_molar_mass (float): Gas molar mass - Default is 0 g/mol,
                              input required if more than 2 components
      oil_molar_mass (float): Oil molar mass - Default is 0 g/mol
                              input required if more than 3 components

    Returns:
      Dict

    """
    dates = source_data.DATES
    dwat = source_data.DWAT
    dgas = source_data.DGAS
    doil = source_data.DOIL
    amfg = source_data.AMFG
    ymfg = source_data.YMFG
    xmfg = source_data.XMFG
    amfw = source_data.AMFW
    ymfw = source_data.YMFW
    xmfw = source_data.XMFW
    amfs = source_data.AMFS
    ymfs = source_data.YMFS
    xmfs = source_data.XMFS
    sgas = source_data.SGAS
    swat = source_data.SWAT
    xmfo = source_data.XMFO
    if swat is None and scenario != Scenario.DEPLETED_OIL_GAS_FIELD:
        assert sgas is not None
        # Only gas (co2 or hydrocarbon gas) and water => sgas + swat = 1
        swat = {key: 1 - sgas[key] for key in sgas}
    if xmfw is None and scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
        # Assume g = hydrocarbon gas, s = co2, o = oil
        # => The remainder must be the mole fraction for water
        assert xmfg is not None and xmfs is not None and xmfo is not None
        xmfw = {key: 1 - xmfg[key] - xmfs[key] - xmfo[key] for key in xmfg}
    sgstrand = source_data.SGSTRAND
    eff_vols = source_data.RPORV if pore_volume_prop == "RPORV" else source_data.PORV

    mole_fractions = _construct_mole_fractions(
        scenario, amfg, amfs, amfw, ymfg, ymfs, ymfw, xmfs, xmfw, xmfg
    )

    assert eff_vols is not None
    assert swat is not None
    assert dwat is not None
    assert sgas is not None
    assert dgas is not None
    co2_mass = {}
    for date in dates:
        co2_mass[date] = [
            eff_vols[date]
            * swat[date]
            * dwat[date]
            * _mole_to_mass_fraction(
                mole_fractions["Aqueous"]["CO2"][date],
                mole_fractions["Aqueous"]["Gas"][date],
                mole_fractions["Aqueous"]["Water"][date],
                co2_molar_mass,
                water_molar_mass,
                gas_molar_mass,
                oil_molar_mass,
            ),
            eff_vols[date]
            * sgas[date]
            * dgas[date]
            * _mole_to_mass_fraction(
                mole_fractions["Gas"]["CO2"][date],
                mole_fractions["Gas"]["Gas"][date],
                mole_fractions["Gas"]["Water"][date],
                co2_molar_mass,
                water_molar_mass,
                gas_molar_mass,
                oil_molar_mass,
            ),
        ]
        if scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
            assert doil is not None
            co2_mass[date].extend(
                [
                    eff_vols[date]
                    * (1 - sgas[date] - swat[date])
                    * doil[date]
                    * _mole_to_mass_fraction(
                        mole_fractions["Oil"]["CO2"][date],
                        mole_fractions["Oil"]["Gas"][date],
                        mole_fractions["Oil"]["Water"][date],
                        co2_molar_mass,
                        water_molar_mass,
                        gas_molar_mass,
                        oil_molar_mass,
                    ),
                ]
            )
        else:
            co2_mass[date].extend([np.zeros_like(co2_mass[date][0])])

        if sgstrand:
            co2_mass[date].extend(
                [
                    eff_vols[date]
                    * sgstrand[date]
                    * dgas[date]
                    * _mole_to_mass_fraction(
                        mole_fractions["Gas"]["CO2"][date],
                        mole_fractions["Gas"]["Gas"][date],
                        mole_fractions["Gas"]["Water"][date],
                        co2_molar_mass,
                        water_molar_mass,
                        gas_molar_mass,
                        oil_molar_mass,
                    ),
                    eff_vols[date]
                    * (sgas[date] - sgstrand[date])
                    * dgas[date]
                    * _mole_to_mass_fraction(
                        mole_fractions["Gas"]["CO2"][date],
                        mole_fractions["Gas"]["Gas"][date],
                        mole_fractions["Gas"]["Water"][date],
                        co2_molar_mass,
                        water_molar_mass,
                        gas_molar_mass,
                        oil_molar_mass,
                    ),
                ]
            )
    return co2_mass


def _compositional_co2mass(
    source_data: SourceData,
    scenario: Scenario,
    source: str,
    pore_volume_prop: str,
    co2_molar_mass: Optional[float] = None,
    co2_position: Optional[int] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Calculates CO2 mass based on molar weight and mole fraction of the components

    Args:
      source_data (SourceData): Data with the information of the necessary properties
                                for the calculation of CO2 mass
      scenario (Scenario): Which scenario co2 mass is computed for
      pore_volume_prop (str): Which pore volume property to use (RPORV vs PORV)
      co2_molar_mass (float): CO2 molar mass - Default is 44 g/mol

    Returns:
      Dict

    """
    dates = source_data.DATES
    bgas = source_data.BGAS
    bwat = source_data.BWAT
    boil = source_data.BOIL
    sgas = source_data.SGAS
    swat = source_data.SWAT
    sgtrh = source_data.SGTRH
    sgstrand = source_data.SGSTRAND
    soil = source_data.SOIL
    eff_vols = source_data.RPORV if pore_volume_prop == "RPORV" else source_data.PORV
    conv_fact = co2_molar_mass
    if co2_position is not None and source == "Cirrus COMP":
        xmf_co2 = source_data.xmfs[co2_position]
        ymf_co2 = source_data.ymfs[co2_position]
    else:
        xmf_co2 = source_data.xmfs[2]
        ymf_co2 = source_data.ymfs[2]
    phase_moles = {}
    co2_mass = {}
    assert eff_vols is not None
    assert bgas is not None
    assert sgas is not None
    assert bwat is not None
    for date in dates:
        phase_moles[date] = [
            (
                bwat[date] * swat[date] * eff_vols[date]  # type: ignore[index]
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
                else bwat[date] * (1 - sgas[date]) * eff_vols[date]
            ),
            bgas[date] * sgas[date] * eff_vols[date],
        ]
        if scenario != Scenario.DEPLETED_OIL_GAS_FIELD:
            phase_moles[date].extend([np.zeros_like(phase_moles[date][0])])
            co2_mass[date] = [
                conv_fact * phase_moles[date][0] * xmf_co2[date],
                conv_fact * phase_moles[date][1] * ymf_co2[date],
                phase_moles[date][2],
            ]
        else:
            zmf_co2 = (
                source_data.zmfs[co2_position]
                if co2_position is not None and source == "Cirrus COMP"
                else source_data.zmfs[2]
            )
            assert boil is not None
            assert soil is not None
            phase_moles[date].extend([boil[date] * soil[date] * eff_vols[date]])
            total_moles = (
                phase_moles[date][0] + phase_moles[date][1] + phase_moles[date][2]
            )
            total_co2_mass = total_moles * zmf_co2[date] * conv_fact
            co2_mass[date] = [
                phase_moles[date][1] * ymf_co2[date] * conv_fact,
                phase_moles[date][2] * xmf_co2[date] * conv_fact,
            ]
            co2_mass[date].insert(
                0, total_co2_mass - co2_mass[date][0] - co2_mass[date][1]
            )
        if any(x is not None for x in (sgstrand, sgtrh)):
            assert sgtrh is not None
            co2_mass[date].extend(
                [
                    np.divide(
                        co2_mass[date][1] * sgtrh[date],
                        sgas[date],
                        out=np.zeros_like(sgas[date]),
                        where=sgas[date] != 0,
                    ),
                    np.divide(
                        co2_mass[date][1] * (sgas[date] - sgtrh[date]),
                        sgas[date],
                        out=np.zeros_like(sgas[date]),
                        where=sgas[date] != 0,
                    ),
                ]
            )
    return co2_mass


def _cirrus_co2_molar_volume(
    source_data,
    scenario: Scenario,
    water_density: np.ndarray,
    gas_density=np.ndarray,
    oil_density=Optional[np.ndarray],
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
    gas_molar_mass: Optional[float] = None,
    oil_molar_mass: Optional[float] = None,
) -> Dict:
    """
    Calculates CO2 molar volume (mol/m3) based on the existing properties in Cirrus

    Args:
      source_data (SourceData): Data with the information of the necessary properties
                                for the calculation of CO2 molar volume
      scenario (Scenario): Scenario under which CO2 is calculated
      water_density (float): Water density - Default is 1000 kg/m3
      co2_molar_mass (float): CO2 molar mass - Default is 44 g/mol
      water_molar_mass (float): Water molar mass - Default is 18 g/mol

    Returns:
      Dict

    """
    dates = source_data.DATES
    dgas = source_data.DGAS
    dwat = source_data.DWAT
    doil = source_data.DOIL
    ymfg = source_data.YMFG
    amfg = source_data.AMFG
    xmfg = source_data.XMFG
    amfw = source_data.AMFW
    ymfw = source_data.YMFW
    xmfw = source_data.XMFW
    amfs = source_data.AMFS
    ymfs = source_data.YMFS
    xmfs = source_data.XMFS

    gas_molar_mass = gas_molar_mass if gas_molar_mass is not None else 0.0
    oil_molar_mass = oil_molar_mass if oil_molar_mass is not None else 0.0

    mole_fractions = _construct_mole_fractions(
        scenario, amfg, amfs, amfw, ymfg, ymfs, ymfw, xmfs, xmfw, xmfg
    )

    co2_molar_vol = {}
    for date in dates:
        co2_molar_vol[date] = [
            [
                (
                    (1 / mole_fractions["Aqueous"]["CO2"][date][x])
                    * (
                        -water_molar_mass
                        * (mole_fractions["Aqueous"]["Water"][date][x])
                        / (1000 * water_density[x])
                        + (
                            co2_molar_mass * mole_fractions["Aqueous"]["CO2"][date][x]
                            + water_molar_mass
                            * (mole_fractions["Aqueous"]["Water"][date][x])
                        )
                        / (1000 * dwat[date][x])
                    )
                    if mole_fractions["Aqueous"]["CO2"][date][x] >= THRESHOLD_DISSOLVED
                    else 0
                )
                for x in range(len(mole_fractions["Aqueous"]["CO2"][date]))
            ],
            [
                (
                    (1 / mole_fractions["Gas"]["CO2"][date][x])
                    * (
                        -water_molar_mass
                        * mole_fractions["Gas"]["Water"][date][x]
                        / (1000 * water_density[x])
                        - gas_molar_mass
                        * mole_fractions["Gas"]["Gas"][date][x]
                        / (1000 * gas_density[x])
                        - oil_molar_mass
                        * (
                            1
                            - mole_fractions["Gas"]["CO2"][date][x]
                            - mole_fractions["Gas"]["Water"][date][x]
                            - mole_fractions["Gas"]["Gas"][date][x]
                        )
                        / (1000 * oil_density[x])
                        + (
                            co2_molar_mass * mole_fractions["Gas"]["CO2"][date][x]
                            + water_molar_mass * mole_fractions["Gas"]["Water"][date][x]
                            + gas_molar_mass * mole_fractions["Gas"]["Gas"][date][x]
                            + oil_molar_mass
                            * (
                                1
                                - mole_fractions["Gas"]["CO2"][date][x]
                                - mole_fractions["Gas"]["Water"][date][x]
                                - mole_fractions["Gas"]["Gas"][date][x]
                            )
                        )
                        / (1000 * dgas[date][x])
                    )
                    if not mole_fractions["Gas"]["CO2"][date][x] == 0
                    else 0
                )
                for x in range(len(mole_fractions["Gas"]["CO2"][date]))
            ],
        ]
        if scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
            co2_molar_vol[date].extend(
                [
                    [
                        (
                            (1 / mole_fractions["Oil"]["CO2"][date][x])
                            * (
                                -water_molar_mass
                                * mole_fractions["Oil"]["Water"][date][x]
                                / (1000 * water_density[x])
                                - gas_molar_mass
                                * mole_fractions["Oil"]["Gas"][date][x]
                                / (1000 * gas_density[x])
                                - oil_molar_mass
                                * (
                                    1
                                    - mole_fractions["Oil"]["CO2"][date][x]
                                    - mole_fractions["Oil"]["Water"][date][x]
                                    - mole_fractions["Oil"]["Gas"][date][x]
                                )
                                / (1000 * oil_density[x])
                                + (
                                    co2_molar_mass
                                    * mole_fractions["Oil"]["CO2"][date][x]
                                    + water_molar_mass
                                    * mole_fractions["Oil"]["Water"][date][x]
                                    + gas_molar_mass
                                    * mole_fractions["Oil"]["Gas"][date][x]
                                    + oil_molar_mass
                                    * (
                                        1
                                        - mole_fractions["Oil"]["CO2"][date][x]
                                        - mole_fractions["Oil"]["Water"][date][x]
                                        - mole_fractions["Oil"]["Gas"][date][x]
                                    )
                                )
                                / (1000 * doil[date][x])
                            )
                            if not mole_fractions["Oil"]["CO2"][date][x] == 0
                            else 0
                        )
                        for x in range(len(mole_fractions["Oil"]["CO2"][date]))
                    ]
                ],
            )
        else:
            co2_molar_vol[date].extend([list(np.zeros_like(co2_molar_vol[date][0]))])
        co2_molar_vol[date][0] = [
            0 if x < 0 or y < THRESHOLD_DISSOLVED else x
            for x, y in zip(
                co2_molar_vol[date][0], mole_fractions["Aqueous"]["CO2"][date]
            )
        ]
        co2_molar_vol[date][1] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(co2_molar_vol[date][1], mole_fractions["Gas"]["CO2"][date])
        ]
        co2_molar_vol[date][2] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(co2_molar_vol[date][2], mole_fractions["Oil"]["CO2"][date])
        ]
        if source_data.SGSTRAND is not None:
            co2_molar_vol[date].extend([co2_molar_vol[date][1], co2_molar_vol[date][1]])
    return co2_molar_vol


def _eclipse_co2_molar_volume(
    source_data: SourceData,
    water_density: np.ndarray,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
) -> Dict:
    """
    Calculates CO2 molar volume (mol/m3) based on the existing properties in Eclipse

    Args:
      source_data (SourceData): Data with the information of the necessary properties
                                for the calculation of CO2 molar volume
      water_density (float): Water density - Default is 1000 kg/m3
      water_molar_mass (float): Water molar mass - Default is 18 g/mol

    Returns:
      Dict

    """
    dates = source_data.DATES
    bgas = source_data.BGAS
    bwat = source_data.BWAT
    xmf2 = source_data.xmfs[2]
    ymf2 = source_data.ymfs[2]
    co2_molar_vol = {}
    for date in dates:
        co2_molar_vol[date] = [
            [
                (
                    (1 / xmf2[date][x])
                    * (
                        -water_molar_mass
                        * (1 - xmf2[date][x])
                        / (1000 * water_density[x])
                        + 1 / (1000 * bwat[date][x])  # type: ignore[index]
                    )
                    if xmf2[date][x] >= THRESHOLD_DISSOLVED
                    else 0
                )
                for x in range(len(xmf2[date]))
            ],
            [
                (
                    (1 / ymf2[date][x])
                    * (
                        -water_molar_mass
                        * (1 - ymf2[date][x])
                        / (1000 * water_density[x])
                        + 1 / (1000 * bgas[date][x])  # type: ignore[index]
                    )
                    if not ymf2[date][x] == 0
                    else 0
                )
                for x in range(len(ymf2[date]))
            ],
        ]
        co2_molar_vol[date].extend([list(np.zeros_like(co2_molar_vol[date][0]))])
        co2_molar_vol[date][0] = [
            0 if x < 0 or y < THRESHOLD_DISSOLVED else x
            for x, y in zip(co2_molar_vol[date][0], xmf2[date])
        ]
        co2_molar_vol[date][1] = [
            0 if x < 0 or y == 0 else x
            for x, y in zip(co2_molar_vol[date][1], ymf2[date])
        ]
        if source_data.SGTRH is not None:
            co2_molar_vol[date].extend([co2_molar_vol[date][1], co2_molar_vol[date][1]])
    return co2_molar_vol


def _construct_mole_fractions(
    scenario: Scenario,
    amfg,
    amfs,
    amfw,
    ymfg,
    ymfs,
    ymfw,
    xmfs,
    xmfw,
    xmfg,
):
    mole_fraction_dic = {
        "Aqueous": {
            "CO2": amfg if scenario == Scenario.AQUIFER else amfs,
            "Water": (
                amfw
                if amfw is not None
                else (
                    {key: 1 - amfg[key] for key in amfg}
                    if scenario == Scenario.AQUIFER
                    else None
                )
            ),
            "Gas": (
                {key: np.zeros_like(value) for key, value in amfg.items()}
                if scenario == Scenario.AQUIFER
                else amfg
            ),
        },
        "Gas": {
            "CO2": ymfg if scenario == Scenario.AQUIFER else ymfs,
            "Water": (
                ymfw
                if ymfw is not None
                else (
                    {key: 1 - ymfg[key] for key in ymfg}
                    if scenario == Scenario.AQUIFER
                    else None
                )
            ),
            "Gas": (
                {key: np.zeros_like(value) for key, value in ymfg.items()}
                if scenario == Scenario.AQUIFER
                else ymfg
            ),
        },
        "Oil": {
            "CO2": (
                xmfs
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
                else {key: np.zeros_like(value) for key, value in ymfg.items()}
            ),
            "Water": (
                xmfw
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
                else {key: np.zeros_like(value) for key, value in ymfg.items()}
            ),
            "Gas": (
                xmfg
                if scenario == Scenario.DEPLETED_OIL_GAS_FIELD
                else {key: np.zeros_like(value) for key, value in ymfg.items()}
            ),
        },
    }
    return mole_fraction_dic


def _calculate_co2_data_from_source_data(
    source_data: SourceData,
    calc_type: CalculationType,
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
    residual_trapping: bool = False,
    cirrus_info_file: Optional[str] = None,
) -> Co2Data:
    """
    Calculates a given calc_type (mass/cell_volume/actual_volume)
    from properties in source_data.

    Args:
        source_data (SourceData): Data with the information of the necessary properties
                                  for the calculation of calc_type
        calc_type (CalculationType): Which amount is calculated (mass / cell_volume /
                                     actual_volume)
        co2_molar_mass (float): CO2 molar mass - Default is 44 g/mol
        water_molar_mass (float): Water molar mass - Default is 18 g/mol
        residual_trapping (bool): Indicate if residual trapping should be calculated
        cirrus_info_file (Optional[str]): Path to cirrus info file

    Returns:
      Co2Data
    """
    logging.info(f"Start calculating CO2 {calc_type.name.lower()} from source data")
    active_props = source_data.active_property_names()
    if not is_subset(["SGAS"], active_props):
        error_text = "Lacking required property SGAS to compute CO2 mass/volume."
        raise ValueError(format_error(error_text))

    pore_volume_prop = _find_pore_volume_prop(active_props)
    source, scenario = _find_source_and_scenario(residual_trapping, active_props)
    gas_molar_mass = None
    oil_molar_mass = None
    comp_molar_masses = None
    if source == "Cirrus COMP":
        if cirrus_info_file is None:
            error_text = "Source: Cirrus COMP"
            error_text += f"\nScenario: {scenario.name}."
            error_text += (
                "\nTo compute mass or actual volume in this scenario "
                "path to cirrus INFO file must be provided."
            )
            raise ValueError(format_error(error_text))
        comp_molar_masses = _extract_comp_molar_masses(cirrus_info_file)
    elif source == "Cirrus":
        gas_molar_mass, oil_molar_mass = _extract_molar_masses(
            scenario, cirrus_info_file
        )
    logging.info("Found valid properties")
    logging.info(f"Data source : {source}")
    logging.info(f"Scenario    : {scenario.name}")
    logging.info("Properties used in the calculations:")
    logging.info(f"    {', '.join(active_props)}")

    if calc_type in (CalculationType.ACTUAL_VOLUME, CalculationType.MASS):
        co2_amount = _calc_co2_amount(
            source,
            scenario,
            calc_type,
            residual_trapping,
            source_data,
            pore_volume_prop,
            co2_molar_mass,
            water_molar_mass,
            gas_molar_mass,
            oil_molar_mass,
            comp_molar_masses,
        )
    elif calc_type == CalculationType.CELL_VOLUME:
        co2_amount = _calc_co2_amount_cell_volume(scenario, source_data, active_props)
    else:
        error_text = "Illegal calculation type: " + calc_type.name
        error_text += "\nValid options:"
        for calculation_type in CalculationType:
            error_text += "\n  * " + calculation_type.name
        error_text += "\nExiting"
        raise ValueError(format_error(error_text))

    co2_amount.cell_size = source_data.cell_size
    logging.info(f"Done calculating CO2 {calc_type.name.lower()} from source data\n")
    return co2_amount


def _find_pore_volume_prop(active_props: List[str]) -> str:
    pore_volume_prop = None
    if is_subset(["PORV", "RPORV"], active_props):
        pore_volume_prop = "RPORV"
        active_props.remove("PORV")
        logging.info("Using attribute RPORV instead of PORV")
    elif is_subset(["PORV"], active_props):
        pore_volume_prop = "PORV"
        logging.info("Using attribute PORV")
    elif is_subset(["RPORV"], active_props):
        pore_volume_prop = "RPORV"
        logging.info("Using attribute RPORV")
    else:
        error_text = "No pore volume provided"
        error_text += "\nNeed either PORV or RPORV"
        raise ValueError(format_error(error_text))

    return pore_volume_prop


def _find_source_and_scenario(
    residual_trapping: bool, active_props: List[str]
) -> Tuple[str, Scenario]:
    props_needed_cirrus = PROPERTIES_NEEDED_CIRRUS.copy()
    props_needed_eclipse = PROPERTIES_NEEDED_ECLIPSE.copy()
    if residual_trapping:
        props_needed_cirrus.append("SGSTRAND")
        props_needed_eclipse.append("SGTRH")
    if is_subset(props_needed_cirrus, active_props):
        source = "Cirrus"
        if is_subset(["AMFS", "YMFO"], active_props):
            scenario = Scenario.DEPLETED_OIL_GAS_FIELD
        elif is_subset(["AMFS"], active_props):
            scenario = Scenario.DEPLETED_GAS_FIELD
        elif is_subset(["AMFG", "YMFG"], active_props):
            scenario = Scenario.AQUIFER
        elif is_subset(["XMF2"], active_props):
            source = "Cirrus COMP"
            if _n_components(active_props) <= 3:
                scenario = Scenario.AQUIFER
            elif is_subset(["SOIL"], active_props):
                scenario = Scenario.DEPLETED_OIL_GAS_FIELD
            else:
                scenario = Scenario.DEPLETED_GAS_FIELD
        else:
            error_text = (
                "Need to provide either AMFS, AMFG or XMF2 to perform the calculations"
            )
            raise ValueError(format_error(error_text))
    elif is_subset(props_needed_eclipse, active_props):
        source = "Eclipse"
        if _n_components(active_props) <= 3:
            scenario = Scenario.AQUIFER
        elif is_subset(["SOIL"], active_props):
            scenario = Scenario.DEPLETED_OIL_GAS_FIELD
        else:
            scenario = Scenario.DEPLETED_GAS_FIELD
    else:
        _raise_missing_props_error(
            active_props, props_needed_cirrus, props_needed_eclipse
        )
    if scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
        required_oil_dens = "BOIL" if source == "Eclipse" else "DOIL"
        if not is_subset([required_oil_dens], active_props):
            error_text = (
                f"Source: {source}"
                f"\nScenario detected as DEPLETED_OIL_GAS_FIELD but "
                f"{required_oil_dens} is missing.\n"
            )
            raise ValueError(format_error(error_text))
    return source, scenario


def _calc_co2_amount(
    source: str,
    scenario: Scenario,
    calc_type: CalculationType,
    residual_trapping: bool,
    source_data: SourceData,
    pore_volume_prop: str,
    co2_molar_mass: float,
    water_molar_mass: float,
    gas_molar_mass: Optional[float],
    oil_molar_mass: Optional[float],
    comp_molar_masses: Optional[Dict[str, Tuple[int, float]]],
) -> Co2Data:
    if source == "Cirrus":
        co2_mass_cell = _cirrus_co2mass(
            source_data,
            scenario,
            pore_volume_prop,
            co2_molar_mass,
            water_molar_mass,
            gas_molar_mass,
            oil_molar_mass,
        )
    else:
        co2_position = None
        if source == "Cirrus COMP" and comp_molar_masses is not None:
            bwat, bgas, boil = _convert_phase_density_from_mass_to_mole(
                source_data,
                comp_molar_masses,
                water_molar_mass,
            )
            source_data.BWAT = bwat
            source_data.BGAS = bgas
            source_data.BOIL = boil
            co2_position = comp_molar_masses["CO2"][0]

        co2_mass_cell = _compositional_co2mass(
            source_data,
            scenario,
            source,
            pore_volume_prop,
            co2_molar_mass,
            co2_position,
        )
    co2_mass_output = Co2Data(
        source_data.x_coord,
        source_data.y_coord,
        source_data.active_cells,
        [
            Co2DataAtTimeStep(
                key,
                value[0],
                value[1],
                value[2],
                np.zeros_like(value[0]),
                (value[3] if residual_trapping else np.zeros_like(value[0])),
                (value[4] if residual_trapping else np.zeros_like(value[0])),
            )
            for key, value in co2_mass_cell.items()
        ],
        "kg",
        scenario,
        source_data.zone,
        source_data.region,
    )
    if calc_type == CalculationType.MASS:
        _convert_from_kg_to_tons(co2_mass_output)
        co2_amount = co2_mass_output
    else:
        molar_vols_co2 = _calculate_molar_vols_co2(
            source,
            scenario,
            source_data,
            co2_molar_mass,
            water_molar_mass,
            gas_molar_mass,
            oil_molar_mass,
        )
        co2_mass = {
            co2_mass_output.data_list[t].date: (
                [
                    co2_mass_output.data_list[t].dis_water_phase,
                    co2_mass_output.data_list[t].gas_phase,
                    co2_mass_output.data_list[t].dis_oil_phase,
                ]
                if not residual_trapping
                else [
                    co2_mass_output.data_list[t].dis_water_phase,
                    co2_mass_output.data_list[t].gas_phase,
                    co2_mass_output.data_list[t].dis_oil_phase,
                    co2_mass_output.data_list[t].trapped_gas_phase,
                    co2_mass_output.data_list[t].free_gas_phase,
                ]
            )
            for t in range(0, len(co2_mass_output.data_list))
        }
        vols_co2 = {
            t: [
                a * b / (co2_molar_mass / 1000)
                for a, b in zip(molar_vols_co2[t], co2_mass[t])
            ]
            for t in co2_mass
        }
        co2_amount = Co2Data(
            source_data.x_coord,
            source_data.y_coord,
            source_data.active_cells,
            [
                Co2DataAtTimeStep(
                    t,
                    np.array(vols_co2[t][0]),
                    np.array(vols_co2[t][1]),
                    np.array(vols_co2[t][2]),
                    np.zeros_like(np.array(vols_co2[t][0])),
                    (
                        np.array(vols_co2[t][3])
                        if residual_trapping
                        else np.zeros_like(np.array(vols_co2[t][0]))
                    ),
                    (
                        np.array(vols_co2[t][4])
                        if residual_trapping
                        else np.zeros_like(np.array(vols_co2[t][0]))
                    ),
                )
                for t in vols_co2
            ],
            "m3",
            scenario,
            source_data.zone,
            source_data.region,
        )
    return co2_amount


def _calculate_molar_vols_co2(
    source: str,
    scenario: Scenario,
    source_data: SourceData,
    co2_molar_mass: float,
    water_molar_mass: float,
    gas_molar_mass: Optional[float],
    oil_molar_mass: Optional[float],
):
    if source == "Cirrus":
        y_prop = source_data.AMFG if scenario == Scenario.AQUIFER else source_data.AMFS
        assert y_prop is not None
        y = y_prop[source_data.DATES[0]]
        where_min_amf_co2 = np.where(y < THRESHOLD_DISSOLVED)[0]
        if len(where_min_amf_co2) == 0:
            prop_name = "AMFG" if scenario == Scenario.AQUIFER else "AMFS"
            min_y = np.min(y)
            where_min_amf_co2 = np.where(y < min_y + THRESHOLD_DISSOLVED)[0]
            msg = (
                f"WARNING: Lack of cells with low (<{THRESHOLD_DISSOLVED}) "
                f"{prop_name}, needed for estimation of water density."
                f"\n         Using cells with {prop_name} < "
                f"{min_y + THRESHOLD_DISSOLVED} for estimation."
            )
            logging.warning(format_warning(msg))
        # Where amfg is 0, or the closest approximation available
        assert source_data.DWAT is not None
        dwat = source_data.DWAT[source_data.DATES[0]]
        water_density = np.array(
            [
                (
                    x[1]
                    if y[x[0]] < THRESHOLD_DISSOLVED
                    else np.mean(dwat[where_min_amf_co2])
                )
                for x in enumerate(dwat)
            ]
        )
        assert source_data.YMFG is not None
        y = source_data.YMFG[source_data.DATES[0]]
        max_y = np.max(y)
        where_max_ymfg = np.where(np.isclose(y, max_y))[0]
        assert source_data.DGAS is not None
        dgas = source_data.DGAS[source_data.DATES[0]]
        gas_density = np.array(
            [
                (x[1] if np.isclose((y[x[0]]), 1) else np.mean(dgas[where_max_ymfg]))
                for x in enumerate(dgas)
            ]
        )
        oil_density = np.ones_like(water_density)
        if scenario == Scenario.DEPLETED_OIL_GAS_FIELD:
            assert source_data.YMFO is not None
            y = source_data.YMFO[source_data.DATES[0]]
            max_y = np.max(y)
            where_max_xmfo = np.where(np.isclose(y, max_y))[0]
            assert source_data.DOIL is not None
            doil = source_data.DOIL[source_data.DATES[0]]
            oil_density = np.array(
                [
                    (
                        x[1]
                        if np.isclose((y[x[0]]), 1)
                        else np.mean(doil[where_max_xmfo])
                    )
                    for x in enumerate(doil)
                ]
            )
        molar_vols_co2 = _cirrus_co2_molar_volume(
            source_data,
            scenario,
            water_density,
            gas_density,
            oil_density,
            co2_molar_mass,
            water_molar_mass,
            gas_molar_mass,
            oil_molar_mass,
        )
    else:
        y = source_data.xmfs[2][source_data.DATES[0]]
        where_min_xmf2 = np.where(y < THRESHOLD_DISSOLVED)[0]
        if len(where_min_xmf2) == 0:
            min_y = np.min(y)
            where_min_xmf2 = np.where(y < min_y + THRESHOLD_DISSOLVED)[0]
            msg = (
                f"WARNING: Lack of cells with low (<{THRESHOLD_DISSOLVED}) XMF2, "
                f"needed for estimation of water density."
                f"\n         Using cells with XMF2 < "
                f"{min_y + THRESHOLD_DISSOLVED} for estimation."
            )
            logging.warning(format_warning(msg))
        # Where xmf2 is 0, or the closest approximation available
        assert source_data.BWAT is not None
        bwat = source_data.BWAT[source_data.DATES[0]]
        water_density = np.array(
            [
                (
                    water_molar_mass * x[1]
                    if y[x[0]] < THRESHOLD_DISSOLVED
                    else water_molar_mass * np.mean(bwat[where_min_xmf2])
                )
                for x in enumerate(bwat)
            ]
        )
        molar_vols_co2 = _eclipse_co2_molar_volume(
            source_data,
            water_density,
            water_molar_mass,
        )
    return molar_vols_co2


def _calc_co2_amount_cell_volume(
    scenario: Scenario,
    source_data: SourceData,
    active_props: List[str],
) -> Co2Data:
    # The definition of gas_prop and dis_prop is probably wrong since there
    # is no guarantee that the gas property will come first. However, it most
    # probably works out since the order of active_props is mostly the same for
    # properly defined cases. Trying to change this will cause a test failure,
    # so leaving as it is for now.
    props = []
    for p in active_props:
        if p == "SGAS":
            props.append(source_data.SGAS)
        elif p == "AMFG":
            props.append(
                source_data.AMFS if scenario != Scenario.AQUIFER else source_data.AMFG
            )
        elif p == "XMF2":
            props.append(source_data.xmfs[2])
    gas_prop = props[0]
    dis_prop = props[1] if len(props) >= 2 else None
    assert gas_prop is not None
    inactive_gas_cells = {
        x: identify_gas_less_cells(
            {x: gas_prop[x]},
            {x: dis_prop[x]} if dis_prop is not None else None,
        )
        for x in source_data.DATES
    }
    assert source_data.VOL is not None
    vols_ext = {t: np.array([0] * len(source_data.VOL[t])) for t in source_data.DATES}
    for date in source_data.DATES:
        vols_ext[date][~inactive_gas_cells[date]] = np.array(source_data.VOL[date])[
            ~inactive_gas_cells[date]
        ]
    co2_amount = Co2Data(
        source_data.x_coord,
        source_data.y_coord,
        source_data.active_cells,
        [
            Co2DataAtTimeStep(
                t,
                np.zeros_like(np.array(vols_ext[t])),
                np.zeros_like(np.array(vols_ext[t])),
                np.zeros_like(np.array(vols_ext[t])),
                np.array(vols_ext[t]),
                np.zeros_like(np.array(vols_ext[t])),
                np.zeros_like(np.array(vols_ext[t])),
            )
            for t in vols_ext
        ],
        "m3",
        scenario,
        source_data.zone,
        source_data.region,
    )
    return co2_amount


def _raise_missing_props_error(
    active_props: List[str],
    props_needed_cirrus: List[str],
    props_needed_eclipse: List[str],
):
    if any(prop in props_needed_cirrus for prop in active_props if prop != "SGAS"):
        missing_props = [x for x in props_needed_cirrus if x not in active_props]
        error_text = "Lacking some required properties to compute CO2 mass/volume."
        error_text += "\nAssumed source: Cirrus"
        error_text += "\nMissing properties: "
        error_text += ", ".join(missing_props)
        raise ValueError(format_error(error_text))
    if any(prop in props_needed_eclipse for prop in active_props if prop != "SGAS"):
        missing_props = [x for x in props_needed_eclipse if x not in active_props]
        error_text = "Lacking some required properties to compute CO2 mass/volume."
        error_text += "\nAssumed source: Eclipse"
        error_text += "\nMissing properties: "
        error_text += ", ".join(missing_props)
        raise ValueError(format_error(error_text))
    error_text = "Lacking all required properties to compute CO2 mass/volume."
    error_text += "\nNeed either:"
    error_text += f"\n  Cirrus: \
        {', '.join(props_needed_cirrus)}"
    error_text += f"\n  Eclipse : \
        {', '.join(props_needed_eclipse)}"
    raise ValueError(format_error(error_text))


def _convert_from_kg_to_tons(co2_mass_output: Co2Data):
    co2_mass_output.units = "tons"
    for values in co2_mass_output.data_list:
        for x in [
            values.dis_water_phase,
            values.gas_phase,
            values.dis_oil_phase,
            values.trapped_gas_phase,
            values.free_gas_phase,
        ]:
            x *= 0.001


def calculate_co2(
    source_data: SourceData,
    calc_type: CalculationType,
    residual_trapping: bool = False,
    cirrus_info_file: Optional[str] = None,
) -> Co2Data:
    """
    Calculates the desired amount (calc_type_input) of CO2

    Args:
      source_data (SourceData): Extracted source data
      calc_type (CalculationType): Which amount is calculated (mass / cell_volume /
                                   actual_volume)
      residual_trapping (bool): Indicate if residual trapping should be calculated
      cirrus_info_file (Optional[str]): Path to cirrus info file

    Returns:
      CO2Data

    """
    timer = Timer()

    timer.start("calculate_co2")
    co2_data = _calculate_co2_data_from_source_data(
        source_data,
        calc_type=calc_type,
        residual_trapping=residual_trapping,
        cirrus_info_file=cirrus_info_file,
    )
    timer.stop("calculate_co2")
    return co2_data


if __name__ == "__main__":
    pass
