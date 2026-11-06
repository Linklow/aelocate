"""Unit conventions used across the package.

AE amplitude follows the usual instrument convention (ASTM E1316, PAC/MISTRAS, Vallen):

    dB_AE = 20 * log10(V / 1 uV)

where ``V`` is the sensor output voltage *before* the preamplifier. All waveforms inside
``aelocate`` are stored sensor-referred (volts at the sensor), so a threshold of 45 dB_AE is
always 178 uV regardless of the preamplifier gain. The gain only matters for ADC clipping
and quantisation, which :class:`aelocate.instrument.FrontEnd` handles.
"""

from __future__ import annotations

import numpy as np

REF_VOLT = 1e-6
"""Reference voltage for dB_AE (1 uV at the sensor)."""

ENERGY_REF_OHM = 10e3
"""Reference impedance for absolute AE energy (Vallen convention), ohms."""

ATTOJOULE = 1e-18


def volts_to_dbae(v):
    """Convert sensor voltage (peak or RMS, volts) to dB_AE."""
    v = np.abs(np.asarray(v, dtype=float))
    return 20.0 * np.log10(np.maximum(v, 1e-15) / REF_VOLT)


def dbae_to_volts(db):
    """Convert dB_AE to sensor voltage in volts."""
    return REF_VOLT * 10.0 ** (np.asarray(db, dtype=float) / 20.0)


def db_to_ratio(db):
    """Amplitude ratio for a gain in dB (e.g. 40 dB -> 100)."""
    return 10.0 ** (np.asarray(db, dtype=float) / 20.0)


def db_per_m_to_np_per_m(alpha_db):
    """Attenuation coefficient from dB/m to Np/m (amplitude)."""
    return np.asarray(alpha_db, dtype=float) * np.log(10.0) / 20.0
