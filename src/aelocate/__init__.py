"""aelocate: acoustic emission acquisition simulator, hit detection and source location."""

from .daq import AcquisitionBlock, AcquisitionSimulator, Emitter, GroundTruthEvent, NoiseGenerator, simulate_event
from .dispersion import DispersionCurve, lamb_dispersion
from .events import Event, EventBuilder
from .hits import ChannelHitDetector, Hit, HitDetector, HitDetectorConfig
from .instrument import FrontEnd, Sensor
from .location import LinearLocator, LocatedEvent, PlanarLocator, calibrate_velocity, linear_location
from .materials import ALUMINIUM, STAINLESS_304, STEEL, Material
from .pickers import AICPicker, ThresholdPicker, aic_onset
from .pipeline import ChainOutput, ProcessingChain
from .presets import PRESETS, Setup, steel_pipe_linear, steel_plate_linear, steel_plate_planar
from .simulator import WaveformSimulator
from .sources import CRACK, PLB, PULSER, BurstSource, StepSource
from .specimen import LinearSpecimen, PlanarSpecimen
from .units import dbae_to_volts, volts_to_dbae

__version__ = "0.1.0"

__all__ = [
    "AcquisitionBlock", "AcquisitionSimulator", "Emitter", "GroundTruthEvent", "NoiseGenerator", "simulate_event",
    "DispersionCurve", "lamb_dispersion", "Event", "EventBuilder", "ChannelHitDetector", "Hit", "HitDetector",
    "HitDetectorConfig", "FrontEnd", "Sensor", "LinearLocator", "LocatedEvent", "PlanarLocator",
    "calibrate_velocity", "linear_location", "ALUMINIUM", "STAINLESS_304", "STEEL", "Material", "AICPicker",
    "ThresholdPicker", "aic_onset", "ChainOutput", "ProcessingChain", "PRESETS", "Setup", "steel_pipe_linear",
    "steel_plate_linear", "steel_plate_planar", "WaveformSimulator", "CRACK", "PLB", "PULSER", "BurstSource",
    "StepSource", "LinearSpecimen", "PlanarSpecimen", "dbae_to_volts", "volts_to_dbae", "__version__",
]
