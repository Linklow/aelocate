"""Demo source scenarios: a defect that emits AE in a cluster plus background noise sources."""

from __future__ import annotations

from .daq import Emitter
from .presets import Setup
from .sources import CRACK, PLB


def default_emitters(setup: Setup, crack_rate: float = 3.0, noise_rate: float = 2.0) -> list[Emitter]:
    """A growing crack (tight cluster) and diffuse background events over the whole object."""
    sp = setup.specimen
    if setup.is_linear:
        xs = sp.sensor_x
        defect = (xs[0] + 0.62 * (xs[-1] - xs[0]),)
        return [
            Emitter("crack", CRACK, crack_rate, defect, spread=0.01, amplitude_db=(55, 80)),
            Emitter("background", PLB, noise_rate, (sp.length / 2,), spread=sp.length / 2,
                    distribution="uniform", amplitude_db=(48, 70)),
        ]
    # Planar location uses one (S0) velocity, so the demo keeps sources strong enough for S0 to
    # trigger on all four sensors; weaker sources show the same mode mismatch as linear location.
    defect = (0.35 * sp.width, 0.6 * sp.height)
    return [
        Emitter("crack", CRACK, crack_rate, defect, spread=0.01, amplitude_db=(82, 95)),
        Emitter("background", PLB, noise_rate, (sp.width / 2, sp.height / 2), spread=sp.width / 2,
                distribution="uniform", amplitude_db=(65, 90)),
    ]
