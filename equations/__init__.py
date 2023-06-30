from .heat_equation import HeatEquation
from .wave_equation import WaveEquation


EquationLookUp = {
    'heat':HeatEquation,
    'wave':WaveEquation,
}
