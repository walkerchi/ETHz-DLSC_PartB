from .heat_equation import HeatEquation
from .wave_equation import WaveEquation
from .poisson_equation import PoissonEquation

EquationLookUp = {
    'heat':HeatEquation,
    'wave':WaveEquation,
    'poisson':PoissonEquation
}
