from ._generator import Generator, default_rng, exponential, standard_exponential  # noqa
from ._legacy import randint, standard_normal, uniform

__all__ = [
    'Generator',
    'randint',
    'standard_normal',
    'uniform',
    'exponential',
    'standard_exponential',
]

# adc, added both exponential lines
