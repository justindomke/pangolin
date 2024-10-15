#from pangolin.inference.numpyro.model import get_numpyro_rv

import pangolin.inference.numpyro.vmap # so handler is registered
import pangolin.inference.numpyro.autoregressive # so handler is registered
from pangolin.inference.numpyro.sampling import sample, sample_flat, E, var, std

__all__ = ['sample', 'sample_flat', 'E', 'var', 'std']