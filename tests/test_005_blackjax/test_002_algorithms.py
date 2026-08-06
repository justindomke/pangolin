import pangolin
from pangolin import interface as pi
from pangolin import blackjax
import numpy as np


def test_nuts():
    x = pi.normal(0, 1)
    # calc = blackjax.Calculate(blackjax.sample_nuts, num_samples=1000)
    calc = blackjax.blackjax_calculate(blackjax.run_nuts, num_samples=1000)
    x_samps = calc.sample(x, [], [])
    assert x_samps.shape == (1000,)
    Ex = calc.E(x, [], [])
    assert Ex.shape == ()


def test_pathfinder():
    x = pi.normal(-4, 2)
    # calc = blackjax.Calculate(blackjax.sample_pathfinder, maxiter=1000, num_samples=1000)
    calc = blackjax.blackjax_calculate(blackjax.run_pathfinder, maxiter=1000, num_samples=1000)
    x_samps = calc.sample(x, [], [])
    assert x_samps.shape == (1000,)
    Ex = calc.E(x, [], [])
    assert Ex.shape == ()
