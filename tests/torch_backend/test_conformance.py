from pangolin.testing.backends import BackendTests
import pangolin.torch_backend
import torch
from pangolin import ir


class TestTorch(BackendTests):
    ancestor_sample_flat = staticmethod(pangolin.torch_backend.ancestor_sample_flat)
    ancestor_log_prob_flat = staticmethod(pangolin.torch_backend.ancestor_log_prob_flat)
    cast = staticmethod(lambda x: torch.tensor(x, dtype=torch.float))
    _ops_without_log_prob_support = frozenset({ir.BetaBinomial, ir.Multinomial})
    _ops_without_sampling_support = frozenset(
        {
            ir.Beta,
            ir.BetaBinomial,
            ir.StudentT,
            ir.Dirichlet,
            ir.Multinomial,
            ir.MultiNormal,
            ir.Wishart,
            ir.Cauchy,
        }
    )
