from pangolin.testing.backends import BackendTests
import pangolin.torch_backend
import torch
from pangolin import ir


class TestTorch(BackendTests):
    _ancestor_sample_flat = pangolin.torch_backend.ancestor_sample_flat
    _ancestor_log_prob_flat = pangolin.torch_backend.ancestor_log_prob_flat
    _cast = lambda x: torch.tensor(x, dtype=torch.float)
    _ops_without_log_prob_support = {ir.BetaBinomial, ir.Multinomial}
    _ops_without_sampling_support = {
        ir.Beta,
        ir.BetaBinomial,
        ir.Exponential,
        ir.StudentT,
        ir.Dirichlet,
        ir.Multinomial,
        ir.MultiNormal,
        ir.Wishart,
    }
