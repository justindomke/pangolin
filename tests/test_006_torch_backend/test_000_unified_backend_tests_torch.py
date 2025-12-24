from pangolin.testing.backends import BackendTests
import pangolin.torch_backend
import torch

class TestTorch(BackendTests):
    _ancestor_sample_flat = pangolin.torch_backend.ancestor_sample_flat
    _ancestor_log_prob_flat = pangolin.torch_backend.ancestor_log_prob_flat
    _cast = lambda x: torch.tensor(x, dtype=torch.float)