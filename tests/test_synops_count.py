"""The harness must see every synapse of our RSNN, including the recurrent V matrices (counted only through
nn.Linear hooks, so they must be module calls), as it does for the upstream model."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from neurobench.benchmarks import Benchmark
from neurobench.metrics.workload import SynapticOperations
from neurobench.models import TorchModel

from nbfscil.snn import RSNN, RadLIFLayer


def test_dense_synops_include_recurrent_weights():
    torch.manual_seed(0)
    n_in, h, n_out, T, B = 4, 8, 3, 5, 2
    net = RSNN(layer_sizes=(h, h), n_in=n_in, n_out=n_out, dropout=0.0, init_state="zeros")
    model = TorchModel(net)
    model.add_activation_module(RadLIFLayer)
    x = (torch.rand(B, T, n_in) > 0.5).float()
    loader = DataLoader(TensorDataset(x, torch.zeros(B, dtype=torch.long)), batch_size=B)
    res = Benchmark(model, metric_list=[[], [SynapticOperations]], dataloader=loader,
                    preprocessors=[], postprocessors=[]).run()
    per_step = n_in * h + h * h + h * h + h * h + h * n_out   # W1, V1, W2, V2, readout
    assert res["SynapticOperations"]["Dense"] == T * per_step
