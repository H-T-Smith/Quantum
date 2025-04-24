import torch.nn as nn
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

class HybridModel(nn.Module):
    def __init__(self, qnn: EstimatorQNN, classical_nn: nn.Module):
        super(HybridModel, self).__init__()
        self.qnn = TorchConnector(qnn)
        self.classical_nn = classical_nn

    def forward(self, x):
        x_qnn = self.qnn(x)
        x_classical = self.classical_nn(x)
        return x_qnn + x_classical
