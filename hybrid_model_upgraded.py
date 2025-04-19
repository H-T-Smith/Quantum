
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# Set random seed
algorithm_globals.random_seed = 42

# Load data
df_train = pd.read_csv("db_sc1_bluetooth.csv")
df_test = pd.read_csv("Tests_Scenario1_bluetooth.csv")

X_train = df_train[["RSSI A", "RSSI B", "RSSI C"]].values
Y_train = df_train[["x", "y"]].values

X_test = df_test[["RSSI A", "RSSI B", "RSSI C"]].values
Y_test = df_test[["x", "y"]].values

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Quantum circuit
feature_map = ZZFeatureMap(feature_dimension=3)
ansatz = RealAmplitudes(num_qubits=3, reps=3)

qc = QuantumCircuit(3)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

# Estimator and QNN
estimator = StatevectorEstimator()
qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    estimator=estimator,
    input_gradients=True
)

# Classical neural network
class ClassicalNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Hybrid model
class HybridModel(nn.Module):
    def __init__(self, qnn, classical_nn):
        super().__init__()
        self.qnn = TorchConnector(qnn)
        self.classical_nn = classical_nn

    def forward(self, x):
        qnn_out = self.qnn(x)
        class_out = self.classical_nn(x)
        return qnn_out + class_out

# Initialize models and training setup
classical_nn = ClassicalNN()
hybrid_model = HybridModel(qnn, classical_nn)
optimizer = optim.Adam(hybrid_model.parameters(), lr=0.1)
loss_func = nn.MSELoss()

# Training loop
import time
epochs = 80
training_times = []

for epoch in range(epochs):
    start_time = time.time()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = hybrid_model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
    training_times.append(time.time() - start_time)

print(f"Total training time: {np.sum(training_times):.2f} seconds")
