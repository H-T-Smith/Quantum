from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap, RealAmplitudes, EfficientSU2, TwoLocal 
from qiskit_machine_learning.neural_networks import EstimatorQNN

def custom_rx_rz_map(num_qubits: int):
    """
    Creates a custom RX-RZ feature map with one parameter per qubit.
    
    Returns:
        qc: QuantumCircuit
        parameters: list of Parameters
    """
    x = ParameterVector("x", length=num_qubits)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(x[i], i)
        qc.rz(x[i], i)
    return qc, list(x)


def create_qnn(
        num_qubits: int=3,
        reps: int=1,
        feature_map_type: str = "zz",
        ansatz_type: str = "real",
        estimator=None,
    ) -> EstimatorQNN:

    """
    Creates a Qiskit EstimatorQNN with configurable feature map and ansatz.

    Args:
        num_qubits: number of qubits (input dimension)
        reps: number of repetitions for both feature map and ansatz
        feature_map_type: one of 'zz', 'z', 'pauli'
        ansatz_type: one of 'real', 'su2', 'twolocal'
        estimator: optional custom estimator backend (e.g., with noise)

    Returns:
        EstimatorQNN: quantum neural network object
    """

    # Feature Map Selection
    if feature_map_type == "zz":
        feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=reps)
        input_params = feature_map.parameters
    elif feature_map_type == "z":
        feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=reps)
        input_params = feature_map.parameters
    elif feature_map_type == "pauli":
        feature_map = PauliFeatureMap(feature_dimension=num_qubits, reps=reps, paulis=["X", "Y", "Z"])
        input_params = feature_map.parameters
    elif feature_map_type == "custom_rxrz":
        feature_map, input_params = custom_rx_rz_map(num_qubits)
    else:
        raise ValueError(f"Unsupported feature map type: {feature_map_type}")
    
    # Trainable Layer (Ansatz) Selection
    if ansatz_type == "real":
        ansatz = RealAmplitudes(num_qubits, reps=reps)
    elif ansatz_type == "su2":
        ansatz = EfficientSU2(num_qubits, reps=reps)
    elif ansatz_type == "twolocal":
        ansatz = TwoLocal(num_qubits, ['ry', 'rz'], 'cx', reps=reps)
    else:
        raise ValueError(f"Unsupported ansatz type: {ansatz_type}")


    # Full Circuit Composition
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    
    # EstimatorQNN Constructor
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=input_params,
        weight_params=ansatz.parameters,
        input_gradients=True,
        estimator=estimator,
    )
    return qnn