from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeManila
from qiskit_ibm_runtime import Estimator, Options

def get_simulator_backend(noise_type="fake_device") -> AerSimulator:
    """
    Returns a Qiskit AerSimulator with optional noise model.
    
    Supported types:
    - 'ideal': no noise
    - 'fake_device': uses Qiskit's FakeManila backend for realistic noise
    """
    if noise_type == "ideal":
        return AerSimulator()
    elif noise_type == "fake_device":
        fake_backend = FakeManila()
        noise_model = NoiseModel.from_backend(fake_backend)
        return AerSimulator(noise_model=noise_model)
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")
