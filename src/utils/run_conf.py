from src.nn.encodings.pennylane_templates import amplitude_embedding, angle_embedding
from src.nn.ansatz.conv_ansatz import QCNN_multiclass, get_num_params_QCNN_multiclass
from src.nn.ansatz.full_entanglement_circuit import full_entanglement_circuit
from src.nn.measurements.multiqubit_observable_measurement import get_pauli_multiqubit_observables,get_pauli_words,random_pauli_string_over_meas_wires,measurement_multiqubit
from src.utils.reshape_data import ReshapeDATA
from src.nn.measurements.default import default_measurement
import torch
import pennylane as qml
from pennylane import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def CargaConf(image_size, n_classes, n_wires):

    # Embedding configuration
    embedding_configuration = {
        'name': 'amplitude',
        'func': amplitude_embedding,
        'func_params': {
            'img_pixels': image_size
        }
    }
    
    # Ansatz configuration
    ansatz_configuration = {
        'name': 'QCNN_multiclass',
        'func': QCNN_multiclass,
        'func_params': {
            'dropped_wires':[],
            'layers_FC': 2,
            }
    }
    
    weight_shapes,_ = get_num_params_QCNN_multiclass(range(n_wires),ansatz_configuration['func_params'])
    
    # Measurement configuration
    meas_wires = [0,4]
    measurement_configuration = {
        'name': 'paulis',
        'func': measurement_multiqubit,
        'func_params': {
            'meas_wires': meas_wires,
            'observables': random_pauli_string_over_meas_wires(range(n_wires),{'meas_wires':meas_wires,'n_obs': n_classes}),
        }
    }
    
    conf = {
        'n_wires': n_wires,
        'embedding': embedding_configuration,
        'ansatz': ansatz_configuration,
        'measurement': measurement_configuration,
        'weight_shapes': {'weights':weight_shapes},
        'reshaper': ReshapeDATA(wires=range(n_wires),params={'structure':'random','img_pixels':image_size})
    }

    return conf

def CargaConfH(image_size):

    # Embedding configuration
    embedding_configuration = {
        'name': 'angle',
        "func": angle_embedding,
        "func_params": {
            "rotation" : "X"
        }
    }
    
    num_layers = 2
    qkernel_shape = 3
    
    # Ansatz configuration
    ansatz_configuration = {
        'name': 'Entanglement',
        "func": full_entanglement_circuit,
        "func_params": {
            "num_layers": num_layers,
            "weights": torch.randn(num_layers, qkernel_shape**2, 3, device=device) % np.pi,
        }
    }
    
    # Measurement configuration
    measurement_configuration = {
        'name': 'default',
        "func": default_measurement,
        "func_params": {
            "observable": qml.PauliZ,
        }
    }
    
    conf = {
        'embedding': embedding_configuration,
        'ansatz': ansatz_configuration,
        'measurement': measurement_configuration,
    }

    return conf