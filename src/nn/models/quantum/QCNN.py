import torch
import torch.nn as nn
import pennylane as qml

from src.nn.encodings.pennylane_templates import amplitude_embedding
from src.utils.reshape_data import ReshapeDATA

import sys
import os
        
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_default_device(device)

class QuantumCircuitModel(nn.Module):
    def __init__(self, n_wires,embedding, circuit=None, measurement=None, params=None, 
                 weight_shapes=None,reshaper=None, qdevice_kwargs=None):
        super(QuantumCircuitModel, self).__init__()
        self.n_wires = n_wires
        self.embedding = embedding
        self.emb = embedding["func"] or amplitude_embedding
        self.embedding_params = embedding["func_params"] or {}
        self.circuit = circuit
        self.circ = circuit["func"]
        self.circuit_params = circuit["func_params"] or None
        self.weight_shapes = weight_shapes or {}
        self.qdevice_kwargs = qdevice_kwargs or {}
        self.torch_device = device
        self.measurement = measurement
        self.meas = measurement["func"] 
        self.measurement_params = measurement["func_params"] or {}
        self.reshaper = reshaper or ReshapeDATA(wires=range(n_wires),params={'structure':'flat'})
        
        qml_device_name = self.qdevice_kwargs.pop('qml_device_name', 'default.qubit')
        self.qml_device = qml.device(
            qml_device_name, wires=self.n_wires, **self.qdevice_kwargs
        )
        self.qnode = qml.QNode(
            self.quantum_circuit,
            self.qml_device,
            interface='torch',
            diff_method='backprop'
        )
        self.qlayer = qml.qnn.TorchLayer(self.qnode,weight_shapes)

        self.dropout = nn.Dropout(p=0.2)
        
    def quantum_circuit(self, inputs,weights):
        wires = range(self.n_wires)
        
        # Embedding block
        self.emb(inputs, wires, self.embedding_params)

        # Circuit block
        params_circuit = self.circuit_params
        params_circuit['weights'] = weights
        self.circ(wires,params_circuit)

        # Measurement block
        return self.meas(wires, self.measurement_params)

    def forward(self,x):
        x = x.to(self.torch_device)
        x = self.reshaper.reshape(x)

        x = self.qlayer(x)
        #print(x.shape)
        #x = self.dropout(x)

        return 15*x