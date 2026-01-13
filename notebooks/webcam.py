import cv2
import torch
from ultralytics import YOLO
from pennylane import numpy as np
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.nn.models.quantum.QCNN import QuantumCircuitModel
from src.utils.run_conf import CargaConf
from src.YOLO.yolo_funcs import quantum_frame_predict


image_size = 16
n_classes = 2

pesos_modelo = ".../model.pth"  
conf = CargaConf(image_size, n_classes, np.ceil(np.log2(3*image_size**2)).astype(int))

# Modelo YOLO
yolo_model = YOLO(".../dataset/best.pt") 

# Modelo cuántico
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_wires = np.ceil(np.log2(3*image_size**2)).astype(int)
modelq = QuantumCircuitModel(
    n_wires=n_wires,
    embedding=conf["embedding"],
    circuit=conf["ansatz"],
    measurement=conf["measurement"],
    weight_shapes=conf["weight_shapes"],
    reshaper=conf["reshaper"],
).to(device)

modelq.load_state_dict(torch.load(pesos_modelo, map_location=device))
modelq.eval()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 140)

if not cap.isOpened():
    print("Camera not found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar semáforo con YOLO
    results = yolo_model.predict(frame, imgsz=320, verbose=False)

    # Pasamos al modelo la imagen
    estado, frame_to_show = quantum_frame_predict(modelq, results, frame, image_size, "Hojas")

    # Mostrar por consola
    print(f"Semáforo: {estado}")

    # Mostrar en cámara
    cv2.imshow("Detección", frame_to_show)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
