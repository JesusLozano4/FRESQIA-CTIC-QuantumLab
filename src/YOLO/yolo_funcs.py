from roboflow import Roboflow
from PIL import Image as Image2
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
from src.nn.models.hybrid.HQNN_quanv import FlexHybridCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def crop(results, output_dir):

    crop_index = 0
    
    for result in results:
        print(f"{result.path} -> {len(result.boxes[0])} detecciones")
        
    for result in results:
        im = Image2.open(result.path)
        boxes = result.boxes[0]
        if boxes is None:
            continue
    
        for box in boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
    
            # Recortar imagen
            cropped = im.crop((x1, y1, x2, y2))
    
            # Guardar imagen recortada
            crop_filename = f"{output_dir}/crop_{crop_index}.jpg"
            print(f"Saved crop {crop_filename}")
            cropped.save(crop_filename)
            crop_index += 1



def clasificar_hojas(crop_image, image_size, modelq):
    clases = ['healthy', 'scorch']

    trans = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    indices_clases_interes = [0,1]

    imagen = Image2.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
    image_tensor = trans(imagen).unsqueeze(0).to(device)

    with torch.no_grad():
        salida = modelq(image_tensor)
        salida = salida[:, indices_clases_interes]
            
        probs = F.softmax(salida, dim=1).cpu().numpy().flatten()
        print(f"Probabilidades: healthy = {probs[0]:.4f}, scorch = {probs[1]:.4f}")
        if probs[0] >= 0.5:
            nombre_clase = 'healthy'
        else:
            nombre_clase = 'scorch'

    return nombre_clase


def quantum_predict(modelq, results, image_size, clases):
    import os
    import cv2

    # Carpeta base
    base_dir = "quantum_results"
    os.makedirs(base_dir, exist_ok=True)

    for result in results:
        img_path = result.path
        original = cv2.imread(img_path)

        # Valor por defecto por si no hay detecciones
        direc = ""

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            crop = original[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 1 or crop.shape[1] < 1:
                continue

            resized_crop = cv2.resize(crop, (16, 16))

            if clases == "Hojas":
                clase = clasificar_hojas(resized_crop, image_size, modelq)
                color = (0, 255, 0) if clase == "healthy" else (0, 255, 255) if clase == "scorch" else (0, 0, 255)
                direc = "_hojas_lejos"
                if isinstance(modelq, FlexHybridCNN):
                    direc += "H"

            cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)
            cv2.putText(original, clase, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Guardar la imagen anotada aunque no haya detecciones
        save_dir = os.path.join(base_dir, direc if direc else "")
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(save_dir, filename), original)

        

def quantum_frame_predict(modelq, results, frame, image_size, clases):
    
    original = frame.copy()
    result = results[0]

    estado = "No se detecta semáforo"

    if len(result.boxes) == 0:
        return estado, frame

    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Recortar el semáforo
        crop = original[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 1 or crop.shape[1] < 1:
            continue

        # Redimensionar a 16x16
        resized_crop = cv2.resize(crop, (16, 16))
        if clases == "Hojas":
            # Clasificar con modelo
            clase = clasificar_hojas(resized_crop, image_size, modelq)

            # Dibujar la caja y la clase sobre la imagen original
            color = (0, 255, 0) if clase == "green" else (0, 255, 255) if clase == "yellow" else (0, 0, 255)

        cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)
        cv2.putText(original, clase, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if i == 0:
            estado = clase

    return estado, original
