from torchvision import transforms
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference(model, classes, input_folder, image_size):
    
    transformaciones = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    indices_classes_interes = [0,3]
    
    print(f"MODELO QCNN")
    
    for nombre_archivo in os.listdir(input_folder):
        if not nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
    
        ruta_imagen = os.path.join(input_folder, nombre_archivo)
        imagen = Image.open(ruta_imagen).convert("RGB")
        image_tensor = transformaciones(imagen).unsqueeze(0).to(device)
    
        with torch.no_grad():
            salida = model(image_tensor)
            salida = salida[:, indices_classes_interes]
                
            probs = F.softmax(salida, dim=1).cpu().numpy().flatten()
            print(f"Probabilidades: red = {probs[0]:.4f}, green = {probs[1]:.4f}")
            pred_clase = torch.argmax(salida, dim=1).item()
            nombre_clase = classes[pred_clase]
    
        plt.imshow(imagen)
        plt.axis('off')
        plt.title(f'{nombre_archivo}    Predicción: {nombre_clase}', fontsize=14)
        plt.show()
