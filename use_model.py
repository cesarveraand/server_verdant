import json
import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
from utils import load_model

# Rutas a los archivos JSON
class_idx_to_species_id_path = r"C:\Users\LENOVO\Desktop\Projects\plnt\plantnet-300kvw\class_idx_to_species_id.json"
species_id_to_name_path = r"C:\Users\LENOVO\Desktop\Projects\plnt\plantnet-300kvw\plantnet300K_species_id_2_name.json"

# Cargar los mapeos
with open(class_idx_to_species_id_path, 'r') as f:
    class_idx_to_species_id = json.load(f)

with open(species_id_to_name_path, 'r') as f:
    species_id_to_name = json.load(f)

# Ruta al archivo de pesos
filename = r"C:\Users\LENOVO\Desktop\Projects\plnt\plantnet-300kvw\resnet18_weights_best_acc.tar"

# Cargar el modelo preentrenado
model = resnet18(num_classes=1081)  # El dataset tiene 1081 clases
device = "cuda" if torch.cuda.is_available() else "cpu"
load_model(model, filename=filename, use_gpu=torch.cuda.is_available())

# Configurar el modelo
model.eval()
model.to(device)

# Ruta al archivo de imagen
image_path = r"C:\Users\LENOVO\Desktop\Projects\plnt\plantnet-300kvw\images\prueba14.jpg"

# Verificar si la imagen existe
if not os.path.exists(image_path):
    raise FileNotFoundError(f"El archivo no existe: {image_path}")

# Preprocesamiento de la imagen
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Abrir y preprocesar la imagen
try:
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
except Exception as e:
    raise RuntimeError(f"Error al procesar la imagen: {e}")

# Realizar predicción
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

# Obtener el ID de la especie
species_id = class_idx_to_species_id.get(str(predicted_class), None)
if species_id is None:
    raise ValueError(f"No se encontró un ID de especie para la clase {predicted_class}")

# Obtener el nombre científico de la especie
species_name = species_id_to_name.get(species_id, None)
if species_name is None:
    raise ValueError(f"No se encontró un nombre científico para el ID de especie {species_id}")

# Imprimir el resultado
print(f"Clase predicha: {predicted_class}")
print(f"ID de especie: {species_id}")
print(f"Especie correspondiente: {species_name}")
