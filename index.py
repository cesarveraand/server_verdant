from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
from utils import load_model
import json
import os

# Configurar Flask
app = Flask(__name__)

# Cargar los mapeos desde archivos locales
class_idx_to_species_id_path = os.path.join(os.path.dirname(__file__), 'class_idx_to_species_id.json')
species_id_to_name_path = os.path.join(os.path.dirname(__file__), 'plantnet300K_species_id_2_name.json')

with open(class_idx_to_species_id_path, 'r') as f:
    class_idx_to_species_id = json.load(f)

with open(species_id_to_name_path, 'r') as f:
    species_id_to_name = json.load(f)

# Ruta al archivo de pesos
model_weights_path = os.path.join(os.path.dirname(__file__), 'resnet18_weights_best_acc.tar')

# Cargar el modelo preentrenado
model = resnet18(num_classes=1081)
device = "cuda" if torch.cuda.is_available() else "cpu"
load_model(model, filename=model_weights_path, use_gpu=torch.cuda.is_available())

# Configurar el modelo
model.eval()
model.to(device)

# Transformaciones para preprocesamiento
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Ruta de la API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        image = Image.open(image_file.stream).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        species_id = class_idx_to_species_id.get(str(predicted_class), None)
        if species_id is None:
            return jsonify({'error': f'No species ID found for class {predicted_class}'}), 404

        species_name = species_id_to_name.get(species_id, None)
        if species_name is None:
            return jsonify({'error': f'No scientific name found for species ID {species_id}'}), 404

        return jsonify({
            'predicted_class': predicted_class,
            'species_id': species_id,
            'species_name': species_name
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
