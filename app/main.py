import torch
import torch.nn as nn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import numpy as np
import base64
import re
from io import BytesIO
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import json

# --- Importaciones de Modelos y Rutas ---
import sys
import os

# Añadimos la carpeta 'src' al path para poder importar 'model.py'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import SimpleNN, ConvNN # Importamos AMBAS clases

# --- Configuración de la App ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
# Servimos /static desde la carpeta 'static'
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuración de Modelos ---
DEVICE = "cpu"
# Apuntamos a las carpetas raíz (fuera de 'app/' y 'src/')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
METADATA_PATH = os.path.join(MODEL_DIR, "models_metadata.json")

# Definición de rangos de Learning Rate
LR_RANGES = {
    "Bajo": (0.00001, 0.0001),
    "Medio Bajo": (0.0001, 0.0005),
    "Medio Alto": (0.0005, 0.001),
    "Alto": (0.001, 0.01)
}

# Almacenes globales
MODEL_METADATA = [] # Lista de todos los metadatos de modelos
LOADED_MODELS_CACHE = {} # Caché para modelos ya cargados en memoria

@app.on_event("startup")
async def startup_event():
    """Al iniciar la app, carga los metadatos de los modelos."""
    global MODEL_METADATA
    if not os.path.exists(METADATA_PATH):
        print(f"ADVERTENCIA: No se encontró {METADATA_PATH}. La app no encontrará modelos.")
        return
    
    try:
        with open(METADATA_PATH, 'r') as f:
            MODEL_METADATA = json.load(f)
        print(f"Cargados {len(MODEL_METADATA)} registros de metadatos de modelos.")
    except Exception as e:
        print(f"Error al cargar {METADATA_PATH}: {e}")


# --- Lógica de Búsqueda y Carga ---

def find_matching_model(params: 'ModelParams'):
    """
    Busca en los metadatos un modelo que coincida con los parámetros.
    """
    lr_range_name = params.learning_rate_range
    if lr_range_name not in LR_RANGES:
        return None # Rango de LR no válido
    
    min_lr, max_lr = LR_RANGES[lr_range_name]

    for metadata in MODEL_METADATA:
        # Comprobación 1: Coincidencia exacta de parámetros
        if (
            metadata['model_type'] == params.model_type and
            metadata['optimizer'] == params.optimizer and
            metadata['batch_size'] == params.batch_size and
            metadata['hidden_layers_config'] == params.hidden_layers_config
        ):
            # Comprobación 2: Rango de Learning Rate
            model_lr = metadata['learning_rate']
            if min_lr <= model_lr <= max_lr:
                # ¡Encontrado!
                return metadata
    
    return None # No se encontró ningún modelo que coincida

def get_model(metadata: dict):
    """
    Carga un modelo desde un archivo .pth (o lo toma del caché).
    """
    model_id = metadata['id']
    
    # 1. Revisar el caché primero
    if model_id in LOADED_MODELS_CACHE:
        return LOADED_MODELS_CACHE[model_id]

    # 2. Si no, instanciar y cargar
    try:
        model_type = metadata['model_type']
        hidden_config = metadata['hidden_layers_config']

        if model_type == "SimpleNN":
            model = SimpleNN(hidden_structure=hidden_config)
        elif model_type == "ConvNN":
            # Asumimos los canales convolucionales por defecto de la app de entrenamiento
            model = ConvNN(
                conv_channels=[16, 32], 
                linear_hidden=hidden_config
            )
        else:
            raise Exception(f"Tipo de modelo desconocido: {model_type}")

        # Cargar los pesos (state_dict)
        model_path = os.path.join(MODEL_DIR, metadata['filename'])
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval() # Poner en modo evaluación
        model.to(DEVICE)
        
        # 3. Guardar en caché y retornar
        LOADED_MODELS_CACHE[model_id] = model
        print(f"Modelo cargado y cacheado: {model_id}")
        return model

    except Exception as e:
        print(f"Error al cargar el modelo {model_id}: {e}")
        return None


# --- Transformación de Inferencia ---
inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- Clases Pydantic para la API ---
class ModelParams(BaseModel):
    model_type: str
    optimizer: str
    learning_rate_range: str # "Bajo", "Medio Bajo", etc.
    batch_size: int
    hidden_layers_config: list[int]

class PredictionRequest(BaseModel):
    params: ModelParams
    image_data: str # Imagen en formato base64 data URL

# --- Rutas de la API ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Sirve la página principal con el canvas."""
    # Ya no pasamos 'model_names', el frontend tiene el formulario
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Recibe un dibujo y un CONJUNTO DE PARÁMETROS,
    encuentra el modelo, y devuelve una predicción.
    """
    params = request.params
    
    # 1. Encontrar el modelo en metadatos
    metadata = find_matching_model(params)
    
    if not metadata:
        # Si no hay modelo, no es un error, es el flujo esperado
        return {"error": "Modelo no entrenado todavía."}

    # 2. Cargar el modelo (o usar caché)
    model = get_model(metadata)
    
    if not model:
        raise HTTPException(status_code=500, detail="Error al cargar el modelo encontrado.")

    try:
        # 3. Decodificar la imagen Base64
        img_data = re.sub('^data:image/.+;base64,', '', request.image_data)
        img_bytes = base64.b64decode(img_data)
        
        # 4. Convertir a imagen PIL y pre-procesar
        img_pil = Image.open(BytesIO(img_bytes)).convert('L')
        img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        img_pil = ImageOps.invert(img_pil)
        
        # 5. Convertir a Tensor y Normalizar
        tensor = inference_transform(img_pil).unsqueeze(0)

        # 6. Realizar la predicción
        with torch.no_grad():
            output = model(tensor.to(DEVICE))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities.max().item() * 100

        # 7. Retornar éxito
        return {
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "found_model": {
                "id": metadata['id'],
                "accuracy": f"{metadata['accuracy']:.2f}%",
                "learning_rate": metadata['learning_rate'],
                "training_time": f"{metadata['training_time']:.2f}s"
            }
        }
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {e}")


if __name__ == "__main__":
    if not os.path.exists(METADATA_PATH):
        print(f"ADVERTENCIA: No se encontró {METADATA_PATH}.")
        print("La app funcionará, pero no encontrará ningún modelo entrenado.")
    print("Inicia el servidor con: uvicorn main:app --reload")