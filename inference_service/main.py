# app/main.py

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

# --- Model Imports and Path Setup ---
import sys
import os
from contextlib import asynccontextmanager

# Make repo root importable and import from training_service
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from training_service.model import SimpleNN, ConvNN


# --- Paths & Globals ---
DEVICE = "cpu"

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
METADATA_PATH = os.path.join(MODEL_DIR, "models_metadata.json")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_STATIC_DIR = os.path.join(THIS_DIR, "static")
APP_TEMPLATES_DIR = os.path.join(THIS_DIR, "templates")

# Learning Rate Ranges (ENGLISH ONLY)
LR_RANGES = {
    "Low": (0.00001, 0.0001),
    "Medium Low": (0.0001, 0.0005),
    "Medium High": (0.0005, 0.001),
    "High": (0.001, 0.01),
}

# Global stores
MODEL_METADATA = []         # List of all models' metadata
LOADED_MODELS_CACHE = {}    # In-memory cache for loaded models


def _load_model_metadata():
    """Load models metadata file into MODEL_METADATA (if present)."""
    global MODEL_METADATA
    if not os.path.exists(METADATA_PATH):
        print(f"WARNING: {METADATA_PATH} not found. The app will not find any models.")
        MODEL_METADATA = []
        return

    try:
        with open(METADATA_PATH, 'r') as f:
            MODEL_METADATA = json.load(f)
        print(f"Loaded {len(MODEL_METADATA)} model metadata records.")
    except Exception as e:
        print(f"Error loading {METADATA_PATH}: {e}")
        MODEL_METADATA = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context replacing deprecated on_event handlers."""
    # --- Startup ---
    _load_model_metadata()
    try:
        yield
    finally:
        # --- Shutdown cleanup (optional) ---
        LOADED_MODELS_CACHE.clear()
        print("Model cache cleared on shutdown.")


# --- App Configuration ---
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=APP_TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=APP_STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Search & Load Logic ---

def find_matching_model(params: 'ModelParams'):
    """
    Look up a model in metadata that matches the given parameters.
    Uses an LR *range* name (English only) and checks if the stored
    model's actual LR falls within that range.
    Requires exact matches on model_type, optimizer, batch_size, hidden_layers_config, and epochs.
    """
    lr_range_name = params.learning_rate_range
    if lr_range_name not in LR_RANGES:
        return None  # Invalid LR range name

    min_lr, max_lr = LR_RANGES[lr_range_name]

    for metadata in MODEL_METADATA:
        # Check 1: exact match on core parameters (including epochs)
        if (
            metadata['model_type'] == params.model_type and
            metadata['optimizer'] == params.optimizer and
            metadata['batch_size'] == params.batch_size and
            metadata['hidden_layers_config'] == params.hidden_layers_config and
            metadata.get('num_epochs') == params.epochs and
            bool(metadata.get('use_dropout', False)) == bool(params.use_dropout) 
        ):
            # Check 2: LR falls inside the selected range
            model_lr = metadata['learning_rate']
            if min_lr <= model_lr <= max_lr:
                # Found!
                return metadata

    return None  # No matching model found


def _safe_torch_load(path: str, map_location: str):
    """
    Safe torch.load with weights_only=True when available.
    Falls back to default behavior if running on older PyTorch.
    """
    try:
        # Newer PyTorch supports weights_only=True (recommended)
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older versions don't have weights_only param
        return torch.load(path, map_location=map_location)


def get_model(metadata: dict):
    """
    Load a model from a .pth file (or pull it from cache).
    """
    model_id = metadata['id']

    # 1) Try cache first
    if model_id in LOADED_MODELS_CACHE:
        return LOADED_MODELS_CACHE[model_id]

    # 2) Instantiate and load if not cached
    try:
        model_type = metadata['model_type']
        hidden_config = metadata['hidden_layers_config']

        if model_type == "SimpleNN":
            model = SimpleNN(hidden_structure=hidden_config,
                            use_dropout=bool(metadata.get('use_dropout', False)), dropout_p=0.3)
        elif model_type == "ConvNN":
            model = ConvNN(conv_channels=[16, 32], linear_hidden=hidden_config,
                        use_dropout=bool(metadata.get('use_dropout', False)), dropout_p=0.3)

        else:
            raise Exception(f"Unknown model type: {model_type}")

        # Load weights (state_dict) with safe torch.load
        model_path = os.path.join(MODEL_DIR, metadata['filename'])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state_dict = _safe_torch_load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()  # Eval mode
        model.to(DEVICE)

        # 3) Cache and return
        LOADED_MODELS_CACHE[model_id] = model
        print(f"Model loaded and cached: {model_id}")
        return model

    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None


# --- Inference Transform ---
inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# --- Pydantic Models for the API ---
class ModelParams(BaseModel):
    model_type: str
    optimizer: str
    learning_rate_range: str  # "Low", "Medium Low", "Medium High", "High"
    batch_size: int
    epochs: int
    hidden_layers_config: list[int]
    use_dropout: bool = False 


class PredictionRequest(BaseModel):
    params: ModelParams
    image_data: str  # Base64 data URL image


# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main canvas page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Receive a drawing and a SET OF PARAMETERS,
    find a matching trained model, and return a prediction.
    """
    params = request.params

    # 1) Find a matching model in metadata
    metadata = find_matching_model(params)

    if not metadata:
        # Not an error; expected when that configuration hasn't been trained yet
        return {"error": "Model not trained yet."}

    # 2) Load model (or use cache)
    model = get_model(metadata)

    if not model:
        raise HTTPException(status_code=500, detail="Failed to load the matched model.")

    try:
        # 3) Decode Base64 image
        img_data = re.sub('^data:image/.+;base64,', '', request.image_data)
        img_bytes = base64.b64decode(img_data)

        # 4) Convert to PIL image and preprocess
        img_pil = Image.open(BytesIO(img_bytes)).convert('L')
        img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        img_pil = ImageOps.invert(img_pil)

        # 5) To tensor & normalize
        tensor = inference_transform(img_pil).unsqueeze(0)

        # 6) Predict
        with torch.no_grad():
            output = model(tensor.to(DEVICE))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities.max().item() * 100

        # 7) Return success payload
        return {
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "found_model": {
                "id": metadata['id'],
                "accuracy": f"{metadata['accuracy']:.2f}%",
                "learning_rate": metadata['learning_rate'],
                "epochs": metadata.get('num_epochs'),
                "training_time": f"{metadata['training_time']:.2f}s"
            }
        }

    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing the image: {e}")


if __name__ == "__main__":
    if not os.path.exists(METADATA_PATH):
        print(f"WARNING: {METADATA_PATH} not found.")
        print("The app will run, but it will not find any trained models.")
    print("Start the server with: uvicorn main:app --reload")
