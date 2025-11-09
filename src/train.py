# En src/train.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import json  # <--- Importar JSON

# --- Definir rutas raíz (un nivel ARRIBA de este archivo) ---
# os.path.dirname(__file__) es la carpeta 'src'
# os.path.abspath(os.path.join(..., "..")) sube un nivel
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# --- (evaluate_model no cambia) ---
def evaluate_model(model, test_loader, device):
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


# --- (save_loss_plot MODIFICADO para usar la nueva ruta) ---
def save_loss_plot(losses, model_type, opt_name, accuracy, time_taken):
    """
    Guarda el gráfico de pérdida en la carpeta raíz /results
    """
    # 1. Asegura que la carpeta raíz /results exista
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # (El resto de la lógica de pandas para la media móvil es igual)
    loss_series = pd.Series(losses)
    ma_short = loss_series.rolling(window=10, min_periods=1).mean()
    ma_long = loss_series.rolling(window=100, min_periods=1).mean()
    ma_combined = ma_short.where(loss_series.index < 500, ma_long)

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Loss (Bruto)", color='C0', alpha=0.3)
    plt.plot(ma_combined, label="Media Móvil (Tendencia)", color='C1', linewidth=2)
    
    title_text = (
        f"Entrenamiento: {model_type} con {opt_name}\n"
        f"Accuracy Final: {accuracy:.2f}% (Tiempo: {time_taken:.2f}s)"
    )
    plt.title(title_text)
    plt.xlabel("Pasos de entrenamiento (Batches)")
    plt.ylabel("Cross-Entropy Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # 2. Guarda el gráfico en la ruta raíz /results
    plot_filename = f"plot_{model_type.lower()}_{opt_name.lower()}_{int(time.time())}.png"
    plot_path = os.path.join(RESULTS_DIR, plot_filename) # <--- CAMBIO DE RUTA
    
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Gráfico guardado en: {plot_path}")
    return plot_path


# --- (save_model MODIFICADO para guardar en /models y escribir JSON) ---
def save_model(model, config, accuracy, training_time):
    """
    Guarda el state_dict del modelo en /models y actualiza models_metadata.json
    """
    # 1. Asegura que la carpeta raíz /models exista
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    timestamp = int(time.time())
    
    # 2. Define nombres y rutas
    model_id = f"model_{config.model_type.lower()}_{config.optimizer.lower()}_{timestamp}"
    model_filename = f"{model_id}.pth"
    model_path = os.path.join(MODELS_DIR, model_filename)
    metadata_json_path = os.path.join(MODELS_DIR, "models_metadata.json")

    # 3. Guarda el archivo .pth del modelo
    torch.save(model.state_dict(), model_path)
    print(f"Artefacto del modelo guardado en: {model_path}")

    # 4. Prepara la nueva entrada de metadatos
    new_metadata = {
        "id": model_id,
        "filename": model_filename,
        "timestamp": timestamp,
        "model_type": config.model_type,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "hidden_layers_config": config.hidden_layers_config,
        "accuracy": accuracy,
        "training_time": training_time
    }

    # 5. Lee el JSON existente, añade la nueva entrada y re-escribe
    metadata_list = []
    if os.path.exists(metadata_json_path):
        try:
            with open(metadata_json_path, 'r') as f:
                metadata_list = json.load(f)
                if not isinstance(metadata_list, list): # Asegura que sea una lista
                    metadata_list = []
        except json.JSONDecodeError:
            metadata_list = [] # El archivo está corrupto, empezar de nuevo

    metadata_list.append(new_metadata)

    try:
        with open(metadata_json_path, 'w') as f:
            json.dump(metadata_list, f, indent=4)
        print(f"Metadatos actualizados en: {metadata_json_path}")
    except Exception as e:
        print(f"Error al escribir metadatos JSON: {e}")

    return model_path