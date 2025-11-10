import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import json  # <--- Import JSON

# --- Define root paths (one level ABOVE this file) ---
# os.path.dirname(__file__) is the 'src' folder
# os.path.abspath(os.path.join(..., "..")) goes up one level
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# --- (evaluate_model remains unchanged) ---
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





# --- (save_model MODIFIED to save in /models and write JSON) ---
def save_model(model, config, accuracy, training_time, models_dir=None ):
    """
    Saves the model's state_dict in /models and updates models_metadata.json
    """
    # 1. Ensure the root /models folder exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    timestamp = int(time.time())
    
    # 2. Define names and paths
    model_id = f"model_{config.model_type.lower()}_{config.optimizer.lower()}_{timestamp}"
    model_filename = f"{model_id}.pth"
    if models_dir:
        model_path = os.path.join(models_dir, model_filename) 
    else:model_path= os.path.join(MODELS_DIR, model_filename)
    metadata_json_path = os.path.join(MODELS_DIR, "models_metadata.json")

    # 3. Save the model's .pth file
    torch.save(model.state_dict(), model_path)
    print(f"Model artifact saved at: {model_path}")

    # 4. Prepare the new metadata entry
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
        "training_time": training_time,
        "use_dropout": bool(getattr(config, "use_dropout", False)),  # <-- add this
        "dropout_p": float(getattr(config, "dropout_p", 0.3)),       # <-- optional
    }

    # 5. Read the existing JSON, add the new entry, and rewrite
    metadata_list = []
    if os.path.exists(metadata_json_path):
        try:
            with open(metadata_json_path, 'r') as f:
                metadata_list = json.load(f)
                if not isinstance(metadata_list, list): # Ensure it's a list
                    metadata_list = []
        except json.JSONDecodeError:
            metadata_list = [] # File is corrupt, start over

    metadata_list.append(new_metadata)

    try:
        with open(metadata_json_path, 'w') as f:
            json.dump(metadata_list, f, indent=4)
        print(f"Metadata updated in: {metadata_json_path}")
    except Exception as e:
        print(f"Error writing metadata JSON: {e}")

    return model_path