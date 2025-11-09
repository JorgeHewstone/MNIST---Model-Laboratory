from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import asyncio
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Importamos las *clases* de modelo y las *funciones* de utilidad
from model import SimpleNN, ConvNN
from data_loader import get_data_loaders
from train import evaluate_model, save_loss_plot, save_model

# Modelo Pydantic para la configuración (sin cambios)
class TrainConfig(BaseModel):
    model_type: str
    optimizer: str
    learning_rate: float
    num_epochs: int
    batch_size: int
    num_layers: int
    hidden_layers_config: list[int]

app = FastAPI()

@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    """
    Este endpoint maneja el entrenamiento completo a través de WebSocket.
    """
    await websocket.accept()
    
    try:
        # 1. Esperar la configuración del cliente
        json_data = await websocket.receive_json()
        config = TrainConfig(**json_data)

        # Preparar el entorno de entrenamiento
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        BATCH_SIZE = config.batch_size
        NUM_EPOCHS = config.num_epochs

        await websocket.send_json({"type": "log", "message": f"Usando dispositivo: {DEVICE}"})
        
        # 2. Cargar datos
        await websocket.send_json({"type": "log", "message": "Cargando dataset MNIST..."})
        train_loader, test_loader = get_data_loaders(BATCH_SIZE)

        # 3. Instanciar Modelo
        await websocket.send_json({"type": "log", "message": f"Instanciando modelo: {config.model_type}"})
        if config.model_type == "SimpleNN":
            model_config = {"hidden_structure": config.hidden_layers_config}
            model = SimpleNN(**model_config)
        elif config.model_type == "ConvNN":
            model_config = {"linear_hidden": config.hidden_layers_config}
            config_cnn = {
                "conv_channels": [16, 32], # Default
                "linear_hidden": model_config.get("linear_hidden", [128])
            }
            model = ConvNN(**config_cnn)
        model.to(DEVICE)

        # 4. Instanciar Optimizador y Loss
        criterion = nn.CrossEntropyLoss()
        opt_name = config.optimizer
        opt_params = {"lr": config.learning_rate}
        
        if opt_name == "SGD":
            opt_params["momentum"] = 0.9
            optimizer = optim.SGD(model.parameters(), **opt_params)
        elif opt_name == "Adam":
            optimizer = optim.Adam(model.parameters(), **opt_params)
        elif opt_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), **opt_params)
        else: # AdamW
            optimizer = optim.AdamW(model.parameters(), **opt_params)

        # --- 5. Bucle de Entrenamiento (Lógica de train_model aquí) ---
        await websocket.send_json({"type": "log", "message": f"--- Iniciando entrenamiento con {opt_name} ---"})
        model.train()
        start_time = time.time()
        
        all_losses = []
        global_step = 0

        for epoch in range(NUM_EPOCHS):
            await websocket.send_json({"type": "log", "message": f"Epoch [{epoch+1}/{NUM_EPOCHS}]..."})
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                all_losses.append(current_loss)

                # --- Envío de datos en tiempo real ---
                # No enviar en cada paso (demasiado rápido), enviar cada 10 pasos
                if (i+1) % 10 == 0:
                    await websocket.send_json({
                        "type": "loss",
                        "step": global_step,
                        "loss": current_loss
                    })
                await asyncio.sleep(0)
                global_step += 1
        
        total_train_time = time.time() - start_time
        await websocket.send_json({"type": "log", "message": "Entrenamiento finalizado."})

        # --- 6. Evaluación y Guardado ---
        await websocket.send_json({"type": "log", "message": "Evaluando en test set..."})
        accuracy = evaluate_model(model, test_loader, DEVICE)
        
        await websocket.send_json({"type": "log", "message": "Guardando resultados..."})
        plot_path = save_loss_plot(all_losses, config.model_type, opt_name, accuracy, total_train_time)
        model_path = save_model(model, config, accuracy, total_train_time)
        # --- 7. Enviar resultado final ---
        await websocket.send_json({
            "type": "result",
            "accuracy": accuracy,
            "training_time": total_train_time,
            "plot_path": f"/results/{os.path.basename(plot_path)}",
            "model_path": f"/models/{os.path.basename(model_path)}",
            "model_type": config.model_type,
            "optimizer": config.optimizer
        })

    except WebSocketDisconnect:
        print("Cliente desconectado")
    except Exception as e:
        # Enviar error al cliente antes de cerrar
        print(f"Error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        await websocket.close()


# Servir la carpeta 'results' (para los gráficos)
app.mount("/results", StaticFiles(directory="results"), name="results")

@app.get("/main.js")
async def get_main_js():
    """Sirve el archivo JavaScript principal."""
    return FileResponse("main.js")
# Servir el panel de control HTML
@app.get("/")
async def get_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print("Iniciando servidor en http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)