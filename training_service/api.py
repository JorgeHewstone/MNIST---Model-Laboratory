from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from starlette.websockets import WebSocketState

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
import contextlib
from contextlib import asynccontextmanager


# Import model *classes* and utility *functions*
from .model import SimpleNN, ConvNN
from .data_loader import get_data_loaders
from .train import evaluate_model, save_model
# -----------------------------
# Paths anchored to this file
# -----------------------------
# -----------------------------
# Paths anchored to this file
# -----------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))  # project root
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

TRAIN_STATIC_DIR = os.path.join(THIS_DIR, 'static')
TRAIN_TEMPLATES_DIR = os.path.join(THIS_DIR, 'templates')

# Pydantic model for configuration
class TrainConfig(BaseModel):
    model_type: str
    optimizer: str
    learning_rate: float
    num_epochs: int
    batch_size: int
    num_layers: int
    hidden_layers_config: list[int]
    use_dropout: bool = False 
    dropout_p: float = 0.3


import json
from typing import Any, Dict, List, Optional

METADATA_PATH = os.path.join(MODELS_DIR, "models_metadata.json")

def load_metadata() -> List[Dict[str, Any]]:
    if not os.path.exists(METADATA_PATH):
        return []
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_metadata(all_meta: List[Dict[str, Any]]) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

def same_config(a: Dict[str, Any], cfg: "TrainConfig") -> bool:
    """
    Define cuándo consideramos que 'ya existe' un modelo:
    - model_type, optimizer
    - learning_rate (numérico)
    - batch_size, num_epochs
    - hidden_layers_config (lista exacta)
    - use_dropout
    """
    return (
        a.get("model_type") == cfg.model_type and
        a.get("optimizer") == cfg.optimizer and
        float(a.get("learning_rate", -1)) == float(cfg.learning_rate) and
        int(a.get("batch_size", -1)) == int(cfg.batch_size) and
        int(a.get("num_epochs", -1)) == int(cfg.num_epochs) and
        list(a.get("hidden_layers_config", [])) == list(cfg.hidden_layers_config) and
        bool(a.get("use_dropout", False)) == bool(getattr(cfg, "use_dropout", False))
    )

def find_existing_metadata(cfg: "TrainConfig") -> Optional[Dict[str, Any]]:
    meta = load_metadata()
    for m in meta:
        if same_config(m, cfg):
            return m
    return None

def delete_model_and_metadata(existing: Dict[str, Any]) -> None:
    """
    Borra el .pth (si existe) y saca la entrada del metadata.
    No toca los plots por simplicidad (puedes añadirlo si quieres).
    """
    # 1) Delete model file
    filename = existing.get("filename")
    if filename:
        path = os.path.join(MODELS_DIR, filename)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[warn] Could not remove previous model file {path}: {e}")

    # 2) Remove entry from metadata
    meta = load_metadata()
    new_meta = [m for m in meta if m is not existing and m.get("id") != existing.get("id")]
    # Fallback by matching structure if id missing
    if len(meta) == len(new_meta):
        new_meta = [m for m in meta if not same_config(m, existing)]
    save_metadata(new_meta)


# -----------------------------
# Lifespan: ensure dirs exist
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[startup] RESULTS_DIR={RESULTS_DIR}")
    print(f"[startup] MODELS_DIR={MODELS_DIR}")
    try:
        yield
    finally:
        # Nothing special for shutdown now
        print("[shutdown] training API stopping...")

app = FastAPI(lifespan=lifespan)

# CORS (handy if you later fetch from 8001 to 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=TRAIN_STATIC_DIR), name="static")
# -----------------------------
# WebSocket training endpoint
# -----------------------------
@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    """
    WebSocket endpoint that handles the full training flow, with:
      - optional dropout support (use_dropout, dropout_p)
      - duplicate-config detection and overwrite handshake
    """
    await websocket.accept()

    try:
        # 1) Receive config
        json_data = await websocket.receive_json()
        config = TrainConfig(**json_data)

        # ---- Duplicate detection & overwrite handshake ----
        # Inline helpers (avoid external deps for this block)
        def _same_config(meta: dict, cfg: TrainConfig) -> bool:
            return (
                meta.get("model_type") == cfg.model_type and
                meta.get("optimizer") == cfg.optimizer and
                float(meta.get("learning_rate", -1)) == float(cfg.learning_rate) and
                int(meta.get("batch_size", -1)) == int(cfg.batch_size) and
                int(meta.get("num_epochs", -1)) == int(cfg.num_epochs) and
                list(meta.get("hidden_layers_config", [])) == list(cfg.hidden_layers_config) and
                bool(meta.get("use_dropout", False)) == bool(getattr(cfg, "use_dropout", False))
            )

        def _load_metadata() -> list[dict]:
            if not os.path.exists(METADATA_PATH):
                return []
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []

        def _save_metadata(all_meta: list[dict]) -> None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(all_meta, f, ensure_ascii=False, indent=2)

        def _find_existing(cfg: TrainConfig) -> dict | None:
            for m in _load_metadata():
                if _same_config(m, cfg):
                    return m
            return None

        def _delete_previous(existing: dict) -> None:
            # delete model file
            filename = existing.get("filename")
            if filename:
                path = os.path.join(MODELS_DIR, filename)
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    print(f"[warn] Could not remove previous model file {path}: {e}")
            # drop metadata entry
            meta = _load_metadata()
            # prefer id-based removal if available
            eid = existing.get("id")
            if eid is not None:
                new_meta = [m for m in meta if m.get("id") != eid]
            else:
                new_meta = [m for m in meta if not _same_config(m, existing)]
            _save_metadata(new_meta)

        existing = _find_existing(config)
        if existing:
            await websocket.send_json({
                "type": "exists",
                "message": "Model already trained with this configuration. Overwrite?",
                "existing": {
                    "id": existing.get("id"),
                    "accuracy": existing.get("accuracy"),
                    "learning_rate": existing.get("learning_rate"),
                    "batch_size": existing.get("batch_size"),
                    "num_epochs": existing.get("num_epochs"),
                    "use_dropout": existing.get("use_dropout", False),
                    "hidden_layers_config": existing.get("hidden_layers_config", [])
                }
            })
            try:
                reply = await websocket.receive_json()
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"Failed to receive overwrite decision: {e}"})
                await websocket.close()
                return

            if not (isinstance(reply, dict) and reply.get("type") == "overwrite"):
                await websocket.send_json({"type": "error", "message": "Protocol error: expected {'type':'overwrite', 'value':...}."})
                await websocket.close()
                return

            if not bool(reply.get("value", False)):
                await websocket.send_json({
                    "type": "result",
                    "skipped": True,
                    "message": "Training skipped by user (existing model kept)."
                })
                await websocket.close()
                return
            else:
                _delete_previous(existing)
                await websocket.send_json({"type": "log", "message": "Previous model deleted. Proceeding with training..."})

        # 2) Prepare training environment
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        BATCH_SIZE = config.batch_size
        NUM_EPOCHS = config.num_epochs

        await websocket.send_json({"type": "log", "message": f"Using device: {DEVICE}"})
        await websocket.send_json({"type": "log", "message": f"Config: Epochs={NUM_EPOCHS}, Batch Size={BATCH_SIZE}"})

        # 3) Load data
        await websocket.send_json({"type": "log", "message": "Loading MNIST dataset..."})
        train_loader, test_loader = get_data_loaders(BATCH_SIZE)

        # 4) Instantiate Model (with optional dropout)
        await websocket.send_json({
            "type": "log",
            "message": f"Instantiating model: {config.model_type} | dropout={'on' if getattr(config, 'use_dropout', False) else 'off'}"
        })
        dropout_p = getattr(config, "dropout_p", 0.3) or 0.3
        use_dropout = bool(getattr(config, "use_dropout", False))

        if config.model_type == "SimpleNN":
            model = SimpleNN(
                hidden_structure=config.hidden_layers_config,
                use_dropout=use_dropout,
                dropout_p=dropout_p
            )
        elif config.model_type == "ConvNN":
            model = ConvNN(
                conv_channels=[16, 32],  # default
                linear_hidden=config.hidden_layers_config,
                use_dropout=use_dropout,
                dropout_p=dropout_p
            )
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")

        model.to(DEVICE)

        # 5) Optimizer and Loss
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
        else:  # AdamW
            optimizer = optim.AdamW(model.parameters(), **opt_params)

            
        abort_event = asyncio.Event()

        async def control_listener():
            try:
                while True:
                    msg = await websocket.receive_json()
                    if isinstance(msg, dict):
                        if msg.get("type") == "abort":
                            await websocket.send_json({"type": "log", "message": "Abort requested by client."})
                            abort_event.set()
            except WebSocketDisconnect:
                abort_event.set()
            except Exception:
                # cualquier error en el canal de control aborta por seguridad
                abort_event.set()

        control_task = asyncio.create_task(control_listener())

        # 6) Training Loop
        await websocket.send_json({"type": "log", "message": f"--- Starting training with {opt_name} ---"})
        model.train()
        start_time = time.time()

        all_losses = []
        global_step = 0
        aborted = False
        for epoch in range(NUM_EPOCHS):
            await websocket.send_json({"type": "log", "message": f"Epoch [{epoch+1}/{NUM_EPOCHS}]..."} )



            for i, (inputs, labels) in enumerate(train_loader):
                if abort_event.is_set():
                    aborted = True
                    break
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                all_losses.append(current_loss)

                # Real-time updates (every 10 steps)
                if (i + 1) % 10 == 0:
                    await websocket.send_json({
                        "type": "loss",
                        "step": global_step,
                        "loss": current_loss
                    })
                    await asyncio.sleep(0)  # let the loop flush

                global_step += 1
            if aborted:
                break
        if aborted:
            await websocket.send_json({"type": "aborted", "message": "Training aborted by user."})
            return

        total_train_time = time.time() - start_time
        await websocket.send_json({"type": "log", "message": "Training finished."})

        # 7) Evaluation and Saving
        await websocket.send_json({"type": "log", "message": "Evaluating on test set..."})
        accuracy = evaluate_model(model, test_loader, DEVICE)

        await websocket.send_json({"type": "log", "message": "Saving results..."})
        model_path = save_model(model, config, accuracy, total_train_time, models_dir=MODELS_DIR)

        # 8) Final result
        await websocket.send_json({
            "type": "result",
            "accuracy": accuracy,
            "training_time": total_train_time,
            "model_path": f"/models/{os.path.basename(model_path)}",
            "model_type": config.model_type,
            "optimizer": config.optimizer
        })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        with contextlib.suppress(Exception):
            control_task.cancel()
        # Avoid closing twice
        try:
            if websocket.application_state != WebSocketState.DISCONNECTED:
                await websocket.close()
        except RuntimeError:
            # already closed
            pass




# -----------------------------
# Static file serving (absolute paths)
# -----------------------------
# Serve the root /results and /models folders (from project root)
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/models", StaticFiles(directory=MODELS_DIR), name="models")

@app.get("/main.js")
async def get_main_js():
    return FileResponse(os.path.join(TRAIN_STATIC_DIR, "js", "main.js"))


# Serve the HTML control panel
@app.get("/")
async def get_index():
    return FileResponse(os.path.join(TRAIN_TEMPLATES_DIR, "index.html"))


# Optional: simple health check
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
