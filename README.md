# MNIST Lab — Realtime Training & In-Browser Inference

A small, full-stack ML playground to **train** MNIST classifiers in real-time and **test** them from a browser canvas.
The project is split into two FastAPI services:

* **`training_service`** (port **8000**): trains models over WebSocket, streams loss, supports **Abort**, **Dropout**, **Duplicate detection + overwrite**, and persists artifacts + metadata.
* **`inference_service`** (port **8001**): a drawing UI that finds a **matching trained model** by configuration and runs inference locally.

---

## Features

* **Two model families**

  * `SimpleNN` (MLP with customizable hidden layers)
  * `ConvNN` (2 conv blocks + linear head, customizable linear layers)
* **Realtime training via WebSocket**

  * Step loss updates, epoch logs
  * **Abort training** from the UI
  * Duplicate-config detection → **prompt to overwrite** or keep existing
* **Regularization**

  * Optional **Dropout** (p = 0.3) for both models (applied only in training)
* **Model discovery for inference**

  * The test module selects a model by **exact config match** (type, optimizer, batch size, hidden layers, epochs, `use_dropout`) and **learning-rate range** (“Low…High”).
* **Artifacts & metadata**

  * Saves `.pth` files into `/models`
  * Appends metadata to `/models/models_metadata.json`:

    ```json
    {
      "id": "model_simplenn_adam_1762740520",
      "filename": "model_simplenn_adam_1762740520.pth",
      "timestamp": 1762740520,
      "model_type": "SimpleNN",
      "optimizer": "Adam",
      "learning_rate": 0.0055,
      "batch_size": 32,
      "num_epochs": 1,
      "hidden_layers_config": [128],
      "use_dropout": false,
      "dropout_p": 0.3,
      "accuracy": 94.1,
      "training_time": 36.86
    }
    ```

---

## Project Structure

```
.
├─ data/                        # Root dataset cache (MNIST)
│  └─ MNIST/raw/                # torchvision files
├─ inference_service/           # Test module (port 8001)
│  ├─ main.py                   # FastAPI app (UI + /predict)
│  ├─ templates/index.html      # Drawing UI
│  └─ static/js/drawing.js      # Canvas + parameter sync
├─ training_service/            # Training module (port 8000)
│  ├─ api.py                    # FastAPI app + /ws/train (WebSocket)
│  ├─ data_loader.py            # MNIST loaders
│  ├─ model.py                  # SimpleNN & ConvNN (+ optional dropout)
│  ├─ train.py                  # evaluate_model, save_model (writes metadata)
│  ├─ templates/index.html      # Training panel UI
│  └─ static/js/main.js         # WebSocket client, chart, Abort, overwrite flow
├─ models/                      # Saved models + metadata.json
├─ results/                     # Training plots (optional)
├─ scripts/run_dev.py           # Launch both services (8000, 8001)
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## Requirements

```
torch
torchvision
numpy
matplotlib
fastapi
uvicorn[standard]
jinja2
pillow
python-multipart     # FastAPI file/form handling (future-safe)
```

> On Windows you’ll also want `pip install watchfiles` if you enable `--reload` locally.

---

## Quick Start

### 1) Install deps

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

### 2) Launch both services

```bash
python scripts/run_dev.py
```

* **Training module** → [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Test module** → [http://127.0.0.1:8001](http://127.0.0.1:8001)

> Windows note: `run_dev.py` uses `multiprocessing` with `spawn`. Avoid nested reloaders; the script starts uvicorn without `--reload`.

---

## How It Works

### Training flow (port 8000)

* Open the **Training Panel** at `http://127.0.0.1:8000/`.
* Configure:

  * Model Type: `SimpleNN` or `ConvNN`
  * Optimizer: `Adam`, `SGD`, `RMSprop`, `AdamW`
  * Learning rate (numeric; derived midpoint from “range” if you use the dropdown)
  * Epochs, Batch Size
  * Hidden layers: variable number & sizes
  * **Use dropout** (p = 0.3)
* Click **Start Training**. Over WebSocket:

  * Server may send `{"type":"exists"}` if same config already trained → UI prompts **Overwrite?**
  * Streamed messages:

    * `{"type":"log","message":...}`
    * `{"type":"loss","step":int,"loss":float}` every ~10 steps
    * `{"type":"aborted", ...}` if you clicked **Abort**
    * `{"type":"result", ...}` with accuracy, paths on completion
* Artifacts saved to `/models`, metadata in `/models/models_metadata.json`.

**Estimating training steps (MNIST):**
`num_steps = ceil(60000 / batch_size) * num_epochs`
(60k training images in MNIST train split.)

### Inference flow (port 8001)

* Open the **MNIST Simulator** at `http://127.0.0.1:8001/`.
* Set the **same parameters** used in training:

  * Type, optimizer, batch size, **hidden layers**, **epochs**, **use_dropout**
  * Select a **Learning Rate Range**:

    * `Low` (1e-5 to 1e-4)
    * `Medium Low` (1e-4 to 5e-4)
    * `Medium High` (5e-4 to 1e-3)
    * `High` (1e-3 to 1e-2)
* Draw a digit and click **Predict**.
* The app searches metadata for an **exact match** on core params + **LR within the chosen range**. If found, it loads the `.pth` and runs inference. If not, you’ll see **“Model not trained yet.”** plus a CTA to **Train this configuration** (the link pre-fills the training UI using query params).

---

## APIs

### Training Service (WebSocket)

`ws://127.0.0.1:8000/ws/train`

* **Client → Server** (first message): JSON `TrainConfig`

  ```json
  {
    "model_type": "SimpleNN",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "num_epochs": 5,
    "batch_size": 64,
    "num_layers": 2,
    "hidden_layers_config": [128, 64],
    "use_dropout": true,
    "dropout_p": 0.3
  }
  ```

* **Server → Client** messages:

  * `{"type":"exists", "message":"...", "existing":{...}}`
    → Client must reply once with: `{"type":"overwrite","value":true|false}`
  * `{"type":"log","message":"..."}`
  * `{"type":"loss","step":123,"loss":0.456}`
  * `{"type":"aborted","message":"Training aborted by user."}`
  * `{"type":"result","accuracy":94.1,"training_time":36.8,"model_path":"/models/..","model_type":"...","optimizer":"..." }`
  * `{"type":"error","message":"..."}`

* **Abort** (Client → Server at any time):
  `{"type":"abort"}`

### Inference Service

* **GET `/`** → HTML drawing UI
* **POST `/predict`**

  ```json
  {
    "params": {
      "model_type": "SimpleNN",
      "optimizer": "Adam",
      "learning_rate_range": "High",
      "batch_size": 64,
      "epochs": 5,
      "hidden_layers_config": [128, 64],
      "use_dropout": true
    },
    "image_data": "data:image/png;base64,..."  // 28x28 implicit; app resizes/inverts
  }
  ```

  **Response**

  ```json
  {
    "prediction": 7,
    "confidence": "98.12%",
    "found_model": {
      "id": "model_simplenn_adam_1762740520",
      "accuracy": "94.10%",
      "learning_rate": 0.0055,
      "epochs": 5,
      "training_time": "36.87s"
    }
  }
  ```

  or

  ```json
  { "error": "Model not trained yet." }
  ```

---

## Development Tips

* **Branching**: create a new branch for releases, e.g.:

  ```bash
  git checkout -b version2.0
  git add .
  git commit -m "feat: split training/inference services, dropout, abort, overwrite, metadata"
  git push -u origin version2.0
  ```
* **Static paths**: Both services mount `/static` and use `templates/` relative to their own service folder.
* **Cross-linking**: The test UI builds a URL to the training UI (port 8000) with query params; the training UI shows **“Test this model”** after success, linking back to port 8001 with the same params.
* **Windows multiprocessing**: `scripts/run_dev.py` uses `spawn`. Don’t use uvicorn’s auto-reloader inside those child processes.

---

## Troubleshooting

* **“Model not trained yet.” even after training**

  * Ensure both services point to the **same `models/models_metadata.json`** and `.pth` directory (repo root).
  * The inference match requires exact: `model_type`, `optimizer`, `batch_size`, `hidden_layers_config`, `epochs`, `use_dropout` **and** LR inside the chosen range.
* **WebSocket error “Cannot call send after close”**

  * Happens if the endpoint tries to `send` after a `close`. We guard against this; if you customize, ensure you `return` immediately after closing and cancel any listener tasks.
* **“No module named 'model'”**

  * Use repo-root on `sys.path` (done in code) or run from project root. Don’t change working dirs arbitrarily.
* **Port already in use**

  * Stop lingering uvicorn processes or change ports in `scripts/run_dev.py`.

---

## Extending

* Add more models to `training_service/model.py` (keep consistent constructor args: `use_dropout`, `dropout_p`).
* Add more **LR buckets** in `inference_service/main.py` (`LR_RANGES`).
* Save extra training stats in metadata (e.g., per-epoch loss arrays) and visualize them in the UIs.

---

## License

MIT (or your preferred license).
