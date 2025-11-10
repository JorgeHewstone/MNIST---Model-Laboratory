# scripts/run_dev.py
import multiprocessing as mp
import uvicorn
import os
import sys
from time import sleep

def project_root():
    # scripts/ is one level below repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_inference():
    """
    Runs the inference/test module (inference_service/) on port 8001
    """
    sys.path.append(project_root())
    uvicorn.run(
        "inference_service.main:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
        workers=1
    )

def run_training():
    """
    Runs the training module (training_service/) on port 8000
    """
    sys.path.append(project_root())
    uvicorn.run(
        "training_service.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=1
    )

if __name__ == "__main__":
    # On Windows, protect the entry point and use spawn
    mp.set_start_method("spawn", force=True)

    p_train = mp.Process(target=run_training)
    p_infer = mp.Process(target=run_inference)

    p_train.start()
    # small stagger to avoid mixed logs
    sleep(0.5)
    p_infer.start()

    print("Both services are starting...")
    print("Training module  ➜ http://127.0.0.1:8000/")
    print("Test module      ➜ http://127.0.0.1:8001/")

    try:
        p_train.join()
        p_infer.join()
    except KeyboardInterrupt:
        print("Shutting down...")
        p_train.terminate()
        p_infer.terminate()
