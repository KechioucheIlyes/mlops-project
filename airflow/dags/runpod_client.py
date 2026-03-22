import os
import time
import requests


RUNPOD_BASE_URL = "https://rest.runpod.io/v1"


def _headers() -> dict:
    api_key = os.environ["RUNPOD_API_KEY"]
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def build_env_payload() -> dict:
    keys = [
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "DATASET_SLUG",
        "DATA_DIR",
        "PREPROCESSED_DIR",
        "OUTPUT_DIR",
        "MODEL_DIR",
        "PLOTS_DIR",
        "RUN_NAME",
        "BATCH_SIZE",
        "EPOCHS",
        "LEARNING_RATE",
        "IMG_SIZE",
        "EARLY_STOPPING_PATIENCE",
        "LR_SCHEDULER_PATIENCE",
        "LR_SCHEDULER_FACTOR",
        "MIN_LR",
        "RANDOM_STATE",
        "REGISTRY_API_URL",
        "REGISTRY_API_TOKEN",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}

def create_runpod_pod() -> str:
    payload = {
        "name": "train-service-airflow",
        "cloudType": os.getenv("RUNPOD_CLOUD_TYPE", "COMMUNITY"),
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": [os.getenv("RUNPOD_GPU_TYPE", "NVIDIA GeForce RTX 4090")],
        "gpuTypePriority": "availability",
        "imageName": os.environ["RUNPOD_IMAGE"],
        "env": build_env_payload(),
        "containerDiskInGb": int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "20")),
        "volumeInGb": int(os.getenv("RUNPOD_VOLUME_GB", "40")),
        "volumeMountPath": "/workspace",
        "dockerEntrypoint": [],
        "dockerStartCmd": [],
    }

    print(f"Payload envoyé: {payload}")

    response = requests.post(
        f"{RUNPOD_BASE_URL}/pods",
        headers=_headers(),
        json=payload,
        timeout=60,
    )
    
    print(f"Status: {response.status_code}")
    print(f"Réponse: {response.text}")
    
    response.raise_for_status()
    data = response.json()
    return data["id"]

def get_pod(pod_id: str) -> dict:
    response = requests.get(
        f"{RUNPOD_BASE_URL}/pods/{pod_id}",
        headers=_headers(),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def wait_until_running(pod_id: str, timeout_seconds: int = 900) -> dict:
    start = time.time()
    while time.time() - start < timeout_seconds:
        pod = get_pod(pod_id)
        desired = pod.get("desiredStatus")
        actual = pod.get("status")
        if desired == "RUNNING" or actual == "RUNNING":
            return pod
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} did not become RUNNING in time.")


def terminate_pod(pod_id: str) -> None:
    response = requests.delete(
        f"{RUNPOD_BASE_URL}/pods/{pod_id}",
        headers=_headers(),
        timeout=60,
    )
    response.raise_for_status()