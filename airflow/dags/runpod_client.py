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
    }

    print(f"Payload envoyé: {payload}")

    response = requests.post(
        f"{RUNPOD_BASE_URL}/pods",
        headers=_headers(),
        json=payload,
        timeout=60,
    )

    print(f"Status création pod: {response.status_code}")
    print(f"Réponse création pod: {response.text}")

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
    data = response.json()
    print(f"Etat brut du pod {pod_id}: {data}")
    return data


def wait_until_job_finishes(pod_id: str, timeout_seconds: int = 7200) -> dict:
    """
    Attend que le conteneur ait réellement fini son exécution.
    On ne s'arrête PAS quand le pod devient RUNNING.
    """
    start = time.time()
    has_been_running = False

    while time.time() - start < timeout_seconds:
        pod = get_pod(pod_id)

        desired_status = pod.get("desiredStatus")
        status = pod.get("status")
        container_status = pod.get("containerStatus")

        print(
            f"[WAIT] pod_id={pod_id} | "
            f"desiredStatus={desired_status} | "
            f"status={status} | "
            f"containerStatus={container_status}"
        )

        # Le pod a bien démarré à un moment donné
        if status == "RUNNING" or desired_status == "RUNNING":
            has_been_running = True
            print("Le pod est en cours d'exécution...")

        # Une fois qu'il a été RUNNING, on attend qu'il quitte cet état
        if has_been_running and status not in [None, "RUNNING"]:
            print(f"Le pod a quitté l'état RUNNING avec status={status}")
            return pod

        time.sleep(15)

    raise TimeoutError(f"Le pod {pod_id} n'a pas terminé dans le délai imparti.")


def terminate_pod(pod_id: str) -> None:
    print(f"Suppression du pod {pod_id} ...")
    response = requests.delete(
        f"{RUNPOD_BASE_URL}/pods/{pod_id}",
        headers=_headers(),
        timeout=60,
    )
    print(f"Status suppression: {response.status_code}")
    print(f"Réponse suppression: {response.text}")
    response.raise_for_status()