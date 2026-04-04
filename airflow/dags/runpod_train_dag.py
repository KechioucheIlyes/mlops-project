import os
from datetime import datetime, timedelta
from pathlib import Path

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

from runpod_client import (
    create_runpod_pod,
    terminate_pod,
    wait_until_job_finishes,
)


def create_pod_task(**context):
    run_name = f"shifaa_airflow_{context['ts_nodash']}"
    pod_id = create_runpod_pod(run_name=run_name)

    context["ti"].xcom_push(key="runpod_pod_id", value=pod_id)
    context["ti"].xcom_push(key="mlflow_run_name", value=run_name)

    print(f"Pod créé: {pod_id}")
    print(f"RUN_NAME injecté dans le pod: {run_name}")


def wait_for_training_task(**context):
    pod_id = context["ti"].xcom_pull(
        key="runpod_pod_id",
        task_ids="create_runpod_pod",
    )

    final_pod_state = wait_until_job_finishes(pod_id)
    context["ti"].xcom_push(key="runpod_final_state", value=final_pod_state)


def register_model_task(**context):
    ti = context["ti"]

    run_name = ti.xcom_pull(
        key="mlflow_run_name",
        task_ids="create_runpod_pod",
    )

    tracking_uri = os.environ["AIRFLOW_MLFLOW_TRACKING_URI"].rstrip("/")
    experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
    registry_api_url = os.environ["AIRFLOW_REGISTRY_API_URL"].rstrip("/")
    registry_api_token = os.environ["REGISTRY_API_TOKEN"]

    # 1) Récupérer l'expérience MLflow par nom
    exp_response = requests.get(
        f"{tracking_uri}/api/2.0/mlflow/experiments/get-by-name",
        params={"experiment_name": experiment_name},
        timeout=30,
    )
    print(f"Experiment lookup status: {exp_response.status_code}")
    print(f"Experiment lookup body: {exp_response.text}")
    exp_response.raise_for_status()

    experiment = exp_response.json().get("experiment")
    if not experiment:
        raise ValueError(f"Expérience MLflow introuvable: {experiment_name}")

    experiment_id = experiment["experiment_id"]

    # 2) Chercher le run via run_name
    search_payload = {
        "experiment_ids": [experiment_id],
        "filter": f"tags.mlflow.runName = '{run_name}'",
        "max_results": 1,
        "order_by": ["attributes.start_time DESC"],
    }

    runs_response = requests.post(
        f"{tracking_uri}/api/2.0/mlflow/runs/search",
        json=search_payload,
        timeout=30,
    )
    print(f"Runs search status: {runs_response.status_code}")
    print(f"Runs search body: {runs_response.text}")
    runs_response.raise_for_status()

    runs = runs_response.json().get("runs", [])
    if not runs:
        raise ValueError(f"Aucun run MLflow trouvé pour run_name={run_name}")

    run = runs[0]
    run_id = run["info"]["run_id"]
    candidate_id = run_id

    print(f"Run MLflow trouvé: run_id={run_id}, experiment_id={experiment_id}")

    artifacts_root = (
        Path("/opt/mlops-storage/mlflow/artifacts")
        / str(experiment_id)
        / run_id
        / "artifacts"
    )

    results_path = artifacts_root / "results" / "results.json"
    final_model_full_path = artifacts_root / "checkpoints" / "final_model_full.pth"
    best_model_path = artifacts_root / "checkpoints" / "best_model.pth"
    final_model_path = artifacts_root / "checkpoints" / "final_model.pth"

    required_files = [results_path, final_model_full_path]
    for required_file in required_files:
        if not required_file.exists():
            raise FileNotFoundError(f"Fichier requis introuvable: {required_file}")

    headers = {
        "Authorization": f"Bearer {registry_api_token}",
    }

    with open(results_path, "rb") as results_file, open(final_model_full_path, "rb") as final_model_full_file:
        files = {
            "results_file": ("results.json", results_file, "application/json"),
            "final_model_full_file": ("final_model_full.pth", final_model_full_file, "application/octet-stream"),
        }

        optional_opened_files = []

        try:
            if best_model_path.exists():
                best_model_file = open(best_model_path, "rb")
                optional_opened_files.append(best_model_file)
                files["best_model_file"] = ("best_model.pth", best_model_file, "application/octet-stream")

            if final_model_path.exists():
                final_model_file = open(final_model_path, "rb")
                optional_opened_files.append(final_model_file)
                files["final_model_file"] = ("final_model.pth", final_model_file, "application/octet-stream")

            upload_response = requests.post(
                f"{registry_api_url}/upload-model",
                headers=headers,
                data={
                    "run_name": run_name,
                    "candidate_id": candidate_id,
                },
                files=files,
                timeout=300,
            )
            print(f"Upload response status: {upload_response.status_code}")
            print(f"Upload response body: {upload_response.text}")
            upload_response.raise_for_status()

        finally:
            for f in optional_opened_files:
                f.close()

    promote_response = requests.post(
        f"{registry_api_url}/promote-model",
        headers={
            "Authorization": f"Bearer {registry_api_token}",
            "Content-Type": "application/json",
        },
        json={"candidate_id": candidate_id},
        timeout=120,
    )
    print(f"Promote response status: {promote_response.status_code}")
    print(f"Promote response body: {promote_response.text}")
    promote_response.raise_for_status()

    ti.xcom_push(key="mlflow_run_id", value=run_id)
    ti.xcom_push(key="registry_candidate_id", value=candidate_id)


def terminate_pod_task(**context):
    pod_id = context["ti"].xcom_pull(
        key="runpod_pod_id",
        task_ids="create_runpod_pod",
    )
    if pod_id:
        terminate_pod(pod_id)


default_args = {
    "owner": "airflow",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="runpod_train_service",
    default_args=default_args,
    start_date=datetime(2026, 3, 21),
    schedule=None,
    catchup=False,
    tags=["mlops", "runpod", "training"],
) as dag:
    create_runpod_pod_op = PythonOperator(
        task_id="create_runpod_pod",
        python_callable=create_pod_task,
    )

    wait_for_training_op = PythonOperator(
        task_id="wait_for_training_completion",
        python_callable=wait_for_training_task,
    )

    register_model_op = PythonOperator(
        task_id="register_model",
        python_callable=register_model_task,
    )

    terminate_pod_op = PythonOperator(
        task_id="terminate_pod",
        python_callable=terminate_pod_task,
        trigger_rule="all_done",
    )

    create_runpod_pod_op >> wait_for_training_op >> register_model_op >> terminate_pod_op