from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from runpod_client import (
    create_runpod_pod,
    wait_until_job_finishes,
    terminate_pod,
)


def create_pod_task(**context):
    pod_id = create_runpod_pod()
    context["ti"].xcom_push(key="runpod_pod_id", value=pod_id)


def wait_for_training_task(**context):
    pod_id = context["ti"].xcom_pull(
        key="runpod_pod_id",
        task_ids="create_runpod_pod",
    )

    final_pod_state = wait_until_job_finishes(pod_id)
    context["ti"].xcom_push(key="runpod_final_state", value=final_pod_state)


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

    terminate_pod_op = PythonOperator(
        task_id="terminate_pod",
        python_callable=terminate_pod_task,
        trigger_rule="all_done",
    )

    create_runpod_pod_op >> wait_for_training_op >> terminate_pod_op