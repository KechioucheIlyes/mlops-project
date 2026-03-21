from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

from runpod_client import create_runpod_pod, wait_until_running, terminate_pod


def create_pod_task(**context):
    pod_id = create_runpod_pod()
    context["ti"].xcom_push(key="runpod_pod_id", value=pod_id)


def wait_pod_task(**context):
    pod_id = context["ti"].xcom_pull(key="runpod_pod_id", task_ids="create_runpod_pod")
    wait_until_running(pod_id)


def terminate_pod_task(**context):
    pod_id = context["ti"].xcom_pull(key="runpod_pod_id", task_ids="create_runpod_pod")
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

    wait_until_running_op = PythonOperator(
        task_id="wait_until_running",
        python_callable=wait_pod_task,
    )

    terminate_pod_op = PythonOperator(
        task_id="terminate_pod",
        python_callable=terminate_pod_task,
        trigger_rule="all_done",
    )

    create_runpod_pod_op >> wait_until_running_op >> terminate_pod_op