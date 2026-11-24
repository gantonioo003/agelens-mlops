"""Scheduled operational inference and independent preparation of labeled train data."""

from datetime import UTC, datetime, timedelta

from airflow.sdk import dag, task

from services.orchestration.operational import process_pending
from services.orchestration.training_data import prepare_training

START = datetime(2025, 1, 1, tzinfo=UTC)


@dag(
    dag_id="agelens_operational_inference",
    start_date=START,
    schedule="* * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["agelens", "inference"],
)
def operational_inference():
    @task(retries=1, retry_delay=timedelta(seconds=20), execution_timeout=timedelta(minutes=6))
    def pump():
        return process_pending()

    pump()


@dag(
    dag_id="agelens_prepare_labeled_train",
    start_date=START,
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["agelens", "train-data"],
)
def labeled_train():
    @task(retries=1, retry_delay=timedelta(seconds=20), execution_timeout=timedelta(minutes=3))
    def materialize():
        return prepare_training()

    materialize()


operational_inference()
labeled_train()
