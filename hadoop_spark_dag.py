from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 24),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('hadoop_spark_pipeline', default_args=default_args, schedule_interval='@daily') as dag:
    spark_job = BashOperator(
        task_id='run_spark_job',
        bash_command='spark-submit --master local /user/dylan/word_counts.parquet',
    )

