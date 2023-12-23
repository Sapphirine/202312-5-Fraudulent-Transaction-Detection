from datetime import datetime, timedelta
from textwrap import dedent
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime as dt
import numpy as np
import os
from pandas import DataFrame as df

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
# These args will get passed on to each operator
# You can override thedef model_comaparison(**kwargs)


####################################################
# DEFINE PYTHON FUNCTIONS
####################################################

def model_comaparison(**kwargs):
    ti = kwargs['ti']


    stats = {
        'logReg' : ti.xcom_pull(task_ids='Logistic_Regression').split(','),
        'KNN' : ti.xcom_pull(task_ids='KNN').split(','),
        'SVC' : ti.xcom_pull(task_ids='SVC').split(','),
        'NaiveBayes' : ti.xcom_pull(task_ids='Naive_Bayes').split(','),
        'MLP' : ti.xcom_pull(task_ids='MLP').split(','),
        'RF' : ti.xcom_pull(task_ids='RF').split(','),
        'xGB' : ti.xcom_pull(task_ids='xGB').split(','),
    }

    rdf = df.from_dict(stats, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1'])
    rdf.to_csv('/home/ggvkulkarni/proj/data/results.csv')
    print(rdf)

############################################
# DEFINE AIRFLOW DAG (SETTINGS + SCHEDULE)
############################################

default_args = {
    'owner': 'Vishnu Kulkarni',
    'depends_on_past': False,
    'email': ['vk2496@columbia.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(days=1),
}

with DAG(
    'frauddetection',
    default_args=default_args,
    description='A DAG for our fraud detection project.',
    schedule_interval=timedelta(hours=24), #run every 24 hours
    start_date=datetime(2023, 12, 1),
    catchup=False,
    tags=['example'],
) as dag:

##########################################
# DEFINE AIRFLOW OPERATORS
##########################################

    # t* examples of tasks created by instantiating operators


    t1 = BashOperator(
        task_id='Data_Fetching_and_Preparation',
        bash_command='python /home/ggvkulkarni/proj/scripts/data.py',
        retries=3,
    )

    t2 = BashOperator(
        task_id='Logistic_Regression',
        bash_command='python /home/ggvkulkarni/proj/scripts/logReg.py',
        retries=3,
    )

    t3 = BashOperator(
        task_id='KNN',
        bash_command='python /home/ggvkulkarni/proj/scripts/KNN.py',
        retries=3,
    )

    t4 = BashOperator(
        task_id='SVC',
        bash_command='python /home/ggvkulkarni/proj/scripts/SVC.py',
        retries=3,
    )

    t5 = BashOperator(
        task_id='Naive_Bayes',
        bash_command='python /home/ggvkulkarni/proj/scripts/NaiveBayes.py',
        retries=3,
    )

    t6 = BashOperator(
        task_id='MLP',
        bash_command='python /home/ggvkulkarni/proj/scripts/MLP.py',
        retries=3,
    )

    t7 = BashOperator(
        task_id='RF',
        bash_command='python /home/ggvkulkarni/proj/scripts/RF.py',
        retries=3,
    )

    t8 = BashOperator(
        task_id='xGB',
        bash_command='python /home/ggvkulkarni/proj/scripts/xGB.py',
        retries=3,
    )

    t9 = PythonOperator(
        task_id='Model_Comparison',
        python_callable=model_comaparison,
        retries=3,
    )


##########################################
# DEFINE TASKS HIERARCHY
##########################################


    t1 >> [t2, t3, t4, t5, t6, t7, t8]
    [t2, t3, t4, t5, t6, t7, t8] >> t9
