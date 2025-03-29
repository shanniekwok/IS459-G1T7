from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.amazon.aws.operators.emr import (
    EmrCreateJobFlowOperator,
    EmrAddStepsOperator,
    EmrTerminateJobFlowOperator,
)
from airflow.providers.amazon.aws.sensors.emr import EmrStepSensor
from airflow.providers.amazon.aws.operators.glue_crawler import GlueCrawlerOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'emr_pyspark_dag',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# Define the EMR cluster configuration
JOB_FLOW_OVERRIDES = {
    'Name': 'smart-meters-emr-cluster',
    'ReleaseLabel': 'emr-7.8.0',
    'Applications': [
        {'Name': 'Hadoop'},
        {'Name': 'Spark'},
        {'Name': 'Hive'},
        {'Name': 'Livy'},
        {'Name': 'JupyterHub'},
        {'Name': 'JupyterEnterpriseGateway'},
        {'Name': 'Hue'},
    ],
    'Configurations': [
        {
            'Classification': 'spark-hive-site',
            'Properties': {
                'hive.metastore.client.factory.class': 'com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory'
            }
        },
        {
            'Classification': 'spark-defaults',
            'Properties': {
                'spark.sql.catalogImplementation': 'hive',
                'spark.hadoop.hive.metastore.client.factory.class': 'com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory'
            }
        }
    ],
    'BootstrapActions': [
        {
            'Name': 'Install Dependencies',
            'ScriptBootstrapAction': {
                'Path': 's3://is459-g1t7-smart-meters-in-london/bootstrap/emr-setup.sh',
            }
        },
    ],
    'Instances': {
        'InstanceGroups': [
            {
                'Name': 'Master node',
                'Market': 'SPOT',
                'InstanceRole': 'MASTER',
                'InstanceType': 'm5.xlarge',
                'InstanceCount': 1,
            },
            {
                'Name': 'Core node',
                'Market': 'SPOT',
                'InstanceRole': 'CORE',
                'InstanceType': 'm5.2xlarge',
                'InstanceCount': 2,
            },
            {
                'Name': 'Task node',
                'Market': 'SPOT',
                'InstanceRole': 'TASK',
                'InstanceType': 'm5.2xlarge',
                'InstanceCount': 2,
            }
        ],
        'Ec2KeyName': 'is459-key',
        'Ec2SubnetId': 'subnet-07735f77141ca3c13',  # Replace with your subnet ID (VPC = vpc-02c11a235d6cac727)
        'KeepJobFlowAliveWhenNoSteps': True,
        'TerminationProtected': False,
    },
    'VisibleToAllUsers': True,
    'JobFlowRole': 'arn:aws:iam::761018854594:instance-profile/AmazonEMR-InstanceProfile-20250210T202740',
    'ServiceRole': 'arn:aws:iam::761018854594:role/service-role/AmazonEMR-ServiceRole-20250310T153207',
    'LogUri': 's3://is459-g1t7-smart-meters-in-london/logs/',  # Ensure this bucket exists
}

# Create the EMR cluster
cluster_creator = EmrCreateJobFlowOperator(
    task_id='create_emr_cluster',
    job_flow_overrides=JOB_FLOW_OVERRIDES,
    aws_conn_id='aws_default',
    emr_conn_id='emr_default',
    dag=dag,
)

# Add Merge Data Step to the EMR cluster
merge_data_step_adder = EmrAddStepsOperator(
    task_id='add_merge_data_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    steps=[
        {
            'Name': 'Merge Daily Data',
            'HadoopJarStep': {
                'Jar': 'command-runner.jar',
                'Args': [
                    'spark-submit',
                    '--deploy-mode', 'cluster',
                    '--master', 'yarn',
                    "s3://is459-g1t7-smart-meters-in-london/pyspark-scripts/q2-merge-daily-data.py",
                ],
            },
        }
    ],
    aws_conn_id='aws_default',
    dag=dag,
)

# Wait for the Merge Data step to complete
merge_data_step_checker = EmrStepSensor(
    task_id='watch_merge_data_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    step_id="{{ task_instance.xcom_pull(task_ids='add_merge_data_step', key='return_value')[0] }}",
    aws_conn_id='aws_default',
    dag=dag,
)

# Add ETL Step to the EMR cluster
etl_step_adder = EmrAddStepsOperator(
    task_id='add_etl_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    steps= [
        {
            'Name': 'ETL: Data Processing',
            'HadoopJarStep': {
                'Jar': 'command-runner.jar',
                'Args': [
                    'spark-submit',
                    '--deploy-mode', 'cluster',
                    '--master', 'yarn',
                    "s3://is459-g1t7-smart-meters-in-london/pyspark-scripts/q2-etl-pyspark.py",
                ],
            },
        }
    ],
    aws_conn_id='aws_default',
    dag=dag,
)

# Wait for the ETL step to complete
etl_step_checker = EmrStepSensor(
    task_id='watch_etl_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    step_id="{{ task_instance.xcom_pull(task_ids='add_etl_step', key='return_value')[0] }}",
    aws_conn_id='aws_default',
    dag=dag,
)

# Add a Glue Crawler to the DAG - Moved before ML step
glue_crawler_task = GlueCrawlerOperator(
    task_id='run_glue_crawler',
    config = {
        'Name': 'q2-ml-forest-output-crawler',
        'Role': 'arn:aws:iam::761018854594:role/service-role/AWSGlueServiceRole-project-q2-real',  # Ensure this role exists
        'DatabaseName': 'q2-processed-data-output',  # Ensure this database exists
        'Targets': {
            'S3Targets': [
                {
                    'Path': 's3://is459-g1t7-smart-meters-in-london/processed-data/merged_df1_df3_df7_df8/',
                    'Exclusions': [],
                    'SampleSize': 2,
                },
            ],
        },
    },
    aws_conn_id='aws_default',
    dag=dag,
)

# Add q2-ml-forest step to the EMR cluster - Now after the Glue crawler
ml_forest_step_adder = EmrAddStepsOperator(
    task_id='add_ml_forest_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    steps=[
        {
            'Name': 'ML Forest',
            'HadoopJarStep': {
                'Jar': 'command-runner.jar',
                'Args': [
                    'spark-submit',
                    '--deploy-mode', 'cluster',
                    '--master', 'yarn',
                    "s3://is459-g1t7-smart-meters-in-london/pyspark-scripts/q2-ml-forest-glue.py",
                ],
            },
        },
    ],
    aws_conn_id='aws_default',
    dag=dag,
)

# Wait for the ML Forest step to complete
ml_forest_step_checker = EmrStepSensor(
    task_id='watch_ml_forest_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    step_id="{{ task_instance.xcom_pull(task_ids='add_ml_forest_step', key='return_value')[0] }}",
    aws_conn_id='aws_default',
    dag=dag,
)

# Add weather API step with Python3 (changed from spark-submit)
weather_api_step_adder = EmrAddStepsOperator(
    task_id='add_weather_api_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    steps=[
        {
            'Name': 'Weather API with Python3',
            'HadoopJarStep': {
                'Jar': 'command-runner.jar',
                'Args': [
                    'bash', '-c',
                    'sudo pip3 install python-dateutil -t /usr/lib/python3.9/site-packages && aws s3 cp s3://is459-g1t7-smart-meters-in-london/pyspark-scripts/weather-api.py /tmp/ && /usr/bin/python3 /tmp/weather-api.py'
                ],
            },
        },
    ],
    aws_conn_id='aws_default',
    dag=dag,
)

# Wait for the weather API step to complete
weather_api_step_checker = EmrStepSensor(
    task_id='watch_weather_api_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    step_id="{{ task_instance.xcom_pull(task_ids='add_weather_api_step', key='return_value')[0] }}",
    aws_conn_id='aws_default',
    dag=dag,
)

# Add 16 days forecast step with Python3 (changed from spark-submit)
forecast_step_adder = EmrAddStepsOperator(
    task_id='add_forecast_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    steps=[
        {
            'Name': '16 Days Forecast with Python3',
            'HadoopJarStep': {
                'Jar': 'command-runner.jar',
                'Args': [
                    'bash', '-c',
                    'pip3 install python-dateutil && aws s3 cp s3://is459-g1t7-smart-meters-in-london/pyspark-scripts/16days-forecast.py /tmp/ &&  /usr/bin/python3 /tmp/16days-forecast.py'
                ],
            },
        },
    ],
    aws_conn_id='aws_default',
    dag=dag,
)

# Wait for the forecast step to complete
forecast_step_checker = EmrStepSensor(
    task_id='watch_forecast_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    step_id="{{ task_instance.xcom_pull(task_ids='add_forecast_step', key='return_value')[0] }}",
    aws_conn_id='aws_default',
    dag=dag,
)

# Terminate the EMR cluster
cluster_terminator = EmrTerminateJobFlowOperator(
    task_id='terminate_emr_cluster',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    aws_conn_id='aws_default',
    dag=dag,
)

# Define the DAG dependencies - Updated to include merge data step before ETL
cluster_creator >> merge_data_step_adder >> merge_data_step_checker >> etl_step_adder >> etl_step_checker >> glue_crawler_task >> ml_forest_step_adder >> ml_forest_step_checker >> weather_api_step_adder >> weather_api_step_checker >> forecast_step_adder >> forecast_step_checker >> cluster_terminator