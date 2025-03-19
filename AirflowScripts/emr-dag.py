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
    # 'Configurations': [
    #     {
    #         'Classification': 'spark',
    #         'Properties': {
    #             'spark.executor.memoryOverhead': '2g',
    #             'spark.executor.memory': '8g',
    #             'spark.driver.memoryOverhead': '2g',
    #             'spark.driver.memory': '8g',
    #         },
    #     },
    # ],
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
                'InstanceType': 'm5.xlarge',
                'InstanceCount': 2,
            },
            {
                'Name': 'Task node',
                'Market': 'SPOT',
                'InstanceRole': 'TASK',
                'InstanceType': 'm5.xlarge',
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

# Add the new ML Job to the EMR cluster
ml_step_adder = EmrAddStepsOperator(
    task_id='add_ml_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    steps= [
        {
            'Name': 'ML: Model Training and Prediction',
            'HadoopJarStep': {
                'Jar': 'command-runner.jar',
                'Args': [
                    'spark-submit',
                    '--deploy-mode', 'cluster',
                    '--master', 'yarn',
                    "s3://is459-g1t7-smart-meters-in-london/pyspark-scripts/q2-ml-pyspark.py",
                ],
            },
        }
    ],
    aws_conn_id='aws_default',
    dag=dag,
)

# Wait for the new ML Job to complete
ml_step_checker = EmrStepSensor(
    task_id='watch_ml_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    step_id="{{ task_instance.xcom_pull(task_ids='add_ml_step', key='return_value')[0] }}",
    aws_conn_id='aws_default',
    dag=dag,
)

# Add a Glue Crawler to the DAG
glue_crawler_task = GlueCrawlerOperator(
    task_id='run_glue_crawler',
    config = {
        'Name': 'mwaa-output-crawler',
        'Role': 'arn:aws:iam::761018854594:role/service-role/AWSGlueServiceRole-project-q2-real',  # Ensure this role exists
        'DatabaseName': 'mwaa-output-database',  # Ensure this database exists
        'Targets': {
            'S3Targets': [
                {
                    'Path': 's3://is459-g1t7-smart-meters-in-london/mwaa-output/merged_df1_df3_df7_df8/',
                    'Exclusions': [],
                    'SampleSize': 2,
                },
            ],
        },
    },
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

# Define the DAG dependencies
cluster_creator >> etl_step_adder >> etl_step_checker >> glue_crawler_task >> ml_step_adder >> ml_step_checker >> cluster_terminator