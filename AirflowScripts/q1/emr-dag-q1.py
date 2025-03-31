# --------------------------------------
# Import libraries
# --------------------------------------
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.amazon.aws.operators.emr import (
    EmrCreateJobFlowOperator,
    EmrAddStepsOperator,
    EmrTerminateJobFlowOperator,
)
from airflow.providers.amazon.aws.sensors.emr import EmrStepSensor
from airflow.providers.amazon.aws.operators.glue_crawler import GlueCrawlerOperator
from airflow.providers.amazon.aws.operators.athena import AthenaOperator

# --------------------------------------
# Set default arguments
# --------------------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# --------------------------------------
# Create DAG
# --------------------------------------
dag = DAG(
    'emr_dag_q1',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# --------------------------------------
# Define EMR cluster configuration
# --------------------------------------
JOB_FLOW_OVERRIDES = {
    'Name': 'smart-meters-emr-cluster-q1',
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
    'Instances': {
        'InstanceGroups': [
            {
                'Name': 'Master node',
                'Market': 'SPOT',
                'InstanceRole': 'MASTER',
                'InstanceType': 'm5.2xlarge',
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
        'Ec2SubnetId': 'subnet-07735f77141ca3c13',
        'KeepJobFlowAliveWhenNoSteps': True,
        'TerminationProtected': False,
    },
    'VisibleToAllUsers': True,
    'JobFlowRole': 'arn:aws:iam::761018854594:instance-profile/AmazonEMR-InstanceProfile-20250210T202740',
    'ServiceRole': 'arn:aws:iam::761018854594:role/service-role/AmazonEMR-ServiceRole-20250310T153207',
    'LogUri': 's3://is459-g1t7-smart-meters-in-london/logs/',
}

# --------------------------------------
# Create EMR cluster
# --------------------------------------
cluster_creator = EmrCreateJobFlowOperator(
    task_id='create_emr_cluster',
    job_flow_overrides=JOB_FLOW_OVERRIDES,
    aws_conn_id='aws_default',
    emr_conn_id='emr_default',
    dag=dag,
)

# --------------------------------------
# Add ETL Step to EMR cluster
# --------------------------------------
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
                    "s3://is459-g1t7-smart-meters-in-london/pyspark-scripts/q1-etl-pyspark.py",
                ],
            },
        }
    ],
    aws_conn_id='aws_default',
    dag=dag,
)

# --------------------------------------
# Wait for ETL Step to complete
# --------------------------------------
etl_step_checker = EmrStepSensor(
    task_id='watch_etl_step',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    step_id="{{ task_instance.xcom_pull(task_ids='add_etl_step', key='return_value')[0] }}",
    aws_conn_id='aws_default',
    dag=dag,
)

# --------------------------------------
# Add Glue Crawler to DAG
# --------------------------------------
glue_crawler_task = GlueCrawlerOperator(
    task_id='run_glue_crawler',
    config={
        'Name': 'glue_crawler_output_q1',
        'Role': 'arn:aws:iam::761018854594:role/AWSGlueServiceRole-project-q1-v1',
        'DatabaseName': 'q1-processed-data-schema',
        'Targets': {
            'S3Targets': [    
                # For df4_melt                                                                        
                {
                    'Path': 's3://is459-g1t7-smart-meters-in-london/processed-data/final_q1_df/df4/',  
                    'Exclusions': [],
                    'SampleSize': 2,
                },
                # For merged_df2_df4_df6_df10_df12_df14
                {
                    'Path': 's3://is459-g1t7-smart-meters-in-london/processed-data/final_q1_df/merged_df2_df4_df6_df10_df12_df14/',
                    'Exclusions': [],
                    'SampleSize': 2,
                },
            ],
        },
    },
    aws_conn_id='aws_default',
    dag=dag,
)

# --------------------------------------
# Add Athena Query Task to DAG
# --------------------------------------

# For df4_melt
athena_query_task_1 = AthenaOperator(
    task_id='execute_athena_query_df4_melt',
    query="""
        SELECT 
            * 
        FROM "q1-processed-data-schema"."df4_melt"
    """,
    database='q1-processed-data-schema',
    output_location='s3://is459-g1t7-smart-meters-in-london/athena-results/final_q1_df/df4_melt/',
    workgroup='primary',
    aws_conn_id='aws_default',
    dag=dag,
)

# For merged_df2_df4_df6_df10_df12_df14
athena_query_task_2 = AthenaOperator(
    task_id='execute_athena_query_merged_df2_df4_df6_df10_df12_df14',
    query="""
        SELECT 
            * 
        FROM "q1-processed-data-schema"."merged_df2_df4_df6_df10_df12_df14"
    """,
    database='q1-processed-data-schema',
    output_location='s3://is459-g1t7-smart-meters-in-london/athena-results/final_q1_df/merged_df2_df4_df6_df10_df12_df14/',
    workgroup='primary',
    aws_conn_id='aws_default',
    dag=dag,
)

# --------------------------------------
# Terminate EMR Cluster
# --------------------------------------
cluster_terminator = EmrTerminateJobFlowOperator(
    task_id='terminate_emr_cluster',
    job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
    aws_conn_id='aws_default',
    dag=dag,
)

# --------------------------------------
# Define DAG Dependency
# --------------------------------------
cluster_creator >> etl_step_adder >> etl_step_checker >> glue_crawler_task >> athena_query_task_1 >> athena_query_task_2 >> cluster_terminator