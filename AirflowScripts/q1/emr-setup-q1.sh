#!/bin/bash

set -x -e

# Install required Python packages
sudo python3 -m pip install boto3 pandas requests
sudo python3 -m pip install apache-airflow[cncf.kubernetes]
sudo python3 -m pip install virtualenv