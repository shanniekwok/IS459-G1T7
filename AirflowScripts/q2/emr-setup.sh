#!/bin/bash

set -x -e

# sudo python3 -m pip install --upgrade pip wheel  
sudo python3 -m pip install wheel numpy scipy pandas pyarrow  
sudo python3 -m pip install python-dateutil boto3 findspark scikit-learn  

# Install for all users across all nodes
# # sudo python3 -m pip install --upgrade pip
# sudo python3 -m pip install wheel numpy scipy pandas python-dateutil boto3 pyarrow findspark scikit-learn

# pip3 install --upgrade pip
# pip3 install wheel numpy scipy pandas python-dateutil boto3 pyarrow findspark scikit-learn