#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --account=[YOUR ACCOUNT NAME]
#SBATCH --output=stratifiedcv-%j.out
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1 # The number of CPU cores 

module --force purge
module load StdEnv/2020 gcc/9.3.0 python scipy-stack opencv
virtualenv ~/env-test
source ~/env-test/bin/activate
pip install --no-index six scikit-learn scikit-image numpy==1.20

python stratifiedcv.py
