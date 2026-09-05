#!/bin/bash
#PBS -N allen_cahn
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -l walltime=02:00:00

cd $PBS_O_WORKDIR

source env/bin/activate

# script to run
python pinn/allen_cahn_1d.py
