#!/bin/bash
#PBS -N hard-vs-soft-on-poisson-3d
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -l walltime=02:00:00

cd $PBS_O_WORKDIR

source env/bin/activate

# script to run
python pinn/poisson_3d.py --arch 20 20 20 20 --resample-period 1 --bc hard
python pinn/poisson_3d.py --arch 20 20 20 20 --resample-period 1 --bc soft
