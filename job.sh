#!/bin/bash
#PBS -N job
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -l walltime=16:00:00

cd $PBS_O_WORKDIR

source env/bin/activate

# script to run
python pinn/allen_cahn_1d.py --arch 100 100 100 100 --resample-period 100 --dist LHS --outdir results/allen_cahn
