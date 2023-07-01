#!/bin/sh -l

#SBATCH -A qfsong
#SBATCH --nodes=1 
#SBATCH --time=24:00:00
#SBATCH --job-name ridgeless

# Print the hostname of the compute node on which this job is running.
/bin/hostname

source ~/bashtf_cpu

cd ~/Manifold-Simulations/Yue code

h=$1
D=$2
codim=$3
n=$4
method=$5
seed=$6

python change_shift_cpu.py $D $codim 0.1 0.01 2000 $seed $h $method $n
