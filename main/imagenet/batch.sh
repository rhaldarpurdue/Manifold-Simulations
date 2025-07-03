NUM_CPUS=4
USERNAME="rhaldar"


for D in 32 64 128 256 320; do
  for SEED in {1..30}; do
    FILENAME="imagenet_${D}_${SEED}"
    PYTHON_COMM="python -u imagenette_dim.py --res ${D} --seed ${SEED}"

echo "#!/bin/bash
#SBATCH -A standby
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${NUM_CPUS}
#SBATCH --time=04:00:00
#SBATCH --job-name ${FILENAME}

# Run python file.

# Load our conda environment
module purge
module load anaconda/2020.11-py38
module load use.own
source activate dl2

# Change to main project directory
cd /home/${USERNAME}/imagenet
${PYTHON_COMM}" > job

sbatch job
  done
done
