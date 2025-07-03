NUM_CPUS=4
USERNAME="rhaldar"


for D in 0 5 10 20 50; do
  for SEED in {1..30}; do
    FILENAME="mnist_${D}_${SEED}"
    PYTHON_COMM="python -u mnist.py --pad ${D} --seed ${SEED}"

echo "#!/bin/bash
#SBATCH -A partner
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${NUM_CPUS}
#SBATCH --time=1-00:00:00
#SBATCH --job-name ${FILENAME}

# Run python file.

# Load our conda environment
module purge
module load anaconda/2020.11-py38
module load use.own
module load conda-env/DL-py3.8.5

# Change to main project directory
cd /home/${USERNAME}
${PYTHON_COMM}" > job

sbatch job
  done
done
