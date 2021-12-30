#!/bin/sh
### General options

### –- specify queue --
#BSUB -q hpc

### -- set the job Name --
#BSUB -J toy

### -- ask for number of cores (default: 1) --
#BSUB -n 1
#BSUB -R "span[hosts=1]"

### -- Select the resources: 1 gpu in exclusive process mode -- 
##BSUB -cpu "num=8"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# request 5GB of system-memory
##BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[model == XeonGold6226R]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u moe.simon@gmail.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o runs/toy-runid-%J.out
#BSUB -e runs/toy-runid-%J.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.6.13
module load cuda/11.3

# Go to directory
cd /zhome/e2/5/127625/DL_project

# Load venv
source DLvenv/bin/activate

python3 -m src.toy.main --baseline --epochs 1000 --freq 100 --num-data 500 --n-total 300 --n-sample 200 --n-skip 1 --latent-dim 4 --hidden-dim 30 --lstm-hidden-dim 45 --lstm-layers 2 --lr 0.001 --solver rk4

# Runs for 05/12 -> 06/12
# Toy 1 Benchmark
# python3 -m src.toy.main --epochs 12000 --freq 100 --num-data 500 --n-total 300 --n-sample 200 --n-skip 2 --latent-dim 4 --hidden-dim 40 --rnn-hidden-dim 25
# Toy 2 Plot z0
# python3 -m src.toy.main --epochs 12000 --freq 100 --num-data 500 --n-total 300 --n-sample 200 --n-skip 2 --latent-dim 2 --hidden-dim 40 --rnn-hidden-dim 25
# LSTM
# python3 -m src.toy.main --epochs 12000 --freq 100 --num-data 500 --n-total 300 --n-sample 200 --n-skip 2 --latent-dim 4 --hidden-dim 40 --lstm-hidden-dim 25 --lr 0.01 
