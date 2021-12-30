#!/bin/sh
### General options

### –- specify queue --
#BSUB -q hpc

### -- set the job Name --
#BSUB -J real

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

#BSUB -o runs/real-runid-%J.out
#BSUB -e runs/real-runid-%J.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.6.13
module load cuda/11.3

# Go to directory
cd /zhome/e2/5/127625/DL_project

# Load venv
source DLvenv/bin/activate

# Run test
python3 -m src.real.main --baseline --epochs 5000 --freq 100 --lr 0.001 --latent-dim 4 --hidden-dim 30 --lstm-hidden-dim 50 --lstm-layers 1

# Runs for 05/12 -> 06/12
# Real Benchmark + Plot z0 (latent dim 2)
# python3 -m src.real.main --epochs 11000 --freq 100 --lr 0.001 --latent-dim 2 --hidden-dim 40 --rnn-hidden-dim 25
# Real large network RNN
# python3 -m src.real.main --epochs 11000 --freq 100 --lr 0.001 --latent-dim 8 --hidden-dim 50 --rnn-hidden-dim 60
# Real example LSTM
# python3 -m src.real.main --epochs 11000 --freq 100 --lr 0.001 --latent-dim 4 ---hidden-dim 40 --lstm-hidden-dim 40
# python3 -m src.real.main --epochs 11000 --freq 100 --lr 0.001 --latent-dim 8 ---hidden-dim 50 --lstm-hidden-dim 60


