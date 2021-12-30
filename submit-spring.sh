#!/bin/sh
### General options

### –- specify queue --
#BSUB -q hpc

### -- set the job Name --
#BSUB -J spring

### -- ask for number of cores (default: 1) --
#BSUB -n 1
#BSUB -R "span[hosts=1]"

### -- Select the resources: 1 gpu in exclusive process mode -- 
##BSUB -cpu "num=8"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# request 5GB of system-memory
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

#BSUB -o runs/spring-runid-%J.out
#BSUB -e runs/spring-runid-%J.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.6.13
module load cuda/11.3

# Go to directory
cd /zhome/e2/5/127625/DL_project

# Load venv
source DLvenv/bin/activate

# Run test
#python3 -m src.spring.main --baseline --epochs 500 --freq 50 --num-data 5000 --latent-dim 4 --hidden-dim 25 --lstm-hidden-dim 30 --lstm-layers 1 --lr 0.001 --n-total 300 --n-sample 200 --n-skip 1

python3 -m src.spring.main --load-dir runs/spring_model_1228_1507_41/ckpt/15_52_v9.pth --baseline --epochs 500 --freq 50 --num-data 5000 --lstm-hidden-dim 30 --lr 0.001 --n-total 300 --n-sample 200 --n-skip 1 --example 3

# Load run
# python3 -m src.spring.main --epochs 5000 --freq 100 --load-dir experiment/models/15_59_v13.pth --lstm-hidden-dim 1
