#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job name --
#BSUB -J grid_FFN_Mag
### -- ask for number of cores (default: 1) --
#BSUB -n 12
#BSUB -R "span[hosts=1]"
### -- select the resources: 1 gpu in exclusice process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 20GB of memory
#BSUB -R "rusage[mem=15GB]"
### -- set the email address --
# please uncomment the following line and in your e-mail address,
# if you wan to receive e-mail notifications on a non-defalt address
#BSUB -u s123028@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo gpu-%J.out
#BSUB -eo gpu-%J.err
# -- end of LSF options
module load cudnn/v6.0-prod
module load python3/3.6.2
source /appl/tensorflow/1.4gpu-python362/bin/activate
python grid_FFN_Framework.py





