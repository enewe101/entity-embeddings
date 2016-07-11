#!/bin/bash
#PBS -N DeepUbuntu
#PBS -A jim-594-aa
#PBS -l walltime=4:00:00
#PBS -l nodes=1:gpus=2

module load apps/python/2.7.5
cd "${PBS_O_WORKDIR}"
source venv/bin/activates
THEANO_FLAGS='floatX=float32,device=gpu' python main.py --encoder rnn --batch_size=512 --hidden_size=50 --optimizer adam --lr 0.001 --fine_tune_W=True --fine_tune_M=True --input_dir dataset_1M
