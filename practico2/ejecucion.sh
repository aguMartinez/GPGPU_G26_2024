#!/bin/bash
#SBATCH --job-name=gpgpu_practico2
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00

#SBATCH --partition=besteffort
# SBATCH --partition=normal

#SBATCH --qos=besteffort_gpu
# SBATCH --qos=gpu

#SBATCH --gres=gpu:p100:1
#SBATCH --mail-type=ALL
# # SBATCH --mail-user=valentinaalaniz2@gmail.com
#SBATCH -o salida.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
cd /clusteruy/home/gpgpu26
./ej1_sol secreto.txt



