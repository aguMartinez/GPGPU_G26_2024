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

export PATH=$PATH:/usr/local/cuda-12.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64

cd /clusteruy/home/gpgpu26/practico5/ej2

output_folder="/clusteruy/home/gpgpu26/practico5/ej2/output/"
mkdir -p $output_folder

echo "Compilando el archivo..."
make -f Makefile_og

# Obtener el nombre del ejercicio sin la extension
exercise_name="ej2_og"

# Inicializar el nombre del archivo de salida
output_file="${output_folder}${exercise_name}.out"

nsys profile --stats=true --output="${output_folder}${exercise_name}" ./main_og $1 $2 $3 $4 $5 > "${output_file}"