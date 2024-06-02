#!/bin/bash
#SBATCH --job-name=gpgpu_practico2
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:00:30

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
cd /clusteruy/home/gpgpu26/practico4


# Se crea la carpeta de salida si no existe
output_folder="/clusteruy/home/gpgpu26/practico4/output/"
mkdir -p $output_folder

echo "Compilando el archivo $1..."
nvcc -lineinfo $1 -o ejercicio_compilado 2>&1
if [ $? -ne 0 ]; then
    echo "Error: NVCC failed to compile $1."
    exit 1
fi

# Obtener el nombre del ejercicio sin la extensión
exercise_name=$(basename -s .cu $1)

#./ejercicio_compilado $2 > "${output_folder}${exercise_name}_salida.out" 2>&1

nsys profile --stats=true --output="${output_folder}${exercise_name}_profile" ./ejercicio_compilado $3 $4 > "${output_folder}${exercise_name}_${3}_${4}.out"
