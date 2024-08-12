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
cd /clusteruy/home/gpgpu26/practico5


# Se crea la carpeta de salida si no existe
output_folder="/clusteruy/home/gpgpu26/practico5/output/"
mkdir -p $output_folder

echo "Compilando el archivo $1..."
nvcc -lineinfo $1 -o ejercicio_compilado 2>&1
if [ $? -ne 0 ]; then
    echo "Error: NVCC failed to compile $1."
    exit 1
fi

# Obtener el nombre del ejercicio sin la extensión
exercise_name=$(basename -s .cu $1)

# Inicializar el nombre del archivo de salida
output_file="${output_folder}${exercise_name}"

if [ ! -z "$3" ]; then
    output_file+="_${3}"
fi
if [ ! -z "$4" ]; then
    output_file+="_${4}"
fi
if [ ! -z "$5" ]; then
    output_file+="_${5}"
fi
if [ ! -z "$6" ]; then
    output_file+="_${6}"
fi
if [ ! -z "$7" ]; then
    output_file+="_${7}"
fi
output_file+=".out"

nsys profile --stats=true --output="${output_folder}${exercise_name}_profile" ./ejercicio_compilado $3 $4 $5 $6 $7 > "${output_file}"