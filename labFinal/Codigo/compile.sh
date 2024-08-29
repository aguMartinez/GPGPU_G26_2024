#!/bin/bash
#SBATCH --job-name=gpgpu_practico2
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort_gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --mail-type=ALL
#SBATCH -o salida.out

# Añadir CUDA al PATH y LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/cuda-12.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64

# Cambiar al directorio del proyecto (CAMBIAR PATH)
cd /clusteruy/home/gpgpu26/labFinal/ejemplo_cimg

# Limpiar archivos compilados anteriores
rm -f blur

# Crear carpeta de salida (CAMBIAR PATH)
output_folder="/clusteruy/home/gpgpu26/labFinal/ejemplo_cimg/output/"
mkdir -p $output_folder

# Compilar y ejecutar el perfilado usando el Makefile
echo "Compilando y ejecutando el perfilado..."
make -f Makefile blur

# Si 'make' falla, el script debe manejar el error aquí
if [ $? -ne 0 ]; then
  echo "Error de compilación: Revisa los logs para más detalles."
  exit 1
fi
