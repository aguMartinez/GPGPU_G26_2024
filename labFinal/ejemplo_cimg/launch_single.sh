#!/bin/bash
#SBATCH --job-name=gpgpu_practico2
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort_gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --mail-type=ALL
#SBATCH -o salida.out

# Verificar que se pase el nombre del archivo .cu como parámetro
if [ -z "$1" ]; then
  echo "Error: No se especificó el archivo .cu a compilar."
  exit 1
fi

CUFILE=$1
CUFILENAME=$(basename -- "$CUFILE")
CUFILENAME="${CUFILENAME%.*}"

# Añadir CUDA al PATH y LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/cuda-12.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64

# Cambiar al directorio del proyecto
cd /clusteruy/home/gpgpu26/labFinal/ejemplo_cimg

# Limpiar archivos compilados anteriores
rm -f blur

# Crear carpeta de salida
output_folder="/clusteruy/home/gpgpu26/labFinal/ejemplo_cimg/output/"
mkdir -p $output_folder

# Compilar y ejecutar el perfilado usando el Makefile
echo "Compilando y ejecutando el perfilado..."
make -f Makefile CUFILE=${CUFILE} blur

# Si 'make' falla, el script debe manejar el error aquí
if [ $? -ne 0 ]; then
  echo "Error de compilación: Revisa los logs para más detalles."
  exit 1
fi

# Verificar la salida del programa
nsys profile --force-overwrite=true --stats=true --output=output/lab_final ./blur /clusteruy/home/gpgpu26/labFinal/ejemplo_cimg/test.pgm > output/lab_final.out

# Si 'nsys' falla, el script debe manejar el error aquí
if [ $? -ne 0 ]; then
  echo "Error durante el perfilado con nsys. Revisa los logs para más detalles."
  exit 1
fi

output_file="${output_folder}lab_final.out"
echo "Contenido del archivo de salida:"
cat "${output_file}"
