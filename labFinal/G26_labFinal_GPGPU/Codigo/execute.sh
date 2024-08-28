#!/bin/bash
#SBATCH --job-name=gpgpu_practico2
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort_gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --mail-type=ALL
#SBATCH -o salida.out


# Cambiar al directorio del proyecto (CAMBIAR PATH)
cd /clusteruy/home/gpgpu26/labFinal/ejemplo_cimg

# Crear carpeta de salida (CAMBIAR PATH)
output_folder="/clusteruy/home/gpgpu26/labFinal/ejemplo_cimg/output/"
mkdir -p $output_folder

# Ejecución:
nsys profile --force-overwrite=true --stats=true --output=output/lab_final ./blur /clusteruy/home/gpgpu26/labFinal/ejemplo_cimg/img_in/${1}.pgm $2 $3 > output/lab_final_${1}_${2}_${3}.out

# Si 'nsys' falla, el script debe manejar el error aquí
if [ $? -ne 0 ]; then
  echo "Error durante el perfilado con nsys. Revisa los logs para más detalles."
  exit 1
fi

output_file="${output_folder}lab_final_${1}_${2}_${3}.out"
echo "Contenido del archivo de salida:"
cat "${output_file}"
