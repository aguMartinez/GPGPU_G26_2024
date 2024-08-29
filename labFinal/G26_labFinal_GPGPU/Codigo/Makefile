# Definir compilador y banderas
CC = nvcc

# Parámetros
VALUE_TYPE = double

# Rutas de instalación de CUDA
CUDA_INSTALL_PATH = /usr/local/cuda-12.1

# Flags para nvcc
NVCC_FLAGS = -O3 -w -m64 -std=c++11 -Xcompiler "-std=c++11" -gencode=arch=compute_60,code=sm_60 -Xptxas -dlcm=cg -rdc=true

# Incluye rutas
INCLUDES = -I$(CUDA_INSTALL_PATH)/include

# Librerías
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib -lcudart -lcuda -lcusparse -lnvidia-ml
LIBS = $(CUDA_LIBS) -lm -lpthread -lX11 -lnvToolsExt

# Archivos fuente
SOURCES = main.cpp src/cpu_baseline.cu src/gpu_baseline.cu src/gpu_v1_0.cu src/gpu_v1_1.cu src/gpu_v1_2.cu src/gpu_v2_0.cu src/gpu_v2_1.cu src/gpu_v3.cu src/gpu_v4_0.cu src/gpu_v4_1.cu src/gpu_v4_2.cu

# Objetivo por defecto
all: blur

# Compilación del programa principal
blur:
	$(CC) $(NVCC_FLAGS) $(SOURCES) $(CUFILE) -o blur $(INCLUDES) $(LIBS) -D VALUE_TYPE=$(VALUE_TYPE)

# Limpiar archivos compilados
clean:
	rm -f blur *.o
