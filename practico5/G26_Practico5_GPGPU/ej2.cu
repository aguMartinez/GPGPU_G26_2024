#include "mmio.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define WARP_PER_BLOCK 32
#define WARP_SIZE 32
#define CUDA_CHK(call) print_cuda_state(call);
#define MAX(A,B)        (((A)>(B))?(A):(B))
#define MIN(A,B)        (((A)<(B))?(A):(B))


// Operación de Paralelizar 2:
struct IncrementarIndice {
    int* ivects;
    const int* niveles;
    const int* RowPtrL_h;

    __device__
    void operator()(int i) {
        int lev = niveles[i] - 1;
        int nnz_row = RowPtrL_h[i+1] - RowPtrL_h[i] - 1;
        int vect_size = 
            (nnz_row == 0) ? 6 :
            (nnz_row == 1) ? 0 :
            (nnz_row <= 2) ? 1 :
            (nnz_row <= 4) ? 2 :
            (nnz_row <= 8) ? 3 :
            (nnz_row <= 16) ? 4 : 5;

        int index = 7 * lev + vect_size;
        atomicAdd(&ivects[index], 1);
    }
};

struct generarOrden {
    int* ivects;
    int* iorder;
    int* ivect_size;
    const int* niveles;
    const int* RowPtrL_h;

    __device__
    void operator()(int i) {
        int idepth = niveles[i]-1;
        int nnz_row = RowPtrL_h[i+1]-RowPtrL_h[i]-1;

        int vect_size = 
            (nnz_row == 0) ? 6 :
            (nnz_row == 1) ? 0 :
            (nnz_row <= 2) ? 1 :
            (nnz_row <= 4) ? 2 :
            (nnz_row <= 8) ? 3 :
            (nnz_row <= 16) ? 4 : 5;


        int key = 7*idepth+vect_size;
        
        // Obtengo el índice de manera atómica y lo incremento
        int index = atomicAdd(&ivects[key], 1);

        // Sección crítica: solo un hilo puede ejecutar esto a la vez para cada key (está salvaguardado por haber incrementado el índice)
        iorder[index] = i;
        ivect_size[index] = (vect_size == 6) ? 0 : pow(2, vect_size);
    }
};

struct CreateKey {
    int* niveles;
    int* ivect_size;

    __host__ __device__
    thrust::tuple<int, int> operator()(const int& idx) {
        return thrust::make_tuple(niveles[idx], ivect_size[idx]);
    }
};

static inline void print_cuda_state(cudaError_t code){

   if (code != cudaSuccess) printf("\ncuda error: %s\n", cudaGetErrorString(code));
   
}


__global__ void kernel_analysis_L(const int* __restrict__ row_ptr,
	const int* __restrict__ col_idx,
	volatile int* is_solved, int n,
	unsigned int* niveles) {
	extern volatile __shared__ int s_mem[];

	if(threadIdx.x==0&&blockIdx.x==0) printf("%i\n", WARP_PER_BLOCK);
	int* s_is_solved = (int*)&s_mem[0];
	int* s_info = (int*)&s_is_solved[WARP_PER_BLOCK];

	int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
	int local_warp_id = threadIdx.x / WARP_SIZE;

	int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

	if (wrp >= n) return;

	int row = row_ptr[wrp];
	int start_row = blockIdx.x * WARP_PER_BLOCK;
	int nxt_row = row_ptr[wrp + 1];

	int my_level = 0;
	if (lne == 0) {
		s_is_solved[local_warp_id] = 0;
		s_info[local_warp_id] = 0;
	}

	__syncthreads();

	int off = row + lne;
	int colidx = col_idx[off];
	int myvar = 0;

	while (off < nxt_row - 1)
	{
		colidx = col_idx[off];
		if (!myvar)
		{
			if (colidx > start_row) {
				myvar = s_is_solved[colidx - start_row];

				if (myvar) {
					my_level = max(my_level, s_info[colidx - start_row]);
				}
			} else
			{
				myvar = is_solved[colidx];

				if (myvar) {
					my_level = max(my_level, niveles[colidx]);
				}
			}
		}

		if (__all_sync(__activemask(), myvar)) {

			off += WARP_SIZE;
			//           colidx = col_idx[off];
			myvar = 0;
		}
	}
	__syncwarp();
	
	for (int i = 16; i >= 1; i /= 2) {
		my_level = max(my_level, __shfl_down_sync(__activemask(), my_level, i));
	}

	if (lne == 0) {

		s_info[local_warp_id] = 1 + my_level;
		s_is_solved[local_warp_id] = 1;
		niveles[wrp] = 1 + my_level;

		__threadfence();

		is_solved[wrp] = 1;
	}
}

    int* RowPtrL_d, *ColIdxL_d;
    VALUE_TYPE* Val_d;


int ordenar_filas( int* RowPtrL, int* ColIdxL, VALUE_TYPE * Val, int n, int* iorder){
    
    int * niveles;

    niveles = (int*) malloc(n * sizeof(int));

    unsigned int * d_niveles;
    int * d_is_solved;
    
    CUDA_CHK( cudaMalloc((void**) &(d_niveles) , n * sizeof(unsigned int)) )
    CUDA_CHK( cudaMalloc((void**) &(d_is_solved) , n * sizeof(int)) )
    
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;

    int grid = ceil ((double)n*WARP_SIZE / (double)(num_threads));

    CUDA_CHK( cudaMemset(d_is_solved, 0, n * sizeof(int)) )
    CUDA_CHK( cudaMemset(d_niveles, 0, n * sizeof(unsigned int)) )


    kernel_analysis_L<<< grid , num_threads, WARP_PER_BLOCK * (2*sizeof(int)) >>>( RowPtrL, 
                                                                                   ColIdxL, 
                                                                                   d_is_solved, 
                                                                                   n, 
                                                                                   d_niveles);

    CUDA_CHK( cudaMemcpy(niveles, d_niveles, n * sizeof(int), cudaMemcpyDeviceToHost) )

    /*Paralelice a partir de aquí*/
    // [begin] evento2
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2, 0);
    // 1)
    /* Obtener el máximo nivel */
    int nLevs = niveles[0]; //TODO: ver si es necesario, agregar al informe si lo es.
    
    //BEGIN Paralelizar maximo/*
        thrust::device_ptr<unsigned int> dev_ptr_d_niveles(d_niveles);
        nLevs = thrust::reduce(thrust::device, dev_ptr_d_niveles, dev_ptr_d_niveles + n, 0, thrust::maximum<unsigned int>());
    //END Paralelizar maximo


    int * RowPtrL_h = (int *) malloc( (n+1) * sizeof(int) );
    CUDA_CHK( cudaMemcpy(RowPtrL_h, RowPtrL, (n+1) * sizeof(int), cudaMemcpyDeviceToHost) )
    int * ivects = (int *) calloc( 7*nLevs, sizeof(int) );
    int * ivect_size  = (int *) calloc(n,sizeof(int));

    // 2)
    // Contar el número de filas en cada nivel y clase de equivalencia de tamaño
    
    //BEGIN Paralelizar Conteo
        thrust::device_vector<int> d_niveles_T(niveles, niveles + n); //se copia el arreglo niveles a un device_vector (tamaño n)
        thrust::device_vector<int> d_RowPtrL_T(RowPtrL_h, RowPtrL_h + n + 1); //se copia el arreglo RowPtrL_h a un device_vector (tamaño n+1)
        thrust::device_vector<int> d_ivects_T(7*nLevs, 0);

        thrust::for_each(
            thrust::device, 
            thrust::counting_iterator<int>(0), 
            thrust::counting_iterator<int>(n), 
            IncrementarIndice{
                thrust::raw_pointer_cast(d_ivects_T.data()), 
                thrust::raw_pointer_cast(d_niveles_T.data()), 
                thrust::raw_pointer_cast(d_RowPtrL_T.data())
            }
        );
    //END Paralelizar Conteo; 


    // 3)
    /* Si se hace una suma prefija del vector se obtiene el punto de comienzo de cada par (tamaño, nivel) en el vector final ordenado */


    //BEGIN Paralelizar scan
        thrust::exclusive_scan(d_ivects_T.begin(), d_ivects_T.end(), d_ivects_T.begin());
    //END Paralelizar scan

    //4)
    /* Usando el offset calculado puedo recorrer por filas y generar un orden
    utilizando el nivel (idepth) y la clase de tamaño (vect_size) como clave.
    Esto se hace asignando a cada fila al punto apuntado por el offset e
    incrementando por 1 luego iorder(ivects(idepth(j)) + offset(idepth(j))) = j
    */

    //BEGIN Paralelizar generar orden

    thrust::device_vector<int> d_iorder_T(n);
    thrust::copy(iorder, iorder+n, d_iorder_T.begin());

    thrust::device_vector<int> d_ivect_size_T(n);
    thrust::copy(ivect_size, ivect_size+n, d_ivect_size_T.begin());


    thrust::for_each(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        generarOrden{
            thrust::raw_pointer_cast(d_ivects_T.data()),
            thrust::raw_pointer_cast(d_iorder_T.data()),
            thrust::raw_pointer_cast(d_ivect_size_T.data()),
            thrust::raw_pointer_cast(d_niveles_T.data()),
            thrust::raw_pointer_cast(d_RowPtrL_T.data())
        }
    );

    //END Paralelizar generar orden
    thrust::copy(d_iorder_T.begin(), d_iorder_T.end(), iorder);
    thrust::copy(d_ivect_size_T.begin(), d_ivect_size_T.end(), ivect_size);

    int ii = 1;
    int filas_warp = 1;

    //5)
    /* Recorrer las filas en el orden dado por iorder y asignarlas a warps
    Dos filas solo pueden ser asignadas a un mismo warp si tienen el mismo 
    nivel y tamaño y si el warp tiene espacio suficiente */

    for (int ctr = 1; ctr < n; ++ctr)
    {
        if( niveles[iorder[ctr]]!=niveles[iorder[ctr-1]] ||
            ivect_size[ctr]!=ivect_size[ctr-1] ||
            filas_warp * ivect_size[ctr] >= 32 ||
            (ivect_size[ctr]==0 && filas_warp == 32) ){

            filas_warp = 1;
            ii++;
        }else{
            filas_warp++;
        }
    }

    int n_warps = ii;
    
    //BEGIN Paralelizar
    /*
    int ii = 0;
    int filas_warp;
    //Creo el conjunto de claves
    thrust::device_vector<thrust::tuple<int, int>> d_keys(n);
    thrust::transform(  
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        d_keys.begin(),
        CreateKey{thrust::raw_pointer_cast(d_niveles_T.data()), thrust::raw_pointer_cast(d_ivect_size_T.data())}
    );

    // Inicializar contadores
    thrust::device_vector<int> d_counts(n, 1);  // Cada fila cuenta como 1


    // Contar elementos por clave usando reduce_by_key
    thrust::device_vector<thrust::tuple<int, int>> d_unique_keys(n);
    thrust::device_vector<int> d_key_counts(n);

    auto new_end = thrust::reduce_by_key(
        thrust::device,
        d_keys.begin(), d_keys.end(),
        d_counts.begin(),
        d_unique_keys.begin(),
        d_key_counts.begin()
    );

    int num_unique_keys = new_end.first - d_unique_keys.begin();

    d_unique_keys.resize(num_unique_keys);
    d_key_counts.resize(num_unique_keys);

    thrust::host_vector<thrust::tuple<int, int>> h_unique_keys(d_unique_keys.begin(), d_unique_keys.begin() + num_unique_keys);
    thrust::host_vector<int> h_key_counts(d_key_counts.begin(), d_key_counts.begin() + num_unique_keys);

    for (int i = 0; i < num_unique_keys; ++i) {
        filas_warp = h_key_counts[i];
        int size = thrust::get<1>(h_unique_keys[i]);
        ii += (filas_warp * size + 31) / 32;
    }

    int n_warps = ii;
    */
    //END Paralelizar
    
    

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);

    float elapsedTime2;
    cudaEventElapsedTime(&elapsedTime2, start2, stop2);
    printf("Tiempo evento interno: %f ms\n", elapsedTime2);
    // [end] evento

    /*Termine aquí*/


    CUDA_CHK( cudaFree(d_niveles) ) 
    CUDA_CHK( cudaFree(d_is_solved) ) 

    return n_warps;

}


int main(int argc, char** argv)
{
    printf("PROGRAMA PARALELO\n");
    // report precision of floating-point
    printf("---------------------------------------------------------------------------------------------\n");
    char* precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char*)"32-bit Single Precision";
    } else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char*)"64-bit Double Precision";
    } else
    {
        printf("Wrong precision. Program exit!\n");
        return 0;
    }

    printf("PRECISION = %s\n", precision);


    int m, n, nnzA;
    int* csrRowPtrA;
    int* csrColIdxA;
    VALUE_TYPE* csrValA;

    int argi = 1;

    char* filename;
    if (argc > argi)
    {
        filename = argv[argi];
        argi++;
    }

    printf("-------------- %s --------------\n", filename);



    // read matrix from mtx file
    int ret_code;
    MM_typecode matcode;
    FILE* f;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_complex(matcode))
    {
        printf("Sorry, data type 'COMPLEX' is not supported.\n");
        return -3;
    }

    char* pch, * pch1;
    pch = strtok(filename, "/");
    while (pch != NULL) {
        pch1 = pch;
        pch = strtok(NULL, "/");
    }

    pch = strtok(pch1, ".");


    if (mm_is_pattern(matcode)) { isPattern = 1; }
    if (mm_is_real(matcode)) { isReal = 1;  }
    if (mm_is_integer(matcode)) { isInteger = 1; }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    if (ret_code != 0)
        return -4;


    if (n != m)
    {
        printf("Matrix is not square.\n");
        return -5;
    }

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric = 1;
        printf("input matrix is symmetric = true\n");
    } else
    {
        printf("input matrix is symmetric = false\n");
    }

    int* csrRowPtrA_counter = (int*)malloc((m + 1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

    int* csrRowIdxA_tmp = (int*)malloc(nnzA_mtx_report * sizeof(int));
    int* csrColIdxA_tmp = (int*)malloc(nnzA_mtx_report * sizeof(int));
    VALUE_TYPE* csrValA_tmp = (VALUE_TYPE*)malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval;
        int ival;
        int returnvalue;

        if (isReal)
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        } else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int*)malloc((m + 1) * sizeof(int));
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m + 1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

    csrColIdxA = (int*)malloc(nnzA * sizeof(int));
    csrValA = (VALUE_TYPE*)malloc(nnzA * sizeof(VALUE_TYPE));

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            } else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    } else
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }
 
    printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);

    // extract L with the unit-lower triangular sparsity structure of A
    int nnzL = 0;
    int* csrRowPtrL_tmp = (int*)malloc((m + 1) * sizeof(int));
    int* csrColIdxL_tmp = (int*)malloc(nnzA * sizeof(int));
    VALUE_TYPE* csrValL_tmp = (VALUE_TYPE*)malloc(nnzA * sizeof(VALUE_TYPE));

    int nnz_pointer = 0;
    csrRowPtrL_tmp[0] = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i + 1]; j++)
        {
            if (csrColIdxA[j] < i)
            {
                csrColIdxL_tmp[nnz_pointer] = csrColIdxA[j];
                csrValL_tmp[nnz_pointer] = 1.0; //csrValA[j];
                nnz_pointer++;
            } else
            {
                break;
            }
        }

        csrColIdxL_tmp[nnz_pointer] = i;
        csrValL_tmp[nnz_pointer] = 1.0;
        nnz_pointer++;

        csrRowPtrL_tmp[i + 1] = nnz_pointer;
    }

    nnzL = csrRowPtrL_tmp[m];
    printf("A's unit-lower triangular L: ( %i, %i ) nnz = %i\n", m, n, nnzL);

    csrColIdxL_tmp = (int*)realloc(csrColIdxL_tmp, sizeof(int) * nnzL);
    csrValL_tmp = (VALUE_TYPE*)realloc(csrValL_tmp, sizeof(VALUE_TYPE) * nnzL);

    printf("---------------------------------------------------------------------------------------------\n");

    int* RowPtrL_d, *ColIdxL_d;
    VALUE_TYPE* Val_d;

    cudaMalloc((void**)&RowPtrL_d, (n + 1) * sizeof(int));
    cudaMalloc((void**)&ColIdxL_d, nnzL * sizeof(int));
    cudaMalloc((void**)&Val_d, nnzL * sizeof(VALUE_TYPE));
  
    cudaMemcpy(RowPtrL_d, csrRowPtrL_tmp, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ColIdxL_d, csrColIdxL_tmp, nnzL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Val_d, csrValL_tmp, nnzL * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    int * iorder  = (int *) calloc(n,sizeof(int));

    // [begin] evento1
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1, 0);

    int nwarps = ordenar_filas(RowPtrL_d,ColIdxL_d,Val_d,n,iorder);

    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);

    float elapsedTime1;
    cudaEventElapsedTime(&elapsedTime1, start1, stop1);
    printf("Tiempo: %f ms\n", elapsedTime1);
    // [end] evento

    printf("Number of warps: %i\n",nwarps);
    for(int i =0; i<n && i<20;i++)
        printf("Iorder[%i] = %i\n",i,iorder[i]);

    printf("Bye!\n");

    // done!
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA);

    free(csrColIdxL_tmp);
    free(csrValL_tmp);
    free(csrRowPtrL_tmp);

    return 0;
}
