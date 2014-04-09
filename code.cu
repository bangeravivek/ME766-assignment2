/*
This program generates 2 N*N matrices and then multiplies them on a GPU
*/


#include<stdio.h>
#include<cuda.h>
#define N 100

__global__ void multiply(double* A, double* B, double* C, int K)
{
	int index1=threadIdx.x*N+threadIdx.y;
	int index2=threadIdx.y*N+threadIdx.x;
	double sum=0.0;
	for (int i=0;i<K;i++)
	{
		sum+=A[threadIdx.x*K+i]*B[i*K+threadIdx.x];
	}
	
	C[threadIdx.x*K];
}

double** Make2DdoubleArray(int arraySizeX, int arraySizeY) {
double** theArray;
theArray = (double**) malloc(arraySizeX*sizeof(double*));
int i;
for (i = 0; i < arraySizeX; i++)
   theArray[i] = (double*) malloc(arraySizeY*sizeof(double));
int j;

for (i=0;i<arraySizeX;i++)
{
    for (j=0;j<arraySizeY;j++)
    {
        theArray[i][j]=((double)rand())/(double)(100);
    }
}

   return theArray;
}

void init_zeros(double** matrix)
{
	int i,j;
	for (i=0;i<N;i++)
	{	
		for (j=0;j<N;j++)
		{
			matrix[i][j]=0;
		}
	}
}

double* Make1DdoubleArray(int arraySizeX) {
double* theArray;
theArray = (double*)malloc(arraySizeX*sizeof(double));
int i;
for (i=0;i<arraySizeX;i++)
{
    theArray[i]=0.0;
}

   return theArray;
}

void printmat(double** matrix)
{
	int i,j;
	
	for (i=0;i<N;i++)
	{	
		printf("\n");
		for (j=0;j<N;j++)
		{
			printf("%f \t",matrix[i][j]);
		}
	}
	printf("\n");
}
 
 
int main(int argc, char *argv[])
{
	const int N = 100;
	double** M1=Make2DdoubleArray(N,N);
	double** M2=Make2DdoubleArray(N,N);
	double** Prod=Make2DdoubleArray(N,N);
	
	init_zeros(Prod);
	
	double* M1_host_flat=Make1DdoubleArray(N*N);
	double* M2_host_flat=Make1DdoubleArray(N*N);
	double* Prod_host_flat=Make1DdoubleArray(N*N);
	
	double* M1_device_flat;
	double* M2_device_flat;
	double* Prod_device_flat;
	int* N_device;
	
	int counter=0;
	int i,j;
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			M1_host_flat[counter]=M1[i][j];
			M2_host_flat[counter]=M2[i][j];
			Prod_host_flat[counter]=Prod[i][j];
			counter+=1;
			
		}
	}
	
	cudaMalloc((void **) &M1_device_flat, sizeof(double)*N*N);
	cudaMalloc((void **) &M2_device_flat, sizeof(double)*N*N);
	cudaMalloc((void **) &Prod_device_flat, sizeof(double)*N*N);
	cudaMalloc((void **) &N_device, sizeof(int));
	
	cudaMemcpy(M1_device_flat, M1_host_flat, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(M2_device_flat, M2_host_flat, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(Prod_device_flat, Prod_host_flat, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(N_device, &N, sizeof(int), cudaMemcpyHostToDevice);
	//Kernel call
	
	//Copy data back to host
	
	cudaMemcpy(Prod_host_flat, Prod_device_flat, sizeof(int)*N*N, cudaMemcpyDeviceToHost);	
	
	return 0;	
}
