/*
This program generates 2 N*N matrices and then multiplies them on a GPU
*/


#include<stdio.h>
#include<cuda.h>
//#define N 100

__global__ void multiply(double* A, double* B, double* C, int K)
{
	int index1=threadIdx.x*K+threadIdx.y;
	int index2=threadIdx.y*K+threadIdx.x;
	double sum=0.0;
	for (int i=0;i<K;i++)
	{
		sum+=A[index1]*B[index2];
	}
	
	C[index1+index2]=sum;
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

void init_zeros(double** matrix, int K)
{
	int i,j;
	for (i=0;i<K;i++)
	{	
		for (j=0;j<K;j++)
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

void printmat(double** matrix, int K)
{
	int i,j;
	
	for (i=0;i<K;i++)
	{	
		printf("\n");
		for (j=0;j<K;j++)
		{
			printf("%f \t",matrix[i][j]);
		}
	}
	printf("\n");
}
 
 
int main(int argc, char *argv[])
{
	const int K = 5;
	double** M1=Make2DdoubleArray(K,K);
	double** M2=Make2DdoubleArray(K,K);
	double** Prod=Make2DdoubleArray(K,K);
	
	init_zeros(Prod, K);
	
	double* M1_host_flat=Make1DdoubleArray(K*K);
	double* M2_host_flat=Make1DdoubleArray(K*K);
	double* Prod_host_flat=Make1DdoubleArray(K*K);
	
	double* M1_device_flat;
	double* M2_device_flat;
	double* Prod_device_flat;
	int* K_device;
	
	int counter=0;
	int i,j;
	for(i=0;i<K;i++)
	{
		for(j=0;j<K;j++)
		{
			M1_host_flat[counter]=M1[i][j];
			M2_host_flat[counter]=M2[i][j];
			Prod_host_flat[counter]=Prod[i][j];
			counter+=1;
			
		}
	}
	
	cudaMalloc((void **) &M1_device_flat, sizeof(double)*K*K);
	cudaMalloc((void **) &M2_device_flat, sizeof(double)*K*K);
	cudaMalloc((void **) &Prod_device_flat, sizeof(double)*K*K);
	cudaMalloc((void **) &K_device, sizeof(int));
	
	cudaMemcpy(M1_device_flat, M1_host_flat, sizeof(double)*K*K, cudaMemcpyHostToDevice);
	cudaMemcpy(M2_device_flat, M2_host_flat, sizeof(double)*K*K, cudaMemcpyHostToDevice);
	cudaMemcpy(Prod_device_flat, Prod_host_flat, sizeof(double)*K*K, cudaMemcpyHostToDevice);
	cudaMemcpy(K_device, &K, sizeof(int), cudaMemcpyHostToDevice);
	//Kernel call
	dim3 threads(K,K);
	multiply<<<1,threads>>>(M1_device_flat,M2_device_flat,Prod_device_flat, K); 
	//Copy data back to host
	
	cudaMemcpy(Prod_host_flat, Prod_device_flat, sizeof(int)*K*K, cudaMemcpyDeviceToHost);	
	
	counter=0;
	printf("\n");
	printf("\n");
	printf("\n");
	for (i=0;i<K;i++)
	{
		//fprintf(results_file,"\n");
		printf("\n");
		for (j=0;j<K;j++)
		{
			printf("%f ", Prod_host_flat[counter]);
			counter+=1;
		}
	}
	printf("\n");

	
	
	return 0;	
}
