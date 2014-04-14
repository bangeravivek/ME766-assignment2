/*
This program generates 2 N*N matrices and then multiplies them on a GPU
*/


#include<stdio.h>
#include<math.h>
#include<cuda.h>
//#define N 100

__global__ void multiply(float* A, float* B, float* C, int K)
{
	int index1=blockIdx.x*blockDim.x+threadIdx.x;
	int index2=blockIdx.x*blockDim.y+threadIdx.y;
	float sum=0.0;
	for (int i=0;i<K;i++)
	{
		sum+=A[index2*K+i]*B[i*K+index1];
	}
	
	C[index2*K+index1]=sum;
}

float** Make2DfloatArray(int arraySizeX, int arraySizeY) {
float** theArray;
theArray = (float**) malloc(arraySizeX*sizeof(float*));
int i;
for (i = 0; i < arraySizeX; i++)
   theArray[i] = (float*) malloc(arraySizeY*sizeof(float));
int j;

for (i=0;i<arraySizeX;i++)
{
    for (j=0;j<arraySizeY;j++)
    {
        theArray[i][j]=rand()%100;
    }
}

   return theArray;
}

void init_zeros(float** matrix, int K)
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

float* Make1DfloatArray(int arraySizeX) {
float* theArray;
theArray = (float*)malloc(arraySizeX*sizeof(float));
int i;
for (i=0;i<arraySizeX;i++)
{
    theArray[i]=0.0;
}

   return theArray;
}

void printmat(float** matrix, int K)
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
	float** M1=Make2DfloatArray(K,K);
	float** M2=Make2DfloatArray(K,K);
	float** Prod=Make2DfloatArray(K,K);
	
	init_zeros(Prod, K);
	
	float* M1_host_flat=Make1DfloatArray(K*K);
	float* M2_host_flat=Make1DfloatArray(K*K);
	float* Prod_host_flat=Make1DfloatArray(K*K);
	
	float* M1_device_flat;
	float* M2_device_flat;
	float* Prod_device_flat;
	int* K_device;
	
	printmat(M1,K);
	printmat(M2,K);
	printmat(Prod,K);
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
	
	cudaMalloc((void **) &M1_device_flat, sizeof(float)*K*K);
	cudaMalloc((void **) &M2_device_flat, sizeof(float)*K*K);
	cudaMalloc((void **) &Prod_device_flat, sizeof(float)*K*K);
	cudaMalloc((void **) &K_device, sizeof(int));
	
	cudaMemcpy(M1_device_flat, M1_host_flat, sizeof(float)*K*K, cudaMemcpyHostToDevice);
	cudaMemcpy(M2_device_flat, M2_host_flat, sizeof(float)*K*K, cudaMemcpyHostToDevice);
	cudaMemcpy(Prod_device_flat, Prod_host_flat, sizeof(float)*K*K, cudaMemcpyHostToDevice);
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
