#include<stdio.h>
#include<cuda.h>


__global__ void multiply()
{

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
 
 
void main(int argc, char *argv[])
{
	double** M1=Make2DdoubleArray(N,N);
	double** M2=Make2DdoubleArray(N,N);
	double** Prod=Make2DdoubleArray(N,N);
	
	double* M1_host_flat;
	double* M2_host_flat;
	
	
	
	
	
}
