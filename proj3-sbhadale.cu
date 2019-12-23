#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>

#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))

//data generator
void dataGenerator(long long * data, long long count, int first, int step)
{
	assert(data != NULL);

	for(int i = 0; i < count; ++i)
		data[i] = first + i * step;
	srand(time(NULL));
    for(int i = count-1; i>0; i--) //knuth shuffle
    {
        int j = RAND_RANGE(i);
        int k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
    }

}

__device__ uint bfe(uint x, uint start, uint nbits)
{
    uint bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
    return bits;
}

__global__ void histogram(int* Hist, long long* arrayofkeys, long long size,int noofpartitions)
{
    register int thd = threadIdx.x;
    register int bD = blockDim.x;
    register int bI = blockIdx.x;
    uint h,start,nbits;

    long long thdindex= bD * bI + thd;
    extern __shared__ int sharedpartitions[];
    int * sharedHist = (int *)&sharedpartitions[noofpartitions];

    for(int m =thd;m<noofpartitions;m=m+bD)
        sharedHist[m]=0;

    __syncthreads();

    start=0;
    nbits=(uint)ceil(log2((float)noofpartitions));
    if(thdindex<size)
    {

        h=bfe(arrayofkeys[thdindex],start,nbits);
        atomicAdd(&(sharedHist[h]),1);
    }
    __syncthreads();

    for(int n=thd;n<noofpartitions;n=n+bD)
        atomicAdd(&(Hist[n]),(sharedHist[n]));

}

__global__ void prefixScan(int* Hist,int* Hist_dev_pre, int noofpartitions,long long size)
{
     extern __shared__ int sharedpartitions[];
    register int thd = threadIdx.x;
    int offset = 1;

    sharedpartitions[2*thd]=Hist[2*thd];
    sharedpartitions[2*thd+1]=Hist[2*thd + 1];

    for(int i = noofpartitions>>1;i>0;i>>=1)
    {
        __syncthreads();
        if(thd<i)
        {
            int x = offset*(2*thd+1)-1;
            int y = offset*(2*thd+2)-1;

            sharedpartitions[y]+=sharedpartitions[x];
        }
        offset*=2;
    }

    if(thd==0){sharedpartitions[noofpartitions-1]=0;}

    for(int i = 1;i<noofpartitions;i*=2)
    {
        offset>>=1;
        __syncthreads();
        if(thd<i)
        {

            int x = offset*(2*thd+1)-1;
            int y = offset*(2*thd+2)-1;

            int tmp = sharedpartitions[x];
            sharedpartitions[x]=sharedpartitions[y];
            sharedpartitions[y]+=tmp;
        }
    }
    __syncthreads();

    Hist_dev_pre[2*thd]=sharedpartitions[2*thd];
    Hist_dev_pre[2*thd+1]=sharedpartitions[2*thd+1];
}

__global__ void Reorder(long long* arrayofkeys, int* Hist_pre, int noofpartitions, long long size, long long* output)
{
    register int thd = threadIdx.x;
    register int bD = blockDim.x;
    register int bI = blockIdx.x;
    uint h,start,nbits;
    
    start=0;
    nbits=(uint)ceil(log2((float)noofpartitions));
    
    long long thdindex= bD * bI + thd;
    
    if(thdindex<size)
    {
        h=bfe(arrayofkeys[thdindex],start,nbits);
        int offset=atomicAdd(&(Hist_pre[h]),1);
        output[offset] = arrayofkeys[thdindex];
        
    }
    
}

int main(int argc, char const *argv[])
{
    cudaEvent_t start,stop; 
    
    long long rSize = atoi(argv[1]);
    int targetpartitions=atoi(argv[2]);

    int blocks = ceil((float)rSize/32);
    int blocksforprefix = ceil((float)targetpartitions/32);

    int *Hist_host;
    int *Hist_dev;
    long long *r_host;
    long long *r_dev;
    int *Hist_host_prefix;
    int *Hist_dev_prefix;

    Hist_host=(int *)malloc(sizeof(int)*targetpartitions);
    cudaMallocHost((void**)&r_host,sizeof(long long)*rSize);

    cudaMalloc((void**)&Hist_dev, sizeof(int)*targetpartitions);
    cudaMalloc((void**)&r_dev, sizeof(long long)*rSize);

    dataGenerator(r_host, rSize, 0, 1);

    cudaMemcpy(r_dev,r_host,sizeof(long long)*rSize,cudaMemcpyHostToDevice);
    cudaMemcpy(Hist_dev,Hist_host,sizeof(int)*targetpartitions,cudaMemcpyHostToDevice);
    cudaEventCreate(&start);        // Using Cuda Events to calculate actual precise Kernel Time.
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    histogram<<<blocks,32,sizeof(int)*targetpartitions>>>( Hist_dev, r_dev, rSize, targetpartitions);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize(  stop  );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    printf( "\n************  Running Time of Histogram Kernel: %0.5f msec i.e %0.5f ************\n",elapsedTime,elapsedTime/1000 );
    
    cudaMemcpy(Hist_host,Hist_dev,sizeof(int)*targetpartitions,cudaMemcpyDeviceToHost);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < targetpartitions; i++)
    {
        printf("%d ", Hist_host[i]);

    }
    printf("\n");

    cudaMalloc((void**)&Hist_dev_prefix, sizeof(int)*targetpartitions );
    cudaMemcpy(Hist_dev,Hist_host,sizeof(int)*targetpartitions,cudaMemcpyHostToDevice);
    cudaEventCreate(&start);        // Using Cuda Events to calculate actual precise Kernel Time.
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    prefixScan<<<blocksforprefix,32,sizeof(int)*targetpartitions>>>( Hist_dev, Hist_dev_prefix, targetpartitions, rSize);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize(  stop  );
    float elapsedTimePrefix;
    cudaEventElapsedTime( &elapsedTimePrefix, start, stop );
    Hist_host_prefix=(int*)malloc(sizeof(int)*targetpartitions);
    cudaMemcpy(Hist_host_prefix,Hist_dev_prefix,sizeof(int)*targetpartitions,cudaMemcpyDeviceToHost);
    //output_histogram(Hist_host);
    printf( "\n************  Running Time of prefixScan Kernel: %0.5f msec i.e %0.5f sec************\n",elapsedTimePrefix,elapsedTimePrefix/1000 );
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < targetpartitions; i++)
    {
        printf("%d ", Hist_host_prefix[i]);
        
    }
    printf("\n");
    long long * reorderOutput_host; 
    long long * reorderOutput_dev;

    reorderOutput_host=(long long *)malloc(sizeof(long long)*rSize);
    cudaMalloc((void**)&reorderOutput_dev, sizeof(long long)*rSize);

    cudaMemcpy(reorderOutput_dev,reorderOutput_host,sizeof(long long)*rSize,cudaMemcpyHostToDevice);
    cudaEventCreate(&start);        // Using Cuda Events to calculate actual precise Kernel Time.
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    Reorder<<<blocks,32>>>( r_dev, Hist_dev_prefix, targetpartitions, rSize, reorderOutput_dev);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize(  stop  );
    float elapsedTimeReorder;
    cudaEventElapsedTime( &elapsedTimeReorder, start, stop );
    cudaMemcpy(reorderOutput_host,reorderOutput_dev,sizeof(long long)*rSize,cudaMemcpyDeviceToHost);

    printf( "\n************  Running Time of ReOrder Kernel: %0.5f msec i.e %0.5f sec ************\n", elapsedTimeReorder, elapsedTimeReorder/1000 );
   

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < rSize; i++)
    {
        printf("%d ", reorderOutput_host[i]);
    }
    printf("\n");

    printf( "\n************  Total Running Time of All Kernels: %0.5f msec i.e %0.5f sec************\n", elapsedTimeReorder + elapsedTimePrefix + elapsedTime,(elapsedTimeReorder + elapsedTimePrefix + elapsedTime)/1000);
    // Freeing the Device memory    
        cudaFreeHost(r_host);
        cudaFree(r_dev);
        cudaFree(Hist_dev);
        free(Hist_host);
        free(Hist_host_prefix);
        free(reorderOutput_host);
        cudaFree(Hist_dev_prefix);
        cudaFree(reorderOutput_dev);


    return 0;

}