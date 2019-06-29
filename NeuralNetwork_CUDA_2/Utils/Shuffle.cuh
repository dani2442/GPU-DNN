#pragma once
#include <cstdio>
#include <stdlib.h>
#if 1
template <class T>
__device__ __inline__
void atomicAdd(T* ptr, T val) {
  *ptr+=val;
}
#endif

__global__ void addBias(float* in, float *b, int in_x, int in_y) {
	__shared__ float bias;
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (0==threadIdx.x)
		bias = b[blockIdx.x];
	__syncthreads();
	for (int i = threadIdx.x + blockIdx.x*in_x; i < in_y*(blockIdx.x + 1); i += blockDim.x) {
		in[i] += bias;
	}
}

// mean
__global__ void device_reduce_atomic_kernel1(float *in, float* out, int N) {
  float sum=0.f;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    sum+=in[i];
  }
  atomicAdd(out,sum/N); // mean
}

void device_reduce_atomic1(float *in, float* out, int N) {
  int threads=256;
  int blocks=min((N+threads-1)/threads,2048);

  cudaMemsetAsync(out,0,sizeof(int));
  device_reduce_atomic_kernel1<<<blocks,threads>>>(in,out,N); 
}

// sum
__global__ void device_reduce_atomic_kernel2(float *in, float* out, int N) {
  float sum=0.f;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    sum+=in[i];
  }
  atomicAdd(out,sum);
}

void device_reduce_atomic2(float *in, float* out, int N) {
  int threads=256;
  int blocks=min((N+threads-1)/threads,2048);

  cudaMemsetAsync(out,0,sizeof(int));
  device_reduce_atomic_kernel2<<<blocks,threads>>>(in,out,N); 
}

//sum of square
__global__ void device_reduce_atomic_kernel3(float *in, float* out, int N) {
  float sum=0.f;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    sum+=pow(in[i],2);
  }
  atomicAdd(out,sum);
}

void device_reduce_atomic3(float *in, float* out, int N) {
  int threads=256;
  int blocks=min((N+threads-1)/threads,2048);

  cudaMemsetAsync(out,0,sizeof(int));
  device_reduce_atomic_kernel3<<<blocks,threads>>>(in,out,N); 
}