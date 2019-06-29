#pragma once
#include <memory>
#include <stdio.h>

__global__ void initKernel(float* devPtr, const float val, const int nwords)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; tidx < nwords; tidx += stride)
        devPtr[tidx] = val;
}

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
class Matrix {
private:
	bool device_allocated;
	bool host_allocated;
	bool m_init;

	void allocateCudaMemory() {
		if (!device_allocated) {
			float* device_memory = nullptr;
			cudaMalloc(&device_memory, x * y * sizeof(float));
			data_device = std::shared_ptr<float>(device_memory, [&](float* ptr) { cudaFree(ptr); });
			device_allocated = true;
		}
	}
	void allocateHostMemory() {
		if (!host_allocated) {
			data_host = std::shared_ptr<float>(new float[x * y], [&](float* ptr) { delete[] ptr; });
		host_allocated = true;
		}
	}

public:
	int x, y;

	std::shared_ptr<float> data_device;
	std::shared_ptr<float> data_host;

	Matrix(int y_dim = 1, int x_dim = 1) :
		x(x_dim), y(y_dim), data_device(nullptr), data_host(nullptr),
		device_allocated(false), host_allocated(false),m_init(false)
	{}

	void resize(int y, int x) {
		this->y = y;
		this->x = x;
	}

	int size() const { return x * y; }

	void init(const float a) {
		if (m_init)return;
		int threads = 1024;
		int blocks = (size() + 1023) / 1024;
		initKernel <<<blocks, threads >>> (data_device.get(), a, size());
		m_init = true;
	}

	void allocateMemory() {
		allocateCudaMemory();
		allocateHostMemory();
	}
	void allocateMemoryIfNotAllocated(int y, int x) {
		if (!device_allocated && !host_allocated) {
			this->x = x;
			this->y = y;
			allocateMemory();
		}
	}

	void copyHostToDevice() {
		if (device_allocated && host_allocated) {
			cudaMemcpy(data_device.get(), data_host.get(), x * y * sizeof(float), cudaMemcpyHostToDevice);
		}
	}
	void copyDeviceToHost() {
		if (device_allocated && host_allocated) {
			cudaMemcpy(data_host.get(), data_device.get(), x * y * sizeof(float), cudaMemcpyDeviceToHost);
		}
	}

	const void print() const {
		for(int i = 0; i < y; ++i){
			for(int j = 0; j < x; ++j){
				printf("%10.7f ", data_host.get()[IDX2C(j, i, x)]);
				//std::cout << data_host.get()[IDX2C(j, i, x)] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	float& operator[](const int index) {
		return data_host.get()[index];
	}
	const float& operator[](const int index) const {
		return data_host.get()[index];
	}
};