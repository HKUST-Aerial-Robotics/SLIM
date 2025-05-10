#ifndef __DEVICE_BUFFER_H__
#define __DEVICE_BUFFER_H__

#include <cuda_runtime.h>
#include <iostream>
#include "macro.h"

template <typename T>
struct DeviceBuffer
{
	DeviceBuffer() : data(nullptr), size(0) {}
	DeviceBuffer(size_t size) : data(nullptr), size(0) { allocate(size); }
	~DeviceBuffer() { destroy(); }

	void allocate(size_t _size)
	{
		if (data && size >= _size)
			return;

		destroy();
		CHECK_CUDA(cudaMalloc(&data, sizeof(T) * _size));
		size = _size;
	}

	void destroy()
	{
		if (data)
			CHECK_CUDA(cudaFree(data));
		data = nullptr;
		size = 0;
	}

	void upload(const T* h_data)
	{
		CHECK_CUDA(cudaMemcpy(data, h_data, sizeof(T) * size, cudaMemcpyHostToDevice));
	}

	void download(T* h_data) const
	{
		CHECK_CUDA(cudaMemcpy(h_data, data, sizeof(T) * size, cudaMemcpyDeviceToHost));
	}

	void copyTo(DeviceBuffer& rhs) const
	{
		CHECK_CUDA(cudaMemcpy(rhs.data, data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
	}

	void copyTo(T* rhs) const
	{
		CHECK_CUDA(cudaMemcpy(rhs, data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
	}

	void fillZero()
	{
		CHECK_CUDA(cudaMemset(data, 0, sizeof(T) * size));
	}

	void fill(T value)
	{
		CHECK_CUDA(cudaMemset(data, value, sizeof(T) * size));
	}

	bool empty() const { return !(data && size > 0); }

	void assign(size_t size, const T* h_data)
	{
		allocate(size);
		upload(h_data);
	}

	void print() const {
		T* h_data = (T*)malloc(sizeof(T) * size);
		download(h_data);
		for(int i = 0; i < size; ++i) {
			std::cout << h_data[i] << ", ";
		}
		std::cout << std::endl;
	}

	T* data;
	size_t size;
};

#endif // !__DEVICE_BUFFER_H__
