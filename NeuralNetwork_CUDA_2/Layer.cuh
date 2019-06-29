#pragma once

#pragma once
#include <vector>
#include "Optimizer.cuh"
class Layer
{
protected:
	 ///cublasHandle_t handle;
     int m_in_size;  // Size of input units
     int m_out_size; // Size of output units

public:
    Layer(const int in_size, const int out_size) :
        m_in_size(in_size), m_out_size(out_size)
    {
		///checkCudaErrors(cublasCreate(&handle));
	}
    virtual ~Layer() {
		///checkCudaErrors(cublasDestroy(handle));
	}

    int in_size() const { return m_in_size; }
    int out_size() const { return m_out_size; }

    virtual void init(const float& mu, const float& sigma, RNG& rng) = 0;
    virtual void forward(const Matrix& prev_layer_data) = 0;

    virtual const Matrix& output() const= 0;

    virtual void backprop(const Matrix& prev_layer_data,const Matrix& next_layer_data) = 0;
    virtual const Matrix& backprop_data() const = 0;

    virtual void update(Optimizer& opt) = 0;

    virtual std::vector<float> get_parameters() const = 0;
    virtual void set_parameters(const std::vector<float>& param) {};
    virtual std::vector<float> get_derivatives() const = 0;
	virtual void copy(Layer* l) {};
};