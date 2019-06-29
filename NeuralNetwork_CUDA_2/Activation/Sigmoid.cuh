#pragma once

__global__ void sigmoid(float*Z, float*A) {
	const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	A[i] = 1.f / (1.f + expf(-Z[i]));
}

__global__ void sigmoid_derivative(float*G, float* A, float* F) {
	const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
	G[i] = A[i] * (1.f - A[i])*F[i];
}

class Sigmoid
{
public:
    // a = activation(z) = max(z, 0)
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    static inline void activate(const Matrix& Z, Matrix& A)
    {
		sigmoid <<<(Z.size()+255)/256, 256 >>> (Z.data_device.get(), A.data_device.get());
    }

    // Apply the Jacobian matrix J to a vector f
    // J = d_a / d_z = diag(sign(a)) = diag(a > 0)
    // g = J * f = (a > 0) .* f
    // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
    // Note: When entering this function, Z and G may point to the same matrix
    static inline void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
    {
		sigmoid_derivative <<<(Z.size()+255)/256, 256 >>> (G.data_device.get(), A.data_device.get(), F.data_device.get());
    }
};