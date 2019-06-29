#pragma once

__global__ void argmax(float*Z, float*A) {
	const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (Z[i] > 0)
		A[i] = Z[i];
	else
		A[i] = 0;
}

__global__ void argmax_derivative(float*G, float* A, float* F) {
	const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
	G[i] = A[i] > 0 ? F[i] : 0;
}

class ReLU
{
public:
    // a = activation(z) = max(z, 0)
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    static inline void activate(const Matrix& Z, Matrix& A)
    {
		argmax <<<(Z.size()+255)/256, 256 >>> (Z.data_device.get(), A.data_device.get());
    }

    // Apply the Jacobian matrix J to a vector f
    // J = d_a / d_z = diag(sign(a)) = diag(a > 0)
    // g = J * f = (a > 0) .* f
    // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
    // Note: When entering this function, Z and G may point to the same matrix
    static inline void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
    {
		argmax_derivative <<<(Z.size()+255)/256, 256 >>> (G.data_device.get(), A.data_device.get(), F.data_device.get());
    }
};