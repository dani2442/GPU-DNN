#pragma once

class RegressionMSE : public Output {
private:
	Matrix m_din;

public:
	void evaluate(const Matrix& prev_layer_data,const Matrix& target) {
		const int nobs = prev_layer_data.x;
        const int nvar = prev_layer_data.y;
		m_din.resize(nvar, nobs); m_din.allocateMemory();

		float alpha = 1.f,beta=-1.f;
		cublasSgeam(handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			nvar, nobs,
			&alpha,
			prev_layer_data.data_device.get(), nvar,
			&beta,
			target.data_device.get(), nvar,
			m_din.data_device.get(), nvar);
		// d(L) / d(in) = y_hat - y
        //m_din.resize(nvar, nobs);
        //m_din.noalias() = prev_layer_data - target;

		//std::cout << "m_din = output - target\n";
		//m_din.copyDeviceToHost(); m_din.print();
    }

	const Matrix& backprop_data() const { return m_din; }

	float loss() const {
		float result;
		cublasSnrm2(handle, m_din.size(), m_din.data_device.get(), 1, &result);
		return result * 0.5f / m_din.x;
		// L = 0.5 * ||yhat - y||^2
        //return m_din.squaredNorm() / m_din.cols() * Scalar(0.5);
	}
};