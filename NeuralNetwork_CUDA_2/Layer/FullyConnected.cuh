#include "Utils/Random.cuh"
#include "Utils/Shuffle.cuh"
#include <thread>

template <typename Activation>
class FullyConnected: public Layer
{
private:
	bool init_stream = false;

	Matrix m_vec;	  // vector to do mean
    Matrix m_weight;  // Weight parameters, W(in_size x out_size)
    Matrix m_bias;    // Bias parameters, b(out_size x 1)
    Matrix m_dw;      // Derivative of weights
    Matrix m_db;      // Derivative of bias
    Matrix m_z;       // Linear term, z = W' * in + b
    Matrix m_a;       // Output of this layer, a = act(z)
    Matrix m_din;     // Derivative of the input of this layer.
                      // Note that input of this layer is also the output of previous layer
	cudaStream_t stream[1000];
	std::thread t[1000];

public:
    FullyConnected(const int in_size, const int out_size) :
        Layer(in_size, out_size)
    {}

    void init(const float& mu, const float& sigma, RNG& rng)
    {
		m_weight.resize(this->m_in_size, this->m_out_size); m_weight.allocateMemory();
		m_bias.resize(this->m_out_size,1); m_bias.allocateMemory();
		m_dw.resize(this->m_in_size, this->m_out_size); m_dw.allocateMemory();
		m_db.resize(this->m_out_size,1); m_db.allocateMemory();

        // Set random coefficients
		
        internal::set_normal_random(m_weight.data_device.get(), m_weight.size(), rng, mu, sigma);
		internal::set_uniform_random(m_bias.data_device.get(), m_bias.size(), rng);
		//cudaMemset(m_bias.data_device.get(), 0, m_bias.size()*sizeof(float));
		
		///std::cout << "weight\n";
		///m_weight.copyDeviceToHost(); m_weight.print();
		///std::cout << "bias\n";
		///m_bias.copyDeviceToHost(); m_bias.print();
		//for (int i = 0; i < 1000; i++) {
			//cudaStreamCreate(&stream[i]);
		//}
    }


    // prev_layer_data: in_size x nobs
    void forward(const Matrix& prev_layer_data)
    {
        const int nobs = prev_layer_data.x;
        // Linear term z = W' * in + b
		m_z.resize(this->m_out_size, nobs); m_z.allocateMemory();

		int a_y=m_weight.y;
		int b_x=prev_layer_data.x;
		int ab_k=m_weight.x;
		float alpha = 1.f, beta = 0.f;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
			b_x, ab_k,a_y,
			&alpha,
			prev_layer_data.data_device.get(), b_x,
			m_weight.data_device.get(), ab_k,
			&beta,
			m_z.data_device.get(), b_x);

		///std::cout << "m_z = W' * in\n";
		///m_z.copyDeviceToHost(); m_z.print();
		

		//for (int i = 0; i < m_z.x; i++) {
			//cublasSetStream_v2(handle,stream[i]);
			//cublasSaxpy(handle,m_z.y,&alpha, m_bias.data_device.get(),1, &m_z.data_device.get()[i], m_z.x);

		//}
		//for (int i = 0; i < m_z.x; i++) {
			//cudaStreamSynchronize(stream[i]);
		//}
		 int threads=min(m_z.x,1024);
		 int blocks=m_z.y;
		 addBias << <blocks, threads,m_z.y >> > (m_z.data_device.get(), m_bias.data_device.get(), m_z.x, m_z.y);

		///std::cout << "m_z = W' * in + b\n";
		///m_z.copyDeviceToHost(); m_z.print();
        //m_z.noalias() = m_weight.transpose() * prev_layer_data;
        //m_z.colwise() += m_bias;

        // Apply activation function
		m_a.resize(this->m_out_size, nobs); m_a.allocateMemory();
        Activation::activate(m_z, m_a);

		///std::cout << "m_a = activation(m_z)\n";
		///m_a.copyDeviceToHost(); m_a.print();
    }

    const Matrix& output() const
    {
        return m_a;
    }

    // prev_layer_data: in_size x nobs
    // next_layer_data: out_size x nobs
    void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data)
    {
        const int nobs = prev_layer_data.x;

        // After forward stage, m_z contains z = W' * in + b
        // Now we need to calculate d(L) / d(z) = [d(a) / d(z)] * [d(L) / d(a)]
        // d(L) / d(a) is computed in the next layer, contained in next_layer_data
        // The Jacobian matrix J = d(a) / d(z) is determined by the activation function
        Activation::apply_jacobian(m_z, m_a, next_layer_data, m_z);

		///std::cout << "m_z=jacobian()\n";
		///m_z.copyDeviceToHost(); m_z.print();

		float alpha = 1.f/nobs, beta = 0.f;
		cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
			m_z.y,prev_layer_data.y,prev_layer_data.x,
			&alpha, 
			m_z.data_device.get(), m_z.x, 
			prev_layer_data.data_device.get(), prev_layer_data.x, 
			&beta, 
			m_dw.data_device.get(), m_z.y);

		///std::cout << "m_dw = prev_layer_data * m_z' / nobs\n";
		///m_dw.copyDeviceToHost(); m_dw.print();
        // Now dLz contains d(L) / d(z)
        // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
		///m_dw.noalias() = prev_layer_data * dLz.transpose() / nobs;
		m_vec.resize(nobs, 1); m_vec.allocateMemory(); m_vec.init(1.f);
		///std::cout << "m_vec = \n";
		///m_vec.copyDeviceToHost(); m_vec.print();

		cublasSgemv(handle, CUBLAS_OP_T, m_z.y, nobs,
			&alpha,
			m_z.data_device.get(),m_z.y,
			m_vec.data_device.get(), 1,
			&beta,
			m_db.data_device.get(), 1);

		//for (int i = 0; i < m_z.y; i++) {
			//t[i] = std::thread(device_reduce_atomic1, &m_z.data_device.get()[i * m_z.x], &m_db.data_device.get()[i], m_z.x, stream[i]);
			//device_reduce_atomic1(m_z.data_device.get(), m_db.data_device.get(), m_z.x);
			//device_reduce_atomic1(&m_z.data_device.get()[i * m_z.x], &m_db.data_device.get()[i], m_z.x);
		//}
		//for (int i = 0; i < min(m_z.y, 1000); i++) {
			//cudaStreamSynchronize(stream[i]);
		//}
		//for (int i = 0; i < m_z.y; i++) {
			//t[i].join();
		//}

		///std::cout << "m_db = mean(rowwise)\n";
		///m_db.copyDeviceToHost(); m_db.print();
        // Derivative for bias, d(L) / d(b) = d(L) / d(z)
        ///m_db.noalias() = dLz.rowwise().mean();

		m_din.resize(this->m_in_size, nobs); m_din.allocateMemory();
		alpha = 1.f; beta = 0.f;
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
			m_z.x, m_weight.y, m_z.y, 
			&alpha, 
			m_z.data_device.get(), m_z.x, 
			m_weight.data_device.get(), m_z.y, 
			&beta, 
			m_din.data_device.get(), m_z.x);
        // Compute d(L) / d_in = W * [d(L) / d(z)]
		//m_din.resize(this->m_in_size, nobs);
        //m_din.noalias() = m_weight * dLz;

		///std::cout << "m_din = m_weight * m_z\n";
		///m_din.copyDeviceToHost(); m_din.print();
		

    }

    const Matrix& backprop_data() const
    {
        return m_din;
    }

    void update(Optimizer& opt)
    {
		//ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
		//ConstAlignedMapVec db(m_db.data(), m_db.size());
		// AlignedMapVec      w(m_weight.data(), m_weight.size());
		//AlignedMapVec      b(m_bias.data(), m_bias.size());

		opt.update(m_dw, m_weight);
		opt.update(m_db, m_bias);
    }

    std::vector<float> get_parameters() const
    {
        std::vector<float> res(m_weight.size() + m_bias.size());
        // Copy the data of weights and bias to a long vector
		///std::cout << m_weight << std::endl;

        ///std::copy(m_weight.data(), m_weight.data() + m_weight.size(), res.begin());
        ///std::copy(m_bias.data(), m_bias.data() + m_bias.size(), res.begin() + m_weight.size());

        return res;
    }

    void set_parameters(const std::vector<float>& param)
    {
        if(static_cast<int>(param.size()) != m_weight.size() + m_bias.size())
            throw std::invalid_argument("Parameter size does not match");

        ///std::copy(param.begin(), param.begin() + m_weight.size(), m_weight.data_host.get());
        ///std::copy(param.begin() + m_weight.size(), param.end(), m_bias.data_host.get());
    }

    std::vector<float> get_derivatives() const
    {
        std::vector<float> res(m_dw.size() + m_db.size());
        // Copy the data of weights and bias to a long vector
        ///std::copy(m_dw.data(), m_dw.data_host.get() + m_dw.size(), res.begin());
        ///std::copy(m_db.data(), m_db.data_host.get() + m_db.size(), res.begin() + m_dw.size());

        return res;
    }

};