#pragma once
#include "../Optimizer.cuh"


class SGD: public Optimizer
{

public:
	float m_lrate;
	float m_decay;

	SGD() :
		m_lrate(-0.1f), m_decay(0.f)
	{}

    void update(const Matrix& dvec, Matrix& vec)
    {
		
		///std::cout << "vec_previous\n";
		///vec.copyDeviceToHost(); vec.print();

		float alpha = 1.f;
		cublasSgeam(handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			vec.y, dvec.x,
			&alpha,
			vec.data_device.get(), vec.y,
			&m_lrate,
			dvec.data_device.get(), dvec.y,
			vec.data_device.get(), vec.y);
			
		
		///std::cout << "vec_updated\n";
		///vec.copyDeviceToHost(); vec.print();
		//vec.noalias() -= m_lrate * dvec;


    }
};
