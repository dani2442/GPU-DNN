#pragma once
#include <curand.h>
#include <vector>
#include <iostream>
#include "../Utils/Matrix.cuh"
#include "../RNG.cuh"


namespace internal {

int create_shuffled_batches(const Matrix& x, const Matrix& y, int batch_size, RNG& rng, std::vector<Matrix>& x_batches, std::vector<Matrix>& y_batches)
{
    const int nobs = x.x;
    const int dimx = x.y;
    const int dimy = y.y;

    // Compute batch size
    if(batch_size > nobs)
        batch_size = nobs;
    const int nbatch = (nobs - 1) / batch_size + 1;
    const int last_batch_size = nobs - (nbatch - 1) * batch_size;

	x_batches.resize(nbatch-1, Matrix(dimx, batch_size));
	x_batches.push_back(Matrix(dimx, last_batch_size));
	y_batches.resize(nbatch-1, Matrix(dimy, batch_size));
	y_batches.push_back(Matrix(dimy, last_batch_size));
    for(int i = 0; i < nbatch; i++)
    {
        const int bsize = (i == nbatch - 1) ? last_batch_size : batch_size;
		x_batches[i].allocateMemory();
		y_batches[i].allocateMemory();
        // Copy data
        //const int offset = i * batch_size;
        for(int j = 0; j < bsize; j++)
        {
			for (int k = 0; k < dimx; k++) {
				//x[i*bsize + j + nobs * k];
				x_batches[i][j + k * bsize] = x[i*bsize + j + nobs * k];
			}
			for (int k = 0; k < dimy; k++) {
				y_batches[i][j + k * bsize] = y[i*bsize + j + nobs * k];
			}
        }
		x_batches[i].copyHostToDevice();
		y_batches[i].copyHostToDevice();
    }
    return nbatch;
}

// Fill array with N(mu, sigma^2) random numbers
void set_normal_random(float* arr,int size, RNG& rng, float mu = 0.f, float sigma = 1.f)
{
	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, rng.seed);
	curandGenerateNormal(gen, arr, size, mu, sigma);
}

void set_uniform_random(float* arr,int size, RNG& rng)
{
	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, rng.seed);
	curandGenerateUniform(gen, arr, size);
}

} // namespace internal
