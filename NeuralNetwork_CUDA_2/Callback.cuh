#pragma once
#include <fstream>

class NeuralNet;

class Callback
{
protected:

public:
    // Public members that will be set by the network during the training process
    int m_nbatch;   // Number of total batches
    int m_batch_id; // The index for the current mini-batch (0, 1, ..., m_nbatch-1)
    int m_nepoch;   // Total number of epochs (one run on the whole data set) in the training process
    int m_epoch_id; // The index for the current epoch (0, 1, ..., m_nepoch-1)

	std::vector<float> m_loss;

	void output_loss(std::string path) {
		std::ofstream file(path);
		for (const auto &e : m_loss) file << e << "\n";
		m_loss.clear();
	}

    Callback() :
        m_nbatch(0), m_batch_id(0), m_nepoch(0), m_epoch_id(0)
    {}

    virtual ~Callback() {}

    // Before training a mini-batch
    virtual void pre_training_batch(const NeuralNet* net, const Matrix& x, const Matrix& y) {}
    //virtual void pre_training_batch(const NeuralNet* net, const Matrix& x, const IntegerVector& y) {}

    // After a mini-batch is trained
    virtual void post_training_batch(const NeuralNet* net, const Matrix& x, const Matrix& y) {}
    //virtual void post_training_batch(const NeuralNet* net, const Matrix& x, const IntegerVector& y) {}
};