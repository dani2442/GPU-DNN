#pragma once 
class VerboseCallback: public Callback
{
public:
    void post_training_batch(const NeuralNet* net, const Matrix& x, const Matrix& y)
    {
       // const float loss = net->get_output()->loss();
        //std::cout << "[Epoch " << m_epoch_id << ", batch " << m_batch_id << "] Loss = " << loss << std::endl;
		//m_loss.push_back(loss);
    }
};