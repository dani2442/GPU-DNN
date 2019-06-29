#pragma once

class Output {
public:
	virtual ~Output() {}

	virtual void check_target_data(const Matrix& target) {}
	//virtual void check_target_data(const IntegerVector& target){throw std::invalid_argument("[class Output]: This output type cannot take class labels as target data");}

	virtual void evaluate(const Matrix& prev_layer_data,const Matrix& target) = 0;
	//virtual void evaluate(const Matrix& prev_layer_data, const IntegerVector& target){throw std::invalid_argument("[class Output]: This output type cannot take class labels as target data");}

	virtual const Matrix& backprop_data() const = 0;
	virtual float loss() const = 0;
};