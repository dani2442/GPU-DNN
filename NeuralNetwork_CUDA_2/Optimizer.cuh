#pragma once

class Optimizer {
public:
	int time = 1;
	virtual ~Optimizer(){}

	virtual void reset() {};
	virtual void update(const Matrix& dvec, Matrix& vec) = 0;
};