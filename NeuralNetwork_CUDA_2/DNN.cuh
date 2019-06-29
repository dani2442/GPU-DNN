#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <curand.h>

#include <iostream>
#include <vector>
#include <stdlib.h>

cublasHandle_t handle;

#include "Utils/Error.cuh"
#include "Utils/Matrix.cuh"

#include "RNG.cuh"

#include "Layer.cuh"
#include "Layer/FullyConnected.cuh"
//#include "Layer/Convolutional.h"
//#include "Layer/MaxPooling.h"
//#include "Layer/AveragePooling.h"

#include "Activation/ReLU.cuh"
//#include "Activation/Identity.h"
#include "Activation/Sigmoid.cuh"
//#include "Activation/Softmax.h"
//#include "Activation/Abs.h"
//#include "Activation/Tanh.h"

#include "Output.cuh"
#include "Output/RegressionMSE.cuh"
//#include "Output/BinaryClassEntropy.h"
//#include "Output/MultiClassEntropy.h"

#include "Optimizer.cuh"
#include "Optimizer/SGD.cuh"
//#include "Optimizer/AdaGrad.h"
//#include "Optimizer/RMSProp.h"
//#include "Optimizer/AdaDelta.h"
//#include "Optimizer/Adam.h"
//#include "Optimizer/Momentum.h"
//#include "Optimizer/NAG.h"
//#include "Optimizer/AdaMax.h"
//#include "Optimizer/Nadam.h"
//#include "Optimizer/AMSGrad.h"

#include "Callback.cuh"

//#include "TrainingSet.h"

#include "NeuralNetwork.cuh"

#include "Callback/VerboseCallback.cuh"