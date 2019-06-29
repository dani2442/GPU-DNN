#include "DNN.cuh"
#include "Utils/DataSet.cuh"
#include <chrono>
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

int main() {
	
	cublasCreate(&handle);
	
	NeuralNet net;

	Layer* layer1 = new FullyConnected<ReLU>(28 * 28, 20);
	Layer* layer2 = new FullyConnected<Sigmoid>(20, 10);

	SGD opt;

	VerboseCallback callback;

	net.add_layer(layer1);
	net.add_layer(layer2);
	net.set_output(new RegressionMSE());
	net.set_callback(callback);

	Matrix x, y;
	int width, height;
	internal::ReadMNIST("C:/Users/NitroPC/source/repos/NeuralNetwork_CUDA_2/NeuralNetwork_CUDA_2/images/test-images.d24",x,width,height);
	internal::ReadMNIST_label("C:/Users/NitroPC/source/repos/NeuralNetwork_CUDA_2/NeuralNetwork_CUDA_2/images/test-labels.d24", y);

	///std::cout << "input:\n";
	///x.copyHostToDevice();x.print();
	///std::cout << "target:\n";
	///y.copyHostToDevice();
	///y.print();

	net.init(0, 0.01f, 1);
	auto t0 = Time::now(); 
	net.fit(opt, x, y, 1024, 1000, 1);
	auto t1 = Time::now();
    fsec fs = t1 - t0;
    std::cout << fs.count() << "s\n";
	cublasDestroy(handle);
	getchar();
	return 0;
	
}

/*int main() {
	
	cublasCreate(&handle);
	
	NeuralNet net;

	Layer* layer1 = new FullyConnected<Sigmoid>(4, 2);
	Layer* layer2 = new FullyConnected<Sigmoid>(2, 1);

	SGD opt;

	VerboseCallback callback;

	net.add_layer(layer1);
	net.add_layer(layer2);
	net.set_output(new RegressionMSE());
	net.set_callback(callback);

	Matrix x, y;
	x.resize(4, 2); x.allocateMemory();
	x[0] = 1;x[1] = 5;x[2] = 2;x[3] = 6;x[4] = 3;x[5] = 7;x[6] = 4;x[7] = 8;
	x.copyHostToDevice();
	y.resize(1, 2); y.allocateMemory();
	y[0] = 0.2f; y[1] = 0.2f;
	y.copyHostToDevice();

	std::cout << "input:\n";
	x.print();
	std::cout << "target:\n";
	y.print();

	net.init(0, 1.f, 1);
	net.fit(opt, x, y, 2, 200, 1);

	cublasDestroy(handle);
	return 0;
	
}*/