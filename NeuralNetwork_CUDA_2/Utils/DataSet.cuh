#pragma once

#include "../Utils/Matrix.cuh"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>

namespace internal {
	static int ReverseInt(int i) {
		unsigned char c1, c2, c3, c4;

		c1 = i & 255;
		c2 = (i >> 8) & 255;
		c3 = (i >> 16) & 255;
		c4 = (i >> 24) & 255;

		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	}

	static void ReadMNIST(std::string path, Matrix& input, int &width, int &height) {
		std::ifstream file(path, std::ifstream::binary);
		int number_of_images = 0;
		if (file.is_open())
		{
			int magic_number = 0;
			height = 0;
			width = 0;
			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);
			file.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);
			file.read((char*)&height, sizeof(height));
			height = ReverseInt(height);
			file.read((char*)&width, sizeof(width));
			width = ReverseInt(width);
			int size = height * width;
			input.resize(size, number_of_images); 
			input.allocateMemory();
			for (size_t i = 0; i < number_of_images; ++i)
			{
				for (int r = 0; r < size; ++r)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					input[r*number_of_images+i] = (((float)temp) / 255.f);
				}
			}
			input.copyHostToDevice();
		}
	}

	static void ReadMNIST_label(std::string path, Matrix& label) {
		std::ifstream file(path, std::ios::binary);
		if (file.is_open())
		{
			int magic_number = 0;
			int number_of_images = 0;
			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);
			file.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);

			label.resize(10, number_of_images); label.allocateMemory();
			
			memset(label.data_host.get(), 0, label.size()*sizeof(float));
			for (int i = 0; i < number_of_images; ++i)
			{
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				label[(__int8)temp*number_of_images+i] = 1.f;
			}
			label.copyHostToDevice();
		}
	}

	/*static void ReadData(std::string path, Matrix& input) {
		std::ifstream file(path);
		std::string line;

		std::vector<std::vector<double>> temp;
		while (std::getline(file, line)) {
			std::istringstream line_in(line);
			std::vector<double> t;
			double n;
			while (line_in >> n) {
				t.push_back(n);
			}
			temp.push_back(t);
		}
		
		input.resize(temp[0].size(), temp.size()); input.allocateMemory();
		int count = 0;
		for (int i = 0; i < temp[0].size(); i++) {
			for (int j = 0; j < temp.size(); j++,count++) {
				input[count]= temp[j][i];
			}
		}
		
	}*/

	static void NormalizeData(Matrix&input, float norm) {
		for (int i = 0; i < input.size(); i++) {
			input[i] *= norm;
		}
	}
}