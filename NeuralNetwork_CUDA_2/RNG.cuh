#pragma once

class RNG {
public:
	RNG(unsigned long seed):
		seed(seed),m_max(2147483647L)
	{}

	long seed;
	const unsigned long m_max;
};