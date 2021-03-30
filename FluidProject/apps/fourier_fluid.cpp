#include "fourier.h"
#include "../utils/logger.h"
#define _CRT_RAND_S
#include "stdlib.h"
#include "../fluid/thread.h"
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <ctime>
#include <random>
#include <chrono>
#include <sstream>      // std::stringstream, std::stringbuf
#include <unordered_set>
#include <set>
#include <vector>

using namespace example;	

void radix2DitCooleyTykeyFft_thread(fftFluidX_E *ft, int K, int N, int* indices, Complex* x, Complex* f, int thread_num, int id) {
	int step = N >> 1 ;
	double dataIn[1];
	double dataOut[2];
	int jstart = id / step * N;
	int jstep = (thread_num - 1) / step * N + N;
	int kstart = id % step;

	for (int j = jstart ; j < K ; j += jstep) {
		for(int k = kstart; k < step ; k+=thread_num) {
			float *fftSin = new float(0.0);
			float *fftCos = new float(0.0);
			float arg = (float)k / N ;
			int eI = j + k ; 
			int oI = j + step + k ;
		
			dataIn[0] = arg;
		
		#pragma parrot(input, "fft", [1]dataIn)
		
			int *call_num1 = new int(0);
			int *call_num2 = new int(0);
			ft->FftSin(arg,fftSin, call_num1);
			ft->FftCos(arg,fftCos, call_num2);
			//fftSinCos(arg, &fftSin, &fftCos);
			ft->calcradix(indices,  x,  f, eI, oI, fftSin, fftCos);
		
			dataOut[0] = *fftSin;
			dataOut[1] = *fftCos;
		
		#pragma parrot(output, "fft", [2]<0.0; 2.0>dataOut)
		
			*fftSin = dataOut[0];
			*fftCos = dataOut[1];
		}
	}
}

void radix2DitCooleyTykeyFft_Fluid_thread(fftFluidX_E *ft, int K, int N, int* indices, Complex* x, Complex* f, int thread_num, int id) {
	int step = N >> 1 ;
	double dataIn[1];
	double dataOut[2];
	int jstart = id / step * N;
	int jstep = (thread_num - 1) / step * N + N;
	int kstart = id % step;
	
	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(10);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);
	tp0->run();


	for (int j = jstart ; j < K ; j += jstep) {
		for(int k = kstart; k < step ; k+=thread_num) {
			float *fftSin = new float(0.0);
			float *fftCos = new float(0.0);
			float arg = (float)k / N ;
			int eI = j + k ; 
			int oI = j + step + k ;
		
			dataIn[0] = arg;

		#pragma parrot(input, "fft", [1]dataIn)

			Data *d2 = new Data;
			d2->set_data((void*)fftSin);
			Data *d3 = new Data;
			d3->set_data((void*)fftSin);
			Data *d4 = new Data;
			d4->set_data((void*)fftCos);
			Data *d5 = new Data;
			d5->set_data((void*)fftCos);
			Data *d6 = new Data;
			d6->set_data((void*)fftCos);

			//fftSinCos(arg, &fftSin, &fftCos);
			int* anything = new int(0);
			auto call_num = new __fluid__<int>(anything);
									

			auto tpb__g1 = std::bind(&fftFluidX_E::FftSin, ft,arg,fftSin ,call_num->p);
			auto tp__g1 = tf->newTask<decltype(tpb__g1)>("fftSin" + std::to_string(arg), {d2}, {d3}, tpb__g1);
			Guard*       g1 = gs->newGuard("fftSin" + std::to_string(arg), {  }, {}, tp__g1, {  });
			//guard_log.push_back(g1);
			g1->set_root();

			int* anything_call_num2 = new int(0);
			auto call_num2 = new __fluid__<int>(anything_call_num2);

			auto tpb__g2 = std::bind(&fftFluidX_E::FftCos, ft,arg,fftCos , call_num2->p);
			auto tp__g2 = tf->newTask<decltype(tpb__g2)>("fftCos" + std::to_string(arg),{d4}, {d5}, tpb__g2);
			Guard*       g2 = gs->newGuard("fftCos" + std::to_string(arg), {}, {}, tp__g2, {  });
			g2->set_root();
			//guard_log.push_back(g2);


			auto v01 = ft->v1.init(call_num, int(ft->rate*MAX_COUNT));
			auto v02 = ft->v1.init(call_num2, int(ft->rate*MAX_COUNT));
			auto tpb__g3 = std::bind(&fftFluidX_E::calcradix, ft, indices,  x,  f, eI, oI, fftSin, fftCos);
			auto tp__g3 = tf->newTask<decltype(tpb__g3)>("calcradix" + std::to_string(arg),{d3,d5}, {d6}, tpb__g3);
			Guard*       g3 = gs->newGuard("calcradix" + std::to_string(arg), { v01, v02 }, {}, tp__g3, {});
			g3->set_leaf();
			//calcradix(indices,  x,  f, eI, oI, fftSin, fftCos);
			gs->sync(tp__g3);

			dataOut[0] = *fftSin;
			dataOut[1] = *fftCos;
		
		#pragma parrot(output, "fft", [2]<0.0; 2.0>dataOut)
		
			*fftSin = dataOut[0];
			*fftCos = dataOut[1];
		}
	}

}

void fftFluidX_E::radix2DitCooleyTykeyFft_multi(int K, int* indices, Complex* x, Complex* f, std::string cmd) 
{

std::cout << "radix2DitCooleyTykeyFft" << std::endl;

	calcFftIndices(K, indices) ;
std::cout << "after calcFftIndices" << std::endl;
std::cout<<"indices k"<<*indices<<K<<std::endl;
	int step ;
	float arg ;
	int eI ;
	int oI ;

	Complex t;
	int i ;
	int N ;
	int j ;
	int k ;

	double dataIn[1];
	double dataOut[2];


  	rate = this->rate;
  	unsigned num_cpus = std::thread::hardware_concurrency();
  	std::cout << "num_cpus: " << num_cpus << std::endl;
	for(i = 0, N = 1 << (i + 1); N <= K ; i++, N = 1 << (i + 1))
		{
		std::cout << "iter: " << i << std::endl;
		std::thread* threads = new std::thread[this->thread_num];
		if (cmd == "FFT_multi") {
			for (int id = 0; id < this->thread_num; id++) {
				threads[id] = std::thread(radix2DitCooleyTykeyFft_thread, this, K, N, indices, x, f, this->thread_num, id);
			}
			//radix2DitCooleyTykeyFft_thread(this, K, N, indices, x, f, this->thread_num, 0);
			for(int i = 0; i < this->thread_num; i++) {
				threads[i].join();
			}
		} else if (cmd == "FFT_multi_Fluid") {
			for (int id = 1; id < this->thread_num; id++) {
				threads[id-1] = std::thread(radix2DitCooleyTykeyFft_Fluid_thread, this, K, N, indices, x, f, this->thread_num, id);
			}
			radix2DitCooleyTykeyFft_Fluid_thread(this, K, N, indices, x, f, this->thread_num, 0);
			for(int i = 1; i < this->thread_num; i++) {
				threads[i-1].join();
			}
		}

	}

	for (int i = 0 ; i < K ; i++)
	{
		f[i] = x[indices[i]] ;
	}
}