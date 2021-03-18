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


void fftFluidX_E::FftFluidX_E(int n)
{
	static int* indices;
//	static Complex* x;
//	static Complex* f;
	 int K = n;

        for(i = 0;i < K ; i++)
        {
  //              x[i].real = i;
   //             x[i].imag = 0 ;
        }
        radix2DitCooleyTykeyFft(K, indices, x, f) ;
//return 0;
}
void calcFftIndices(int K, int* indices)
{
	int i, j ;
	int N ;

	N = (int)log2f(K) ;

	indices[0] = 0 ;
	indices[1 << 0] = 1 << (N - (0 + 1)) ;
	for (i = 1; i < N; ++i)
	{
		for(j = (1 << i) ; j < (1 << (i + 1)); ++j)
		{
			indices[j] = indices[j - (1 << i)] + (1 << (N - (i + 1))) ;
		}
	}
}
void fftFluidX_E::radix2DitCooleyTykeyFft(int K, int* indices, Complex* x, Complex* f)
{

	calcFftIndices(K, indices) ;

	int step ;
	float arg ;
	int eI ;
	int oI ;

	float fftSin=0;
	float fftCos=0;

	Complex t;
	int i ;
	int N ;
	int j ;
	int k ;

	double dataIn[1];
	double dataOut[2];
/////////////////Fluid begins////////////////////
	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(4);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

  	double rate = this->rate;
	
	for(i = 0, N = 1 << (i + 1); N <= K ; i++, N = 1 << (i + 1))
	{
		for(j = 0 ; j < K ; j += N)
		{
			step = N >> 1 ;
			for(k = 0; k < step ; k++)
			{
			//	arg = (float)k / N ;
				eI = j + k ; 
				oI = j + step + k ;


#pragma parrot(input, "fft", [1]dataIn)
				Data *d1 = new Data;
				d1->set_data((void*)arg);
				Data *d2 = new Data;
				d2->set_data((void*)fftSin);
				Data *d3 = new Data;
				d3->set_data((void*)fftCos);
				int* anything = new int(0);
				auto call_num = new __fluid__<int>(anything);
				
				auto tpb__g1 = std::bind(&fftFluidX_E::calcarg, this,k,N , &arg);
				auto tp__g1 = tf->newTask<decltype(tpb__g1)>("calcarg" + std::to_string(arg), {d1}, {d2}, tpb__g1);
				Guard*       g1 = gs->newGuard("calcarg" + std::to_string(arg), { v0 }, {}, tp__g1, {  });
				g1->set_root();

				auto tpb__g2 = std::bind(&fftFluidX_E::fftSin, this,arg,&fftSin , call_num1);
				auto tp__g2 = tf->newTask<decltype(tpb__g2)>("fftSin" + std::to_string(arg), {d1}, {d2}, tpb__g2);
				Guard*       g2 = gs->newGuard("fftSin" + std::to_string(arg), { v0 }, {}, tp__g2, {  });
				
				auto tpb__g3 = std::bind(&fftFluidX_E::fftCos, this,arg,&fftCos , call_num1);
				auto tp__g3 = tf->newTask<decltype(tpb__g3)>("fftCos" + std::to_string(arg), {d1}, {d3}, tpb__g3);
				Guard*       g3 = gs->newGuard("fftCos" + std::to_string(arg), { v0 }, {}, tp__g3, {  });

				auto tpb__g4 = std::bind(&fftFluidX_E::calcradix, this,arg,&fftCos , call_num1);
				auto tp__g4 = tf->newTask<decltype(tpb__g4)>("fftCos" + std::to_string(arg), {d1}, {d3}, tpb__g4);
				Guard*       g4 = gs->newGuard("fftCos" + std::to_string(arg), { v0 }, {}, tp__g4, {  });
				g4->set_leaf();

				dataIn[0] = arg;

			//	fftSinCos(arg, &fftSin, &fftCos);

				dataOut[0] = fftSin;
				dataOut[1] = fftCos;

#pragma parrot(output, "fft", [2]<0.0; 2.0>dataOut)

				fftSin = dataOut[0];
				fftCos = dataOut[1];

	                    }				// Non-approximate
		}
	}

	for (int i = 0 ; i < K ; i++)
	{
		f[i] = x[indices[i]] ;
		std::cout<<"REsults"<<f[i].real << " " << f[i].imag << std::endl;
	}
}
void fftFluidX_E::fftSin(float x, float* s, int *call_num) {
   float k = -2* PI * x;
   long int i;
   for(i=1;i<=1000000;i++)  {
       *s = *s + (pow(-1,i-1)*pow(k,2*i-1))/factorial(2*i-1);
	(*call_num)++;
}
}

void fftFluidX_E::fftCos(float x, float* c,int *call_num){
   float k = -2* PI * x;
   long int i;
   for(i=0;i<=1000000;i++)  
{       *c = *c + (pow(-1,i)*pow(k,2*i))/factorial(2*i);
	(*call_num)++;
}


}
void fftFluidX_E::calcarg(int k,int n, float *arg)
{
	*arg= (float)k/n;
}
void fftFluidX_E::calcradix(int* indices, Complex* x, Complex* f,int t,int eI,int oI, float fftSin, float fftCos)
{
				t =  x[indices[eI]] ;
				x[indices[eI]].real = t.real + (x[indices[oI]].real * fftCos - x[indices[oI]].imag * fftSin);
                x[indices[eI]].imag = t.imag + (x[indices[oI]].imag * fftCos + x[indices[oI]].real * fftSin);

                x[indices[oI]].real = t.real - (x[indices[oI]].real * fftCos - x[indices[oI]].imag * fftSin);
                x[indices[oI]].imag = t.imag - (x[indices[oI]].imag * fftCos + x[indices[oI]].real * fftSin);

}  
