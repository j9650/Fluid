#pragma once
#include <fstream>
#include <cmath>
#include <map>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <mutex>
#include <cassert>
#include <vector>
#include <unistd.h>
#include <stdio.h>
#include "../utils/logger.h"
#include "../fluid/thread.h"
#include "../fluid/guard.h"
#include "../fluid/valve.h"
#include "../fluid/fluid.h"
#include "../fluid/guardscheduler.h"
#include "math.h"
//http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.302.9543&rep=rep1&type=pdf
#include <chrono> 
using namespace std::chrono; 
#define PI 3.1415926535897931
#define MAX_COUNT 1200000
#define STABLE_TH 6
#define STABLE_DI 1
namespace example {
typedef struct {
   float real;
   float imag;
} Complex;
class fftFluidX_E {
public:
double rate;
int n;
int thread_num;
ValveGT<int> v1;
std::vector<Guard*> guard_log;

void Fft(int n, std::string cmd)
{
static int* indices;
static Complex* x;
static Complex* f;
int i;	
std::cout << "fourier.h" << std::endl;
	std::string outputFilename 	=  "a32_100.txt";

	// prepare the output file for writting the theta values
	std::ofstream outputFileHandler;
	outputFileHandler.open(outputFilename);
	outputFileHandler.precision(5);


	x = (Complex*)malloc(n * sizeof (Complex));
	f = (Complex*)malloc(n * sizeof (Complex));
	indices = (int*)malloc(n * sizeof (int));

	if(x == NULL || f == NULL || indices == NULL)
	{
		std::cout << "cannot allocate memory for the triangle coordinates!" << std::endl;
	}

	int K = n;

	for(i = 0;i < K ; i++)
	{
		x[i].real = i;
		x[i].imag = 0 ;
	}

    if(cmd == "FFT") {
    	radix2DitCooleyTykeyFft(K, indices, x, f);
    } else if (cmd == "FFT_Fluid") {
		radix2DitCooleyTykeyFft_Fluid(K, indices, x, f);
	} else if (cmd == "FFT_multi" || cmd == "FFT_multi_Fluid") {
		radix2DitCooleyTykeyFft_multi(K, indices, x, f, cmd);
	}
	//std::cout<<"$######################Printing Results###########################";
	for(i = 0;i < K ; i++)
	{
		//std::cout<<f[i].real <<" "<<f[i].imag<<std::endl;
		outputFileHandler << f[i].real << " " << f[i].imag << std::endl;
	}
	outputFileHandler.close();

}

void radix2DitCooleyTykeyFft(int K, int* indices, Complex* x, Complex* f) 
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
	for(i = 0, N = 1 << (i + 1); N <= K ; i++, N = 1 << (i + 1))
		{
		std::cout << "iter: " << i << std::endl;
		for(j = 0 ; j < K ; j += N)
		{
			step = N >> 1 ;
			std::cout<<step;
			for(k = 0; k < step ; k++)
			{
				float *fftSin = new float(0.0);
				float *fftCos = new float(0.0);
				arg = (float)k / N ;
				eI = j + k ; 
				oI = j + step + k ;

				dataIn[0] = arg;

#pragma parrot(input, "fft", [1]dataIn)

				int *call_num1 = new int(0);
				int *call_num2 = new int(0);
				FftSin(arg,fftSin, call_num1);
				FftCos(arg,fftCos, call_num2);
				//fftSinCos(arg, &fftSin, &fftCos);
				calcradix(indices,  x,  f, eI, oI, fftSin, fftCos);

				dataOut[0] = *fftSin;
				dataOut[1] = *fftCos;

#pragma parrot(output, "fft", [2]<0.0; 2.0>dataOut)

				*fftSin = dataOut[0];
				*fftCos = dataOut[1];
			}
		}
	}

	for (int i = 0 ; i < K ; i++)
	{
		f[i] = x[indices[i]] ;
	}
}

void radix2DitCooleyTykeyFft_Fluid(int K, int* indices, Complex* x, Complex* f) 
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
	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(10);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);
	tp0->run();


  	 rate = this->rate;
	for(i = 0, N = 1 << (i + 1); N <= K ; i++, N = 1 << (i + 1))
		{
		std::cout << "iter: " << i << std::endl;
		for(j = 0 ; j < K ; j += N)
		{
			step = N >> 1 ;
			std::cout<<step;
			for(k = 0; k < step ; k++)
			{
				arg = (float)k / N ;
				eI = j + k ; 
				oI = j + step + k ;

				dataIn[0] = arg;
				float *fftSin = new float(0.0);
				float *fftCos = new float(0.0);

#pragma parrot(input, "fft", [1]dataIn)
				Data *d2 = new Data;
				d2->set_data(fftSin);
				Data *d3 = new Data;
				d3->set_data(fftSin);
				Data *d4 = new Data;
				d4->set_data(fftCos);
				Data *d5 = new Data;
				d5->set_data(fftCos);

			//	fftSinCos(arg, &fftSin, &fftCos);
				int* anything = new int(0);
				auto call_num = new __fluid__<int>(anything);
										

				auto tpb__g1 = std::bind(&fftFluidX_E::FftSin, this,arg,fftSin ,call_num->p);
				auto tp__g1 = tf->newTask<decltype(tpb__g1)>("fftSin" + std::to_string(arg), {}, {}, tpb__g1);
				Guard*       g1 = gs->newGuard("fftSin" + std::to_string(arg), {  }, {}, tp__g1, {  });
				guard_log.push_back(g1);
				g1->set_root();

				int* anything_call_num2 = new int(0);
				auto call_num2 = new __fluid__<int>(anything_call_num2);

				auto tpb__g2 = std::bind(&fftFluidX_E::FftCos, this,arg,fftCos , call_num2->p);
				auto tp__g2 = tf->newTask<decltype(tpb__g2)>("fftCos" + std::to_string(arg),{}, {}, tpb__g2);
				Guard*       g2 = gs->newGuard("fftCos" + std::to_string(arg), {}, {}, tp__g2, {  });
				guard_log.push_back(g2);


				auto v01 = v1.init(call_num, int(rate*MAX_COUNT));
				auto v02 = v1.init(call_num2, int(rate*MAX_COUNT));
				auto tpb__g3 = std::bind(&fftFluidX_E::calcradix, this, indices,  x,  f, eI, oI, fftSin, fftCos);
				auto tp__g3 = tf->newTask<decltype(tpb__g3)>("calcradix" + std::to_string(arg),{}, {}, tpb__g3);
				Guard*       g3 = gs->newGuard("calcradix" + std::to_string(arg), { v01, v02 }, {}, tp__g3, {  });
				//calcradix(indices,  x,  f, eI, oI, fftSin, fftCos);
				gs->sync(tp__g3);

/*
		auto v2 = v1.init(call_num2, rate*100000000);
		int* anything_call_num3 = new int(0);
		auto call_num3 = new __fluid__<int>(anything_call_num3);
		auto ve = v1.init(call_num3, 0);
*/
		//		auto tpb__g4 = std::bind(&fftFluidX_E::calcradix, this,indices, x, f, eI, oI,  fftSin, fftCos, call_num3->p);
		//		auto tp__g4 = tf->newTask<decltype(tpb__g4)>("calcradix"+std::to_string(1) , {}, {}, tpb__g4);
		//		Guard*       g4 = gs->newGuard("calcradix"+std::to_string(1), { v2 }, {ve}, tp__g4, {  });
		//		g2->set_leaf();
		//		gs->synctask(tp__g2);
		//		gs->synctask(tp__g4);

   				//t =  x[indices[eI]] ;
                //x[indices[eI]].real = t.real + (x[indices[oI]].real * fftCos - x[indices[oI]].imag * fftSin);
                //x[indices[eI]].imag = t.imag + (x[indices[oI]].imag * fftCos + x[indices[oI]].real * fftSin);
//
                //x[indices[oI]].real = t.real - (x[indices[oI]].real * fftCos - x[indices[oI]].imag * fftSin);
                //x[indices[oI]].imag = t.imag - (x[indices[oI]].imag * fftCos + x[indices[oI]].real * fftSin);

//			FftSin(arg,&fftSin);
//				FftCos(arg,&fftCos);
				dataOut[0] = *fftSin;
				dataOut[1] = *fftCos;

#pragma parrot(output, "fft", [2]<0.0; 2.0>dataOut)

				*fftSin = dataOut[0];
				*fftCos = dataOut[1];

				// Non-approximate
				/*t =  x[indices[eI]] ;
				x[indices[eI]].real = t.real + (x[indices[oI]].real * fftCos - x[indices[oI]].imag * fftSin);
                x[indices[eI]].imag = t.imag + (x[indices[oI]].imag * fftCos + x[indices[oI]].real * fftSin);

                x[indices[oI]].real = t.real - (x[indices[oI]].real * fftCos - x[indices[oI]].imag * fftSin);
                x[indices[oI]].imag = t.imag - (x[indices[oI]].imag * fftCos + x[indices[oI]].real * fftSin);*/
			}
		}
	}

			gs->sync(tp0);	
	for (int i = 0 ; i < K ; i++)
	{
		f[i] = x[indices[i]] ;
	}
}

void fftSinCos(float x, float* s, float* c) {
    *s = sin(-2 * PI * x);
    *c = cos(-2 * PI * x);
}





void FftSin(float x, float *s,int *call_num)
{
	float k = -2* PI * x;
	*s=k;
	float t= k;

	for((*call_num)=1;(*call_num)<=MAX_COUNT;(*call_num)++) {
		t=(t*(-1)*k*k)/(2*(*call_num)*(2*(*call_num)+1));
		*s=*s+t;
		//(*call_num)++;
	}
 //   *s = sin(-2 * PI * x);
//		std::cout<<"sin is.............."<<*s<<"................"<<std::endl;	
}
void FftCos(float x, float *c,int *call_num)
{
	float k =  PI * x*-2;
	float t=1;
	*c=1;
	for((*call_num)=1;(*call_num)<=MAX_COUNT;(*call_num)++) {       
		t=t*(-1)*k*k/(2*(*call_num)*(2*(*call_num)-1));
		*c=*c+t;
}
//    *c = cos(-2 * PI * x);
		//std::cout<<"cos is.........."<<*c<<"..............."<<std::endl;	
}
void calcradix(int* indices, Complex* x, Complex* f,int eI,int oI, float *fftSin, float *fftCos)
{ //        std::cout<<"sine is ...."<<fftSin<<"......."<<fftCos;
	Complex t;
	t =  x[indices[eI]] ;
	x[indices[eI]].real = t.real + (x[indices[oI]].real * (*fftCos) - x[indices[oI]].imag * (*fftSin));
	x[indices[eI]].imag = t.imag + (x[indices[oI]].imag * (*fftCos) + x[indices[oI]].real * (*fftSin));

	x[indices[oI]].real = t.real - (x[indices[oI]].real * (*fftCos) - x[indices[oI]].imag * (*fftSin));
	x[indices[oI]].imag = t.imag - (x[indices[oI]].imag * (*fftCos) + x[indices[oI]].real * (*fftSin));
	//*call_num=5;

}  
void calcarg(int k,int n, float *arg)
{
	*arg= (float)k/n;
}

void calcFftIndices(int K, int* indices)
{
	int i, j ;
	int N ;

	N = (int)log2f(K) ;
	indices[0]=0;
	indices[1 << 0] = 1 << (N - (0 + 1)) ;
	for (i = 1; i < N; ++i)
	{
		std::cout << i << std::endl;
		for(j = (1 << i) ; j < (1 << (i + 1)); ++j)
		{
			indices[j] = indices[j - (1 << i)] + (1 << (N - (i + 1))) ;
		}
	}
	std::cout << "after calcFftIndices loop" << std::endl;
}


/////////////////////////////////////////////////multi-thread//////////////////////////////////////////////////////


void radix2DitCooleyTykeyFft_multi(int K, int* indices, Complex* x, Complex* f, std::string cmd);

};
}

using namespace example;
