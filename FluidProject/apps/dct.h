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
#include <stdio.h>
#include "../utils/logger.h"
#include "../fluid/thread.h"
#include "../fluid/guard.h"
#include "../fluid/valve.h"
#include "../fluid/fluid.h"
#include "../fluid/guardscheduler.h"
#include "math.h"
#define pi 3.1426457
//http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.302.9543&rep=rep1&type=pdf
#include<functional>
#include <chrono> 
using namespace std::chrono; 
namespace example {

class dctFluidX_E {
public:

int m=8; //no of rows
int n=8;//no of columns;
int rate;
int a;
std::vector<Guard*> guard_log;

ValveGT<int> v1;
int matrix[8][8];

void dctCos(float x, float *c,int *call_num)
{
   float k =   x;
   long int i;
std::cout<<"incoming cos k = "<<k<<std::endl;
float t=1;
*c=1;
   for(i=1;i<=10;i++)  
{       
t=t*(-1)*k*k/(2*i*(2*i-1));
	*c=*c+t;
	(*call_num)++;

}
//    *c = cos( x);
		std::cout<<"cos is.........."<<*c<<"..............."<<std::endl;	
}
void generate_matrix(int (&matrix)[8][8], int a)
{
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			matrix[i][j]=a;
		}

	}
}

void generate_c(float (*c)[8][8],int i,int j)
{

	float ci, cj;

			// ci and cj depends on frequency as well as 
			// number of row and columns of specified matrix 
			if (i == 0) 
				ci = 1 / sqrt(m); 
			else
				ci = sqrt(2) / sqrt(m); 
			if (j == 0) 
				cj = 1 / sqrt(n); 
			else
				cj = sqrt(2) / sqrt(n); 
			(*c)[i][j]=ci*cj;
} 

void generate_sum(float (*dct)[8][8], int matrix[8][8],int i,int j)
{
	float sum,dct1;
	float cosine;
	float cosine1;
	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(100);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);
	tp0->run();
  	 rate = this->rate;


			for (int k = 0; k < m; k++) { 
				for (int l = 0; l < n; l++) { 

	float arg = (2 * k + 1) * i * pi / (2 * m);
	float arg1 = (2 * l + 1) * j * pi / (2 * n);
	
					int* anything = new int(0);
					auto call_num = new __fluid__<int>(anything);
	auto tpb__g1 = std::bind(&dctFluidX_E::dctCos, this,arg,&cosine,call_num->p);
				auto tp__g1 = tf->newTask<decltype(tpb__g1)>("dctCos0", {}, {}, tpb__g1);
				Guard*       g1 = gs->newGuard("dctCos0" , {  }, {}, tp__g1, {  });
		  guard_log.push_back(g1);
		g1->set_root();	
		int* anything_call_num2 = new int(0);
		auto call_num2 = new __fluid__<int>(anything_call_num2);
		auto v0 = v1.init(call_num, 10);
				
				auto tpb__g2 = std::bind(&dctFluidX_E::dctCos, this,arg1,&cosine1,call_num2->p );
				auto tp__g2 = tf->newTask<decltype(tpb__g2)>("Cos1",{}, {}, tpb__g2);
				Guard*       g2 = gs->newGuard("Cos1" , { v0 }, {}, tp__g2, {  });
		  guard_log.push_back(g2);
				
				g2->set_leaf();
				gs->synctask(tp__g2);
cosine = cos(arg);
cosine1 = cos(arg1);
			dct1 = matrix[k][l] * cosine* cosine1; 
					sum = sum + dct1; 
				} 
			} 
			gs->sync(tp0);	
			(*dct)[i][j]=sum;
}

void dct_transformation(float (&fin_dct)[8][8], float c[8][8], float dct[8][8])
{
	std::string outputFilename 	=  "dctfluid8.txt";

	// prepare the output file for writting the theta values
	std::ofstream outputFileHandler;
	outputFileHandler.open(outputFilename);
	outputFileHandler.precision(5);


	for (int i = 0; i < m; i++) { 
		for (int j = 0; j < n; j++) { 
			fin_dct[i][j]=c[i][j]*dct[i][j];
		}
	}
	std::cout<<"$######################Printing Results###########################";
	for (int i = 0; i < m; i++) { 
		for (int j = 0; j < n; j++) { 

			outputFileHandler << fin_dct[i][j] << " " ;
		}
		outputFileHandler<<std::endl;
	}
	outputFileHandler.close();


}

void dct_calc()
{

float c[8][8];
float dct[8][8];

generate_matrix(matrix,a);

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(100);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);
	tp0->run();
  	 rate = this->rate;
	for (int i = 0; i < m; i++) { 
		for (int j = 0; j < n; j++) { 

/*			int* anything = new int(0);
			auto call_num = new __fluid__<int>(anything);
			auto tpb__g1 = std::bind(&dctFluidX_E::generate_sum, this,&dct,matrix,i,j);	
			auto  tp__g1= tf->newTask<decltype(tpb__g1)>("generate_sum",tpb__g1);
			Guard*  g1 = gs->newGuard("generate_sum" + std::to_string(i), {  }, tp__g1, {  });

			g1->set_root();	
			int* anything_call_num2 = new int(0);
			auto call_num2 = new __fluid__<int>(anything_call_num2);
			auto v0 = v1.init(call_num, 0);
			auto tpb__g2 = std::bind(&dctFluidX_E::generate_c, this,&c,i,j , call_num2->p);
			auto tp__g2 = tf->newTask<decltype(tpb__g2)>("generate_c" + std::to_string(i), tpb__g2);
			Guard*       g2 = gs->newGuard("generate_c" + std::to_string(i), { v0 }, {}, tp__g2, {  });

			g2->set_leaf();
			gs->synctask(tp__g2);
*/
		generate_sum(&dct,matrix,i,j);
		generate_c(&c,i,j);	
}
}
			gs->sync(tp0);	

float fin[8][8];
dct_transformation(fin,c,dct);

}
void no_fluid()
{
float c[8][8];
float dct[8][8];

generate_matrix(matrix,a);
int i, j, k, l; 
float ci, cj, dct1, sum; 

	for (i = 0; i < m; i++) { 
		for (j = 0; j < n; j++) { 

			// ci and cj depends on frequency as well as 
			// number of row and columns of specified matrix 
			if (i == 0) 
				ci = 1 / sqrt(m); 
			else
				ci = sqrt(2) / sqrt(m); 
			if (j == 0) 
				cj = 1 / sqrt(n); 
			else
				cj = sqrt(2) / sqrt(n); 

			// sum will temporarily store the sum of 
			// cosine signals 
			sum = 0; 
			for (k = 0; k < m; k++) { 
				for (l = 0; l < n; l++) { 
					dct1 = matrix[k][l] * 
						cos((2 * k + 1) * i * pi / (2 * m)) * 
						cos((2 * l + 1) * j * pi / (2 * n)); 
					sum = sum + dct1; 
				} 
			} 
			dct[i][j] = ci * cj * sum; 
		} 
	} 

	for (i = 0; i < m; i++) { 
		for (j = 0; j < n; j++) { 
			printf("%f\t", dct[i][j]); 
		} 
		printf("\n"); 
	} 
} 


};
}

using namespace example;
